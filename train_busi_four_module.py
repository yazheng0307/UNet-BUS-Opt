import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.BUSIFourModuleUNet import build_busi_four_module_unet
from train_busi_opt import StrongBUSITransform, batch_overlap_metrics, seed_everything
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train BUSI four-module ablations on split 3")
    parser.add_argument(
        "--variant",
        choices=("B0", "B1", "B1D", "B2", "B2V2", "B3", "B3V2", "FG", "FGV2", "FR", "F", "FLEGACY"),
        default="B3",
    )
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--boundary_weight", type=float, default=0.20)
    parser.add_argument("--distance_weight", type=float, default=0.15)
    parser.add_argument("--error_weight", type=float, default=0.15)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def region_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits).flatten(1)
    flat_target = target.flatten(1)
    intersection = (probability * flat_target).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1e-5) / (
        probability.sum(1) + flat_target.sum(1) + 1e-5
    )).mean()
    return 0.5 * bce + dice


def morphological_boundary(target):
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def signed_distance_target(target):
    masks = (target.detach().cpu().numpy()[:, 0] >= 0.5).astype(np.uint8)
    distances = []
    for mask in masks:
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        signed = inside - outside
        scale = max(float(np.abs(signed).max()), 1.0)
        distances.append(signed / scale)
    distance = torch.from_numpy(np.stack(distances)[:, None]).to(device=target.device, dtype=target.dtype)
    return distance


class FourModuleLoss(nn.Module):
    def __init__(
        self,
        boundary_weight,
        distance_weight,
        error_weight,
        coarse_weight=0.20,
        deep_supervision_scale=1.0,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.distance_weight = distance_weight
        self.error_weight = error_weight
        self.coarse_weight = coarse_weight
        self.deep_supervision_scale = deep_supervision_scale

    def forward(self, outputs, target):
        segmentation = outputs["segmentation"]
        components = {"region": region_loss(segmentation, target)}
        total = components["region"]

        coarse = outputs["coarse_segmentation"]
        if coarse is not segmentation:
            components["coarse"] = region_loss(coarse, target)
            total = total + self.coarse_weight * components["coarse"]
        else:
            components["coarse"] = segmentation.new_zeros(())

        auxiliary_loss = segmentation.new_zeros(())
        auxiliary_weights = (0.10, 0.15, 0.20)
        for weight, auxiliary in zip(auxiliary_weights, outputs["auxiliary_segmentations"]):
            resized_target = F.interpolate(target, size=auxiliary.shape[-2:], mode="nearest")
            auxiliary_loss = auxiliary_loss + weight * region_loss(auxiliary, resized_target)
        components["deep_supervision"] = auxiliary_loss
        total = total + self.deep_supervision_scale * auxiliary_loss

        boundary_logits = outputs["boundary_logits"]
        if boundary_logits:
            boundary_target = morphological_boundary(target)
            boundary_loss = segmentation.new_zeros(())
            for boundary_logit in boundary_logits:
                resized_boundary = F.interpolate(boundary_target, size=boundary_logit.shape[-2:], mode="nearest")
                resized_segmentation = F.interpolate(
                    segmentation.detach(), size=boundary_logit.shape[-2:], mode="bilinear", align_corners=False
                )
                probability = torch.sigmoid(resized_segmentation)
                entropy = -probability * torch.log(probability.clamp_min(1e-6))
                entropy -= (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-6))
                entropy = entropy / np.log(2.0)
                weights = (1.0 + 4.0 * resized_boundary) * (1.0 + 2.0 * entropy * resized_boundary)
                boundary_loss = boundary_loss + (
                    F.binary_cross_entropy_with_logits(boundary_logit, resized_boundary, reduction="none") * weights
                ).mean()
            components["boundary"] = boundary_loss / len(boundary_logits)

            if outputs["distance_maps"]:
                distance_target = signed_distance_target(target)
                distance_loss = segmentation.new_zeros(())
                for distance in outputs["distance_maps"]:
                    resized_distance = F.interpolate(
                        distance_target, size=distance.shape[-2:], mode="bilinear", align_corners=False
                    )
                    distance_loss = distance_loss + F.smooth_l1_loss(distance, resized_distance)
                components["distance"] = distance_loss / len(outputs["distance_maps"])
            else:
                components["distance"] = segmentation.new_zeros(())
            total = total + self.boundary_weight * components["boundary"]
            total = total + self.distance_weight * components["distance"]
        else:
            components["boundary"] = segmentation.new_zeros(())
            components["distance"] = segmentation.new_zeros(())

        false_negative = outputs["false_negative_logits"]
        false_positive = outputs["false_positive_logits"]
        if false_negative is not None:
            coarse_probability = torch.sigmoid(coarse).detach()
            if outputs["uder_version"] >= 2:
                predicted_positive = coarse_probability >= 0.5
                target_positive = target >= 0.5
                fn_target = (target_positive & ~predicted_positive).to(dtype=target.dtype)
                fp_target = (~target_positive & predicted_positive).to(dtype=target.dtype)
                fn_weight = 1.0 + 12.0 * fn_target
                fp_weight = 1.0 + 12.0 * fp_target
            else:
                fn_target = target * (1.0 - coarse_probability)
                fp_target = (1.0 - target) * coarse_probability
                fn_weight = 1.0 + 4.0 * fn_target
                fp_weight = 1.0 + 4.0 * fp_target
            fn_loss = (
                F.binary_cross_entropy_with_logits(false_negative, fn_target, reduction="none") * fn_weight
            ).mean()
            fp_loss = (
                F.binary_cross_entropy_with_logits(false_positive, fp_target, reduction="none") * fp_weight
            ).mean()
            components["dual_error"] = 0.5 * (fn_loss + fp_loss)
            total = total + self.error_weight * components["dual_error"]
        else:
            components["dual_error"] = segmentation.new_zeros(())
        components["total"] = total
        return total, components


def load_backbone_initialization(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    current = model.backbone.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    decoder_tensors = 0
    if model.boundary_decoders is not None:
        boundary_state = model.boundary_decoders.state_dict()
        decoder_compatible = {}
        for key, value in state.items():
            if not key.startswith("decoders."):
                continue
            boundary_key = key[len("decoders."):]
            if boundary_key in boundary_state and boundary_state[boundary_key].shape == value.shape:
                decoder_compatible[boundary_key] = value
        model.boundary_decoders.load_state_dict(decoder_compatible, strict=False)
        decoder_tensors = len(decoder_compatible)
    print(
        "initialized backbone tensors={} decoder tensors={} missing={} unexpected={}".format(
            len(compatible), decoder_tensors, len(missing), len(unexpected)
        ),
        flush=True,
    )
    return len(compatible) + decoder_tensors


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "region_loss": 0.0,
        "boundary_loss": 0.0,
        "distance_loss": 0.0,
        "dual_error_loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "samples": 0,
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None and scaler.is_enabled()):
                outputs = model(images)
                loss, components = criterion(outputs, labels)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss detected")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        iou, dice, intersection, union, predicted, actual = batch_overlap_metrics(
            outputs["segmentation"], labels
        )
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["region_loss"] += components["region"].item() * batch_size
        totals["boundary_loss"] += components["boundary"].item() * batch_size
        totals["distance_loss"] += components["distance"].item() * batch_size
        totals["dual_error_loss"] += components["dual_error"].item() * batch_size
        totals["iou"] += iou.sum().item()
        totals["dice"] += dice.sum().item()
        totals["samples"] += batch_size
    samples = totals.pop("samples")
    result = {key: value / samples for key, value in totals.items()}
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if args.output_dir is None:
        args.output_dir = "./runs/busi_four_module/{}".format(args.variant)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda")

    train_dataset = MedicalDataSets(
        base_dir=args.base_dir,
        split="train",
        transform=StrongBUSITransform(args.img_size),
        train_file_dir=args.train_file,
        val_file_dir=args.val_file,
    )
    val_dataset = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=ValTransform(args.img_size),
        train_file_dir=args.train_file,
        val_file_dir=args.val_file,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = build_busi_four_module_unet(args.variant, args.base_channels).to(device)
    initialized_tensors = 0
    if args.init_checkpoint:
        initialized_tensors = load_backbone_initialization(model, args.init_checkpoint)
    criterion = FourModuleLoss(args.boundary_weight, args.distance_weight, args.error_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_iou = -1.0
    best_epoch = 0
    history_path = os.path.join(args.output_dir, "history.csv")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    metric_names = (
        "loss", "region_loss", "boundary_loss", "distance_loss", "dual_error_loss",
        "iou", "dice",
    )
    fields = ["epoch"] + [prefix + name for prefix in ("train_", "val_") for name in metric_names] + ["lr"]
    started = time.time()
    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, criterion, device)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({"train_" + key: value for key, value in train_metrics.items()})
            row.update({"val_" + key: value for key, value in val_metrics.items()})
            writer.writerow(row)
            history_file.flush()
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()
            print(
                "epoch [{}/{}] train_loss={:.4f} val_loss={:.4f} "
                "val_IoU={:.4f} val_Dice={:.4f}".format(
                    epoch,
                    args.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["iou"],
                    val_metrics["dice"],
                ),
                flush=True,
            )
    summary = {
        "variant": args.variant,
        "best_epoch": best_epoch,
        "best_val_per_image_iou": best_iou,
        "parameters": parameter_count,
        "initialized_backbone_tensors": initialized_tensors,
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
