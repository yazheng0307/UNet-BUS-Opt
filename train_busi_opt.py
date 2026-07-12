import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.BUSIOptUNet import build_busi_opt_unet
from train_unet import ValTransform


class StrongBUSITransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        if random.random() < 0.5:
            axis = random.choice((0, 1, -1))
            image = cv2.flip(image, axis)
            mask = cv2.flip(mask, axis)
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        image = cv2.resize(np.ascontiguousarray(image), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if random.random() < 0.6:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.9, 1.1)
            center = (self.size / 2.0, self.size / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            matrix[:, 2] += np.array([
                random.uniform(-0.05, 0.05) * self.size,
                random.uniform(-0.05, 0.05) * self.size,
            ])
            image = cv2.warpAffine(
                image, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            mask = cv2.warpAffine(
                mask, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            )
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20.0, 20.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            gamma = random.uniform(0.75, 1.35)
            lookup = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8)
            image = cv2.LUT(image, lookup)
        if random.random() < 0.25:
            sigma = random.uniform(2.0, 10.0)
            noise = np.random.normal(0.0, sigma, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if mask.ndim == 2:
            mask = mask[..., None]
        return {"image": np.ascontiguousarray(image), "mask": np.ascontiguousarray(mask)}


class BCEDiceLoss(nn.Module):
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        intersection = (probability * target).sum(1)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (probability.sum(1) + target.sum(1) + 1e-5)).mean()
        return 0.5 * bce + dice_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train BUSI chapter-1 ablations on split 3")
    parser.add_argument(
        "--variant",
        choices=("A0", "A1", "A2", "A2R", "A2G", "A3", "A1V2", "A3V2", "A1V4", "A3V4"),
        default="A3",
    )
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_overlap_metrics(logits, target):
    prediction = torch.sigmoid(logits) >= 0.5
    target = target >= 0.5
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    intersection = (prediction & target).sum(1).float()
    union = (prediction | target).sum(1).float()
    predicted = prediction.sum(1).float()
    actual = target.sum(1).float()
    iou = (intersection + 1e-5) / (union + 1e-5)
    dice = (2 * intersection + 1e-5) / (predicted + actual + 1e-5)
    return iou, dice, intersection.sum(), union.sum(), predicted.sum(), actual.sum()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "samples": 0}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None and scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss detected")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        iou, dice, intersection, union, predicted, actual = batch_overlap_metrics(logits, labels)
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["iou"] += iou.sum().item()
        totals["dice"] += dice.sum().item()
        totals["samples"] += batch_size
    samples = totals.pop("samples")
    return {key: value / samples for key, value in totals.items()}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if args.output_dir is None:
        args.output_dir = "./runs/busi_opt_ch1/{}".format(args.variant)
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

    model = build_busi_opt_unet(args.variant, args.base_channels).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_iou = -1.0
    best_epoch = 0
    history_path = os.path.join(args.output_dir, "history.csv")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    fields = [
        "epoch", "train_loss", "train_iou", "train_dice",
        "val_loss", "val_iou", "val_dice", "lr",
    ]
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
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
