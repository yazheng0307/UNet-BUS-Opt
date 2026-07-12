import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ThesisFourStageUNet import ThesisFourStageUNet
from train_busi_four_module import FourModuleLoss, run_epoch
from train_busi_opt import StrongBUSITransform, seed_everything
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train progressive thesis ablations on BUSI split 3")
    parser.add_argument("--variant", choices=tuple(ThesisFourStageUNet.VARIANTS), required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--new_module", choices=("none", "A", "B", "C", "D"), required=True)
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--base_lr", type=float, default=2e-5)
    parser.add_argument("--module_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def module_prefix(name):
    return {
        "none": (),
        "A": ("module_a.",),
        "B": ("module_b.",),
        "C": ("module_c.", "boundary_head.", "distance_head."),
        "D": ("module_d.", "auxiliary_heads."),
    }[name]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    train_dataset = MedicalDataSets(
        base_dir=args.base_dir, split="train", transform=StrongBUSITransform(args.img_size),
        train_file_dir=args.train_file, val_file_dir=args.val_file,
    )
    val_dataset = MedicalDataSets(
        base_dir=args.base_dir, split="val", transform=ValTransform(args.img_size),
        train_file_dir=args.train_file, val_file_dir=args.val_file,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = ThesisFourStageUNet(args.variant, args.base_channels).to(device)
    loaded, missing, unexpected = model.load_progressive_checkpoint(args.init_checkpoint)
    prefixes = module_prefix(args.new_module)
    new_parameters = [
        parameter for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    new_ids = {id(parameter) for parameter in new_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in new_ids]
    groups = [{"params": base_parameters, "lr": args.base_lr}]
    if new_parameters:
        groups.append({"params": new_parameters, "lr": args.module_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.base_lr * 0.05
    )
    criterion = FourModuleLoss(
        boundary_weight=0.10,
        distance_weight=0.10,
        error_weight=0.10,
        coarse_weight=0.05,
        deep_supervision_scale=0.25,
    ).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    metric_names = (
        "loss", "region_loss", "boundary_loss", "distance_loss",
        "dual_error_loss", "iou", "dice",
    )
    fields = ["epoch"] + [
        prefix + name for prefix in ("train_", "val_") for name in metric_names
    ] + ["base_lr", "module_lr"]
    best_iou = -1.0
    best_dice = -1.0
    best_epoch = 0
    started = time.time()
    history_path = os.path.join(args.output_dir, "history.csv")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "base_lr": optimizer.param_groups[0]["lr"],
                "module_lr": optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            }
            row.update({"train_" + key: value for key, value in train_metrics.items()})
            row.update({"val_" + key: value for key, value in val_metrics.items()})
            writer.writerow(row)
            file.flush()
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_dice = val_metrics["dice"]
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()
            print(
                "epoch [{}/{}] val_IoU={:.4f} val_Dice={:.4f}".format(
                    epoch, args.epochs, val_metrics["iou"], val_metrics["dice"]
                ), flush=True,
            )
    summary = {
        "variant": args.variant,
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "best_val_dice": best_dice,
        "loaded_tensors": loaded,
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
