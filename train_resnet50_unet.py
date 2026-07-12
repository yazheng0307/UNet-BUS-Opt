import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ResNet50UNet import DEFAULT_DEEPLAB_CHECKPOINT, ResNet50UNet
from train_busi_opt import BCEDiceLoss, StrongBUSITransform, run_epoch, seed_everything
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cached-pretrained ResNet50 U-Net on BUSI split 3")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--encoder_checkpoint", default=DEFAULT_DEEPLAB_CHECKPOINT)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./runs/resnet50_unet_split3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--freeze_encoder_bn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    model = ResNet50UNet(
        checkpoint_path=args.encoder_checkpoint,
        freeze_encoder_bn=args.freeze_encoder_bn,
    ).to(device)
    initialized_tensors = 0
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        initialized_tensors = len(state)
        print("initialized segmentation checkpoint tensors={}".format(initialized_tensors), flush=True)
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    best_iou = -1.0
    best_epoch = 0
    fields = [
        "epoch", "train_loss", "train_iou", "train_dice",
        "val_loss", "val_iou", "val_dice", "lr",
    ]
    started = time.time()
    with open(os.path.join(args.output_dir, "history.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, criterion, device)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({"train_" + key: value for key, value in train_metrics.items()})
            row.update({"val_" + key: value for key, value in val_metrics.items()})
            writer.writerow(row)
            file.flush()
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            scheduler.step()
            print(
                "epoch [{}/{}] train_loss={:.4f} val_loss={:.4f} val_IoU={:.4f} "
                "val_Dice={:.4f}".format(
                    epoch, args.epochs, train_metrics["loss"], val_metrics["loss"], val_metrics["iou"],
                    val_metrics["dice"]
                ),
                flush=True,
            )
    summary = {
        "best_epoch": best_epoch,
        "best_val_per_image_iou": best_iou,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loaded_encoder_tensors": model.loaded_encoder_tensors,
        "initialized_segmentation_tensors": initialized_tensors,
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
