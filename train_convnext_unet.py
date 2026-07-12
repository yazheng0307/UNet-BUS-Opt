import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ConvNeXtUNet import ConvNeXtTinyUNet
from train_busi_opt import BCEDiceLoss, StrongBUSITransform, run_epoch, seed_everything
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny U-Net on BUSI split 3")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/convnext_tiny_unet_split3")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


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
    model = ConvNeXtTinyUNet(pretrained=True).to(device)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": decoder_parameters, "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.encoder_lr * 0.05
    )
    criterion = BCEDiceLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    fields = [
        "epoch", "train_loss", "train_iou", "train_dice",
        "val_loss", "val_iou", "val_dice", "encoder_lr", "decoder_lr",
    ]
    best_iou = -1.0
    best_epoch = 0
    started = time.time()
    history_path = os.path.join(args.output_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "decoder_lr": optimizer.param_groups[1]["lr"],
            }
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
                "epoch [{}/{}] val_IoU={:.4f} val_Dice={:.4f}".format(
                    epoch, args.epochs, val_metrics["iou"], val_metrics["dice"]
                ),
                flush=True,
            )
    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
