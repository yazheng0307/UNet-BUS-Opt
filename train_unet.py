import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.U_Net import U_Net
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import iou_score


class TrainTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if random.random() < 0.5:
            axis = random.choice((0, 1))
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)
        return resize_pair(image, mask, self.size)


class ValTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        return resize_pair(image, mask, self.size)


def resize_pair(image, mask, size):
    image = cv2.resize(np.ascontiguousarray(image), (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(np.ascontiguousarray(mask), (size, size), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 2:
        mask = mask[..., None]
    return {"image": image, "mask": mask}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on the BUSI dataset")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/unet_busi_split3")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, lr_state=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "samples": 0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None and scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_state["iteration"] += 1
                lr = lr_state["base_lr"] * (
                    1.0 - lr_state["iteration"] / lr_state["max_iterations"]
                ) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr

        iou, dice, _, _, _, _, _ = iou_score(outputs, labels)
        totals["loss"] += loss.item() * batch_size
        totals["iou"] += iou * batch_size
        totals["dice"] += dice * batch_size
        totals["samples"] += batch_size

    samples = totals.pop("samples")
    return {key: value / samples for key, value in totals.items()}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    train_dataset = MedicalDataSets(
        base_dir=args.base_dir,
        split="train",
        transform=TrainTransform(args.img_size),
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
    train_loader = make_loader(train_dataset, args.batch_size, True, args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.workers)

    model = U_Net(output_ch=1).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    amp_enabled = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    lr_state = {
        "iteration": 0,
        "base_lr": args.lr,
        "max_iterations": len(train_loader) * args.epochs,
    }
    start_epoch = 0
    best_iou = 0.0
    last_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    best_path = os.path.join(args.output_dir, "U_Net_model.pth")
    history_path = os.path.join(args.output_dir, "history.csv")

    if args.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        lr_state["iteration"] = state["iteration"]

    write_header = start_epoch == 0 or not os.path.isfile(history_path)
    started = time.time()
    with open(history_path, "a", newline="", encoding="utf-8") as history_file:
        fieldnames = ["epoch", "train_loss", "train_iou", "train_dice", "val_loss", "val_iou", "val_dice", "lr"]
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, lr_state)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, criterion, device)
            row = {
                "epoch": epoch + 1,
                **{"train_" + key: value for key, value in train_metrics.items()},
                **{"val_" + key: value for key, value in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"epoch [{epoch + 1}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['iou']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['iou']:.4f} "
                f"val_dice={val_metrics['dice']:.4f}",
                flush=True,
            )

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_iou": best_iou,
                    "iteration": lr_state["iteration"],
                    "args": vars(args),
                },
                last_path,
            )

    summary = {
        "best_val_iou": best_iou,
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - started,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
