import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

from evaluate_busi_opt import METRIC_NAMES, confusion_metrics
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ConvNeXtUNet import ConvNeXtTinyUNet
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-Tiny U-Net")
    parser.add_argument("--model_path", default="./runs/convnext_tiny_unet_split3/best_model.pth")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/convnext_tiny_unet_split3/evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = MedicalDataSets(
        base_dir=args.base_dir, split="val", transform=ValTransform(args.img_size),
        val_file_dir=args.val_file,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = ConvNeXtTinyUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    rows = []
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            metrics = confusion_metrics(model(images), labels, args.threshold)
            sample_count += images.size(0)
            for name in METRIC_NAMES:
                totals[name] += metrics[name].sum().item()
            for item_index, dataset_index in enumerate(batch["idx"].tolist()):
                row = {"case": dataset.sample_list[dataset_index]}
                for name in METRIC_NAMES:
                    row[name] = metrics[name][item_index].item()
                rows.append(row)
    results = {
        "checkpoint": os.path.abspath(args.model_path),
        "val_file": args.val_file,
        "samples": sample_count,
        "threshold": args.threshold,
        "iou": totals["iou"] / sample_count,
        "dice": totals["dice"] / sample_count,
    }
    with open(os.path.join(args.output_dir, "per_case_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("case",) + METRIC_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
