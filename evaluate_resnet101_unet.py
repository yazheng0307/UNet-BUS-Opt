import argparse
import csv
import json
import os
import torch
from torch.utils.data import DataLoader
from evaluate_busi_opt import METRIC_NAMES, confusion_metrics
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ResNet101UNet import ResNet101UNet
from train_unet import ValTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./runs/resnet101_unet_split3/best_model.pth")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/resnet101_unet_split3/evaluation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = MedicalDataSets(base_dir=args.base_dir, split="val",
        transform=ValTransform(args.img_size), val_file_dir=args.val_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    device = torch.device("cuda")
    model = ResNet101UNet(checkpoint_path=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    rows = []
    samples = 0
    with torch.no_grad():
        for batch in loader:
            metrics = confusion_metrics(model(batch["image"].to(device)), batch["label"].to(device), 0.5)
            samples += len(batch["idx"])
            for name in METRIC_NAMES:
                totals[name] += metrics[name].sum().item()
            for i, index in enumerate(batch["idx"].tolist()):
                rows.append({"case": dataset.sample_list[index],
                             "iou": metrics["iou"][i].item(), "dice": metrics["dice"][i].item()})
    results = {"checkpoint": os.path.abspath(args.model_path), "val_file": args.val_file,
               "samples": samples, "threshold": 0.5,
               "iou": totals["iou"] / samples, "dice": totals["dice"] / samples}
    with open(os.path.join(args.output_dir, "per_case_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("case", "iou", "dice")); writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
