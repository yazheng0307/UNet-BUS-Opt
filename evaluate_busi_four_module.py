import argparse
import csv
import json
import os

import cv2
import torch
from torch.utils.data import DataLoader

from evaluate_busi_opt import METRIC_NAMES, confusion_metrics
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.BUSIFourModuleUNet import build_busi_four_module_unet
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a BUSI four-module checkpoint")
    parser.add_argument(
        "--variant",
        choices=("B0", "B1", "B1D", "B2", "B2V2", "B3", "B3V2", "FG", "FGV2", "FR", "F", "FLEGACY"),
        default="B3",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_predictions", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if args.model_path is None:
        args.model_path = "./runs/busi_four_module/{}/best_model.pth".format(args.variant)
    if args.output_dir is None:
        args.output_dir = "./runs/busi_four_module/{}/evaluation".format(args.variant)
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    prediction_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    dataset = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=ValTransform(args.img_size),
        val_file_dir=args.val_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = build_busi_four_module_unet(args.variant, args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    totals = {name: 0.0 for name in METRIC_NAMES}
    rows = []
    sample_count = 0
    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["segmentation"]
            metrics = confusion_metrics(logits, labels, args.threshold)
            batch_size = images.size(0)
            sample_count += batch_size
            for name in METRIC_NAMES:
                totals[name] += metrics[name].sum().item()

            predictions = (torch.sigmoid(logits) >= args.threshold).byte().cpu().numpy()
            uncertainty = outputs["uncertainty"]
            uncertainty = uncertainty.cpu().numpy() if uncertainty is not None else None
            for item_index, dataset_index in enumerate(batch["idx"].tolist()):
                case = dataset.sample_list[dataset_index]
                row = {"case": case}
                for name in METRIC_NAMES:
                    row[name] = metrics[name][item_index].item()
                rows.append(row)
                if saved < args.save_predictions:
                    cv2.imwrite(os.path.join(prediction_dir, case + ".png"), predictions[item_index, 0] * 255)
                    if uncertainty is not None:
                        cv2.imwrite(
                            os.path.join(prediction_dir, case + "_uncertainty.png"),
                            (uncertainty[item_index, 0].clip(0.0, 1.0) * 255).astype("uint8"),
                        )
                    saved += 1

    per_image = {name: totals[name] / sample_count for name in METRIC_NAMES}
    results = {
        "variant": args.variant,
        "checkpoint": os.path.abspath(args.model_path),
        "val_file": args.val_file,
        "samples": sample_count,
        "threshold": args.threshold,
        "per_image": per_image,
        "saved_predictions": saved,
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
