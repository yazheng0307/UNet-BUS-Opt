import argparse
import csv
import json
import os

import cv2
import torch
from torch.utils.data import DataLoader

from evaluate_busi_opt import METRIC_NAMES, confusion_metrics
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ResNet50FourModuleUNet import ResNet50FourModuleUNet
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the ResNet50 four-module model")
    parser.add_argument("--variant", choices=("G", "BD", "DS", "U", "GBDU"), default="GBDU")
    parser.add_argument(
        "--model_path", default="./runs/resnet50_four_module_split3/best_model.pth"
    )
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument(
        "--output_dir", default="./runs/resnet50_four_module_split3/evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_predictions", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
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
    variants = {
        "G": (True, False, False, False),
        "BD": (False, True, False, False),
        "DS": (False, False, False, True),
        "U": (False, False, True, True),
        "GBDU": (True, True, True, True),
    }
    model = ResNet50FourModuleUNet(
        encoder_checkpoint=None,
        use_graph=variants[args.variant][0],
        use_geometry=variants[args.variant][1],
        use_uder=variants[args.variant][2],
        use_deep_supervision=variants[args.variant][3],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
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
            sample_count += images.size(0)
            for name in METRIC_NAMES:
                totals[name] += metrics[name].sum().item()

            predictions = (torch.sigmoid(logits) >= args.threshold).byte().cpu().numpy()
            uncertainty_output = outputs["uncertainty"]
            uncertainty = uncertainty_output.cpu().numpy() if uncertainty_output is not None else None
            for item_index, dataset_index in enumerate(batch["idx"].tolist()):
                case = dataset.sample_list[dataset_index]
                row = {"case": case}
                for name in METRIC_NAMES:
                    row[name] = metrics[name][item_index].item()
                rows.append(row)
                if saved < args.save_predictions:
                    cv2.imwrite(
                        os.path.join(prediction_dir, case + ".png"),
                        predictions[item_index, 0] * 255,
                    )
                    if uncertainty is not None:
                        cv2.imwrite(
                            os.path.join(prediction_dir, case + "_uncertainty.png"),
                            (uncertainty[item_index, 0].clip(0.0, 1.0) * 255).astype("uint8"),
                        )
                    saved += 1

    results = {
        "checkpoint": os.path.abspath(args.model_path),
        "val_file": args.val_file,
        "samples": sample_count,
        "threshold": args.threshold,
        "per_image": {name: totals[name] / sample_count for name in METRIC_NAMES},
        "saved_predictions": saved,
    }
    with open(
        os.path.join(args.output_dir, "per_case_metrics.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=("case",) + METRIC_NAMES
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(
        os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
