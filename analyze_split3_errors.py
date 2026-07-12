import argparse
import csv
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.U_Net import U_Net
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Audit split 3 U-Net errors per case")
    parser.add_argument("--model_path", default="./runs/unet_busi_split3/U_Net_model.pth")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/unet_busi_split3/error_analysis")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_metrics(probability, target, threshold):
    prediction = probability >= threshold
    target = target >= 0.5
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "iou": iou,
        "dice": dice,
        "fp_fraction": safe_div(fp, tp + fp),
        "fn_fraction": safe_div(fn, tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def image_characteristics(image, mask):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask_u8 = mask.astype(np.uint8)
    area = int(mask_u8.sum())
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    complexity = safe_div(perimeter * perimeter, 4 * math.pi * area)
    kernel = np.ones((9, 9), dtype=np.uint8)
    ring = np.logical_and(cv2.dilate(mask_u8, kernel, iterations=2) > 0, mask_u8 == 0)
    inside_mean = float(gray[mask].mean()) if area else 0.0
    outside_mean = float(gray[ring].mean()) if ring.any() else float(gray[~mask].mean())
    contrast = abs(inside_mean - outside_mean) / 255.0
    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    boundary_gradient = float(gradient[boundary].mean() / 255.0) if boundary.any() else 0.0
    return {
        "lesion_ratio": safe_div(area, mask.size),
        "boundary_complexity": complexity,
        "local_contrast": contrast,
        "boundary_gradient": boundary_gradient,
        "gt_components": int(max(cv2.connectedComponents(mask_u8)[0] - 1, 0)),
    }


def uncertainty_features(probability, mask):
    eps = 1e-7
    entropy = -(probability * np.log(probability + eps) + (1 - probability) * np.log(1 - probability + eps))
    mask_u8 = mask.astype(np.uint8)
    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
    return {
        "mean_entropy": float(entropy.mean()),
        "boundary_entropy": float(entropy[boundary].mean()) if boundary.any() else 0.0,
    }


def aggregate_metrics(rows):
    keys = ("iou", "dice")
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tn = sum(row["tn"] for row in rows)
    return {"per_image_mean": mean}


def subgroup_summary(rows, feature, high_is_hard):
    values = np.array([row[feature] for row in rows])
    cutoff = float(np.quantile(values, 0.25 if not high_is_hard else 0.75))
    if high_is_hard:
        selected = [row for row in rows if row[feature] >= cutoff]
        rule = ">="
    else:
        selected = [row for row in rows if row[feature] <= cutoff]
        rule = "<="
    result = aggregate_metrics(selected)["per_image_mean"]
    result.update({"cases": len(selected), "cutoff": cutoff, "rule": rule})
    return result


def save_overlay(output_dir, case, image, target, prediction, iou):
    overlay = (image * 255).astype(np.uint8).copy()
    target_edge = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    pred_edge = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    overlay[target_edge] = (0, 255, 0)
    overlay[pred_edge] = (0, 0, 255)
    cv2.putText(overlay, "IoU={:.3f}".format(iou), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, case + ".png"), overlay)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for error analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    worst_dir = os.path.join(args.output_dir, "worst_cases")
    os.makedirs(worst_dir, exist_ok=True)

    dataset = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=ValTransform(args.img_size),
        val_file_dir=args.val_file,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda")
    model = U_Net(output_ch=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    rows = []
    cache = {}
    all_probabilities = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(images)).cpu().numpy()[:, 0]
            targets = batch["label"].numpy()[:, 0] >= 0.5
            images_np = batch["image"].numpy().transpose(0, 2, 3, 1)
            for probability, target, image, dataset_index in zip(
                    probabilities, targets, images_np, batch["idx"].tolist()):
                case = dataset.sample_list[dataset_index]
                row = {"case": case, "category": case.split(" ", 1)[0]}
                row.update(binary_metrics(probability, target, args.threshold))
                row.update(image_characteristics(image, target))
                row.update(uncertainty_features(probability, target))
                prediction = probability >= args.threshold
                row["pred_components"] = int(max(cv2.connectedComponents(prediction.astype(np.uint8))[0] - 1, 0))
                rows.append(row)
                cache[case] = (image, target, prediction)
                all_probabilities.append(probability)
                all_targets.append(target)

    thresholds = {}
    stacked_probabilities = np.stack(all_probabilities)
    stacked_targets = np.stack(all_targets)
    for threshold in np.arange(0.1, 0.91, 0.05):
        sweep_rows = [binary_metrics(p, t, threshold) for p, t in zip(stacked_probabilities, stacked_targets)]
        thresholds["{:.2f}".format(threshold)] = aggregate_metrics(sweep_rows)

    subgroups = {
        "small_lesion": subgroup_summary(rows, "lesion_ratio", high_is_hard=False),
        "low_contrast": subgroup_summary(rows, "local_contrast", high_is_hard=False),
        "complex_boundary": subgroup_summary(rows, "boundary_complexity", high_is_hard=True),
        "high_uncertainty": subgroup_summary(rows, "boundary_entropy", high_is_hard=True),
    }
    summary = {
        "checkpoint": os.path.abspath(args.model_path),
        "val_file": args.val_file,
        "threshold": args.threshold,
        "cases": len(rows),
        "overall": aggregate_metrics(rows),
        "subgroups": subgroups,
        "threshold_sweep": thresholds,
        "worst_cases": [
            {key: row[key] for key in ("case", "iou", "dice", "lesion_ratio", "local_contrast",
                                       "boundary_complexity", "fp_fraction", "fn_fraction")}
            for row in sorted(rows, key=lambda item: item["iou"])[:20]
        ],
    }

    fieldnames = list(rows[0].keys())
    with open(os.path.join(args.output_dir, "per_case_metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    for row in sorted(rows, key=lambda item: item["iou"])[:20]:
        image, target, prediction = cache[row["case"]]
        save_overlay(worst_dir, row["case"], image, target, prediction, row["iou"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
