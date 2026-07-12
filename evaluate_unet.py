import argparse
import json
import os

import cv2
import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.U_Net import U_Net
from src.utils.metrics import iou_score
from train_unet import ValTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint")
    parser.add_argument("--model_path", default="./runs/unet_busi_split3/U_Net_model.pth")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="./runs/unet_busi_split3/evaluation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_predictions", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
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
    model = U_Net(output_ch=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    totals = {"iou": 0.0, "dice": 0.0}
    sample_count = 0
    saved = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            metrics = iou_score(outputs, labels)
            batch_size = images.size(0)
            values = {
                "iou": metrics[0],
                "dice": metrics[1],
            }
            for key, value in values.items():
                totals[key] += value * batch_size
            sample_count += batch_size

            predictions = (torch.sigmoid(outputs) > 0.5).byte().cpu().numpy()
            for item, dataset_index in zip(predictions, batch["idx"].tolist()):
                if saved >= args.save_predictions:
                    break
                case = dataset.sample_list[dataset_index]
                cv2.imwrite(os.path.join(prediction_dir, case + ".png"), item[0] * 255)
                saved += 1

    results = {key: value / sample_count for key, value in totals.items()}
    results.update({
        "samples": sample_count,
        "checkpoint": os.path.abspath(args.model_path),
        "val_file": args.val_file,
        "saved_predictions": saved,
    })
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
