import argparse
import hashlib
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Audit BUSI split files for exact and near-duplicate leakage")
    parser.add_argument("--base_dir", default="./data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--phash_distance", type=int, default=4)
    parser.add_argument("--correlation_threshold", type=float, default=0.98)
    parser.add_argument("--output", default="./runs/busi_split3_audit.json")
    return parser.parse_args()


def read_cases(path):
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def load_image(path, flags):
    image = cv2.imread(path, flags)
    if image is None:
        raise RuntimeError("Could not read {}".format(path))
    return image


def pixel_digest(image):
    header = np.asarray(image.shape, dtype=np.int32).tobytes()
    return hashlib.sha256(header + np.ascontiguousarray(image).tobytes()).hexdigest()


def perceptual_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(gray)[:8, :8]
    median = np.median(low_frequency[1:])
    bits = (low_frequency > median).reshape(-1)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def normalized_thumbnail(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    return thumb / max(float(norm), 1e-6)


def grouped_duplicates(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record["case"])
    return [sorted(cases) for cases in groups.values() if len(cases) > 1]


def crosses_split(cases, train_set, val_set):
    return bool(set(cases) & train_set) and bool(set(cases) & val_set)


def main():
    args = parse_args()
    train_path = os.path.join(args.base_dir, args.train_file)
    val_path = os.path.join(args.base_dir, args.val_file)
    train_cases = read_cases(train_path)
    val_cases = read_cases(val_path)
    train_set = set(train_cases)
    val_set = set(val_cases)
    all_cases = sorted(train_set | val_set)

    records = []
    thumbnails = {}
    normalized_masks = {}
    for case in all_cases:
        image_path = os.path.join(args.base_dir, "images", case + ".png")
        mask_path = os.path.join(args.base_dir, "masks", "0", case + ".png")
        image = load_image(image_path, cv2.IMREAD_COLOR)
        mask = load_image(mask_path, cv2.IMREAD_GRAYSCALE)
        records.append(
            {
                "case": case,
                "split": "train" if case in train_set else "val",
                "image_sha256": pixel_digest(image),
                "mask_sha256": pixel_digest(mask),
                "phash": perceptual_hash(image),
            }
        )
        thumbnails[case] = normalized_thumbnail(image)
        normalized_masks[case] = cv2.resize(
            mask, (256, 256), interpolation=cv2.INTER_NEAREST
        ) >= 128

    exact_image_groups = grouped_duplicates(records, "image_sha256")
    exact_mask_groups = grouped_duplicates(records, "mask_sha256")
    near_pairs = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            distance = bin(left["phash"] ^ right["phash"]).count("1")
            if distance > args.phash_distance:
                continue
            correlation = float((thumbnails[left["case"]] * thumbnails[right["case"]]).sum())
            left_mask = normalized_masks[left["case"]]
            right_mask = normalized_masks[right["case"]]
            mask_intersection = np.logical_and(left_mask, right_mask).sum()
            mask_union = np.logical_or(left_mask, right_mask).sum()
            mask_iou = float((mask_intersection + 1e-5) / (mask_union + 1e-5))
            near_pairs.append(
                {
                    "left": left["case"],
                    "right": right["case"],
                    "phash_distance": distance,
                    "thumbnail_correlation": correlation,
                    "mask_iou_after_resize": mask_iou,
                    "cross_split": (left["split"] != right["split"]),
                    "exact_image": left["image_sha256"] == right["image_sha256"],
                }
            )
    near_pairs.sort(key=lambda item: (item["phash_distance"], -item["thumbnail_correlation"]))

    likely_pairs = [
        pair for pair in near_pairs
        if pair["thumbnail_correlation"] >= args.correlation_threshold
    ]
    report = {
        "base_dir": os.path.abspath(args.base_dir),
        "train_file": args.train_file,
        "val_file": args.val_file,
        "train_count": len(train_cases),
        "val_count": len(val_cases),
        "unique_case_count": len(all_cases),
        "case_name_overlap": sorted(train_set & val_set),
        "exact_image_duplicate_groups": exact_image_groups,
        "cross_split_exact_image_groups": [
            group for group in exact_image_groups if crosses_split(group, train_set, val_set)
        ],
        "exact_mask_duplicate_groups": exact_mask_groups,
        "cross_split_exact_mask_groups": [
            group for group in exact_mask_groups if crosses_split(group, train_set, val_set)
        ],
        "near_image_pairs": near_pairs,
        "cross_split_near_image_pairs": [pair for pair in near_pairs if pair["cross_split"]],
        "likely_near_image_pairs": likely_pairs,
        "cross_split_likely_near_image_pairs": [pair for pair in likely_pairs if pair["cross_split"]],
        "phash_distance_threshold": args.phash_distance,
        "thumbnail_correlation_threshold": args.correlation_threshold,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)

    print("train={} val={} unique={}".format(len(train_cases), len(val_cases), len(all_cases)))
    print("case-name overlap: {}".format(len(report["case_name_overlap"])))
    print("exact image duplicate groups: {}".format(len(exact_image_groups)))
    print("cross-split exact image groups: {}".format(len(report["cross_split_exact_image_groups"])))
    print("near-image candidates: {}".format(len(near_pairs)))
    print("cross-split near-image candidates: {}".format(len(report["cross_split_near_image_pairs"])))
    print("high-confidence near-image pairs: {}".format(len(likely_pairs)))
    print("cross-split high-confidence near-image pairs: {}".format(
        len(report["cross_split_likely_near_image_pairs"])
    ))
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
