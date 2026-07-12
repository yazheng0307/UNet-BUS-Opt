import argparse
import os
import shutil
import zipfile
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the Kaggle BUSI archive for this benchmark")
    parser.add_argument("--archive", default="./data/busi.zip")
    parser.add_argument("--output_dir", default="./data/busi")
    parser.add_argument("--split_dir", default="./data")
    return parser.parse_args()


def decode_png(archive, member, flags):
    data = np.frombuffer(archive.read(member), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError("Could not decode {}".format(member))
    return image


def main():
    args = parse_args()
    image_dir = os.path.join(args.output_dir, "images")
    mask_dir = os.path.join(args.output_dir, "masks", "0")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        png_members = [name for name in archive.namelist() if name.lower().endswith(".png")]
        images = {}
        masks = defaultdict(list)
        for member in png_members:
            filename = os.path.basename(member)
            stem = os.path.splitext(filename)[0]
            category = stem.split(" ", 1)[0].lower()
            if category not in ("benign", "malignant"):
                continue
            if "_mask" in stem:
                case = stem.split("_mask", 1)[0]
                masks[case].append(member)
            else:
                images[stem] = member

        missing_masks = sorted(set(images) - set(masks))
        if missing_masks:
            raise RuntimeError("Images without masks: {}".format(missing_masks[:10]))
        if len(images) != 647:
            raise RuntimeError("Expected 647 benign/malignant cases, found {}".format(len(images)))

        for index, case in enumerate(sorted(images), start=1):
            image = decode_png(archive, images[case], cv2.IMREAD_COLOR)
            merged_mask = None
            for member in masks[case]:
                mask = decode_png(archive, member, cv2.IMREAD_GRAYSCALE)
                merged_mask = mask if merged_mask is None else np.maximum(merged_mask, mask)
            if not cv2.imwrite(os.path.join(image_dir, case + ".png"), image):
                raise RuntimeError("Could not write image {}".format(case))
            if not cv2.imwrite(os.path.join(mask_dir, case + ".png"), merged_mask):
                raise RuntimeError("Could not write mask {}".format(case))
            if index % 100 == 0:
                print("prepared {}/647".format(index), flush=True)

    for split_index in (1, 2, 3):
        for split in ("train", "val"):
            filename = "busi_{}{}.txt".format(split, split_index)
            source = os.path.join(args.split_dir, filename)
            destination = os.path.join(args.output_dir, filename)
            shutil.copyfile(source, destination)

    print("Prepared 647 BUSI cases in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
