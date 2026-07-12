import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "runs/busi_opt_ch1/screen100_A0/best_model.pth"
REFERENCE_METRICS = {
    "U": "runs/thesis_ch1/U/evaluation/metrics.json",
    "UA": "runs/thesis_ch1/UA_seed41_e60/evaluation/metrics.json",
    "UB": "runs/thesis_ch1/UB_seed41_e60/evaluation/metrics.json",
    "UAB": "runs/thesis_ch1/UAB_seed41_e60/evaluation/metrics.json",
    "UABC": "runs/thesis_ch2/UABC_seed41_e60/evaluation/metrics.json",
    "UABCD": "runs/thesis_ch2/UABCD_seed41_e60/evaluation/metrics.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable additional-seed thesis experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 73))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_root", default="runs/thesis_multiseed")
    parser.add_argument("--artifact_dir", default="thesis_artifacts/multiseed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command):
    print("RUN " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def stages(seed_root):
    ua = seed_root / "UA"
    ub = seed_root / "UB"
    uab = seed_root / "UAB"
    uabc = seed_root / "UABC"
    uabcd = seed_root / "UABCD"
    return [
        ("UA", "A", BASELINE, ua),
        ("UB", "B", BASELINE, ub),
        ("UAB", "A", ub / "best_model.pth", uab),
        ("UABC", "C", uab / "best_model.pth", uabc),
        ("UABCD", "D", uabc / "best_model.pth", uabcd),
    ]


def train_and_evaluate(seed, args):
    seed_root = (ROOT / args.output_root / ("seed{}".format(seed))).resolve()
    seed_root.mkdir(parents=True, exist_ok=True)
    for variant, new_module, init_checkpoint, output_dir in stages(seed_root):
        metrics_path = output_dir / "evaluation/metrics.json"
        if metrics_path.is_file() and not args.force:
            print("SKIP completed {} seed {}".format(variant, seed), flush=True)
            continue
        if not init_checkpoint.is_file():
            raise FileNotFoundError(init_checkpoint)
        run([
            sys.executable, "train_thesis_stages.py",
            "--variant", variant,
            "--init_checkpoint", str(init_checkpoint),
            "--new_module", new_module,
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--seed", str(seed),
            "--workers", str(args.workers),
        ])
        run([
            sys.executable, "evaluate_thesis_stages.py",
            "--variant", variant,
            "--model_path", str(output_dir / "best_model.pth"),
            "--output_dir", str(output_dir / "evaluation"),
            "--workers", str(args.workers),
        ])
    write_aggregate(args)


def collected_runs(args):
    runs = []
    reference = {
        model: read_json(ROOT / path) for model, path in REFERENCE_METRICS.items()
    }
    for model, metrics in reference.items():
        runs.append({"seed": 41, "model": model, "iou": metrics["iou"], "dice": metrics["dice"]})
    baseline = reference["U"]
    for seed in args.seeds:
        runs.append({"seed": seed, "model": "U", "iou": baseline["iou"], "dice": baseline["dice"]})
        seed_root = ROOT / args.output_root / ("seed{}".format(seed))
        for model in ("UA", "UB", "UAB", "UABC", "UABCD"):
            path = seed_root / model / "evaluation/metrics.json"
            if path.is_file():
                metrics = read_json(path)
                runs.append({"seed": seed, "model": model, "iou": metrics["iou"], "dice": metrics["dice"]})
    return runs


def write_aggregate(args):
    artifact_dir = ROOT / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = collected_runs(args)
    with (artifact_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("seed", "model", "iou", "dice"))
        writer.writeheader()
        writer.writerows(runs)

    complete_seeds = []
    seed_checks = {}
    by_seed = {}
    for row in runs:
        by_seed.setdefault(row["seed"], {})[row["model"]] = row
    required = {"U", "UA", "UB", "UAB", "UABC", "UABCD"}
    for seed, values in sorted(by_seed.items()):
        if set(values) != required:
            continue
        complete_seeds.append(seed)
        seed_checks[str(seed)] = {
            "chapter1_uab_best_iou": values["UAB"]["iou"] > max(values[name]["iou"] for name in ("U", "UA", "UB")),
            "chapter1_uab_best_dice": values["UAB"]["dice"] > max(values[name]["dice"] for name in ("U", "UA", "UB")),
            "chapter2_iou_strict": values["UAB"]["iou"] < values["UABC"]["iou"] < values["UABCD"]["iou"],
            "chapter2_dice_strict": values["UAB"]["dice"] < values["UABC"]["dice"] < values["UABCD"]["dice"],
        }

    aggregate = []
    for model in ("U", "UA", "UB", "UAB", "UABC", "UABCD"):
        model_rows = [row for row in runs if row["seed"] in complete_seeds and row["model"] == model]
        if not model_rows:
            continue
        aggregate.append({
            "model": model,
            "seeds": len(model_rows),
            "iou_mean": float(np.mean([row["iou"] for row in model_rows])),
            "iou_std": float(np.std([row["iou"] for row in model_rows], ddof=1)) if len(model_rows) > 1 else 0.0,
            "dice_mean": float(np.mean([row["dice"] for row in model_rows])),
            "dice_std": float(np.std([row["dice"] for row in model_rows], ddof=1)) if len(model_rows) > 1 else 0.0,
        })
    with (artifact_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("model", "seeds", "iou_mean", "iou_std", "dice_mean", "dice_std")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    report = {
        "protocol": "Fixed BUSI split 3, threshold 0.5, 60 epochs per progressive transition",
        "complete_seeds": complete_seeds,
        "seed_checks": seed_checks,
        "all_complete_seeds_satisfy_requested_order": bool(seed_checks) and all(
            all(checks.values()) for checks in seed_checks.values()
        ),
        "aggregate": aggregate,
        "scope_note": (
            "Additional seeds vary fine-tuning order and augmentation from a shared pretrained U checkpoint. "
            "They do not replace patient-level deduplication or external validation."
        ),
    }
    with (artifact_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main():
    args = parse_args()
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    for seed in args.seeds:
        train_and_evaluate(seed, args)


if __name__ == "__main__":
    main()
