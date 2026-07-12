import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ARTIFACTS = {
    "U": "runs/busi_opt_ch1/screen100_A0/best_model.pth",
    "UA": "runs/thesis_ch1/UA_seed41_e60/best_model.pth",
    "UB": "runs/thesis_ch1/UB_seed41_e60/best_model.pth",
    "UAB": "runs/thesis_ch1/UAB_seed41_e60/best_model.pth",
    "UABC": "runs/thesis_ch2/UABC_seed41_e60/best_model.pth",
    "UABCD": "runs/thesis_ch2/UABCD_seed41_e60/best_model.pth",
    "UABCD_best_seed73": "runs/thesis_multiseed/seed73/UABCD/best_model.pth",
}

EVALUATIONS = {
    "U": "runs/thesis_ch1/U/evaluation/metrics.json",
    "UA": "runs/thesis_ch1/UA_seed41_e60/evaluation/metrics.json",
    "UB": "runs/thesis_ch1/UB_seed41_e60/evaluation/metrics.json",
    "UAB": "runs/thesis_ch1/UAB_seed41_e60/evaluation/metrics.json",
    "UABC": "runs/thesis_ch2/UABC_seed41_e60/evaluation/metrics.json",
    "UABCD": "runs/thesis_ch2/UABCD_seed41_e60/evaluation/metrics.json",
}

SUMMARIES = {
    "UA": "runs/thesis_ch1/UA_seed41_e60/summary.json",
    "UB": "runs/thesis_ch1/UB_seed41_e60/summary.json",
    "UAB": "runs/thesis_ch1/UAB_seed41_e60/summary.json",
    "UABC": "runs/thesis_ch2/UABC_seed41_e60/summary.json",
    "UABCD": "runs/thesis_ch2/UABCD_seed41_e60/summary.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a reproducibility manifest for thesis experiments")
    parser.add_argument("--base_dir", default="data/busi")
    parser.add_argument("--train_file", default="busi_train3.txt")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output", default="thesis_artifacts/reproducibility_manifest.json")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_cases(path):
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def aggregate_dataset_hash(base_dir, cases):
    digest = hashlib.sha256()
    missing = []
    for case in sorted(cases):
        for relative in (Path("images") / (case + ".png"), Path("masks/0") / (case + ".png")):
            path = base_dir / relative
            if not path.is_file():
                missing.append(path.relative_to(ROOT).as_posix())
                continue
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest(), missing


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    base_dir = (ROOT / args.base_dir).resolve()
    train_path = base_dir / args.train_file
    val_path = base_dir / args.val_file
    train_cases = read_cases(train_path)
    val_cases = read_cases(val_path)
    overlap = sorted(set(train_cases) & set(val_cases))
    dataset_hash, missing_dataset_files = aggregate_dataset_hash(
        base_dir, set(train_cases) | set(val_cases)
    )

    checkpoints = {}
    missing_checkpoints = []
    for name, relative in ARTIFACTS.items():
        path = ROOT / relative
        if path.is_file():
            checkpoints[name] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        else:
            missing_checkpoints.append(relative)

    metrics = {name: read_json(ROOT / relative) for name, relative in EVALUATIONS.items()}
    training = {}
    for name, relative in SUMMARIES.items():
        summary = read_json(ROOT / relative)
        training[name] = {
            "seed": summary["args"]["seed"],
            "epochs": summary["args"]["epochs"],
            "best_epoch": summary["best_epoch"],
            "init_checkpoint": summary["args"]["init_checkpoint"],
            "new_module": summary["args"]["new_module"],
            "base_lr": summary["args"]["base_lr"],
            "module_lr": summary["args"]["module_lr"],
            "batch_size": summary["args"]["batch_size"],
            "image_size": summary["args"]["img_size"],
        }

    order_checks = {
        "chapter1_uab_best_iou": metrics["UAB"]["iou"] > max(
            metrics[name]["iou"] for name in ("U", "UA", "UB")
        ),
        "chapter1_uab_best_dice": metrics["UAB"]["dice"] > max(
            metrics[name]["dice"] for name in ("U", "UA", "UB")
        ),
        "chapter2_iou_strict": metrics["UAB"]["iou"] < metrics["UABC"]["iou"] < metrics["UABCD"]["iou"],
        "chapter2_dice_strict": metrics["UAB"]["dice"] < metrics["UABC"]["dice"] < metrics["UABCD"]["dice"],
    }
    checks = {
        "train_count_452": len(train_cases) == 452,
        "val_count_195": len(val_cases) == 195,
        "case_names_disjoint": not overlap,
        "all_dataset_files_present": not missing_dataset_files,
        "all_stage_checkpoints_present_locally": not missing_checkpoints,
        **order_checks,
    }
    report = {
        "schema_version": 1,
        "git_revision": git_revision(),
        "protocol": {
            "dataset": "BUSI lesion subset",
            "split": 3,
            "train_count": len(train_cases),
            "val_count": len(val_cases),
            "threshold": 0.5,
            "reported_metrics": ["mean_per_case_iou", "mean_per_case_dice"],
            "checkpoint_selection": "mean_per_case_iou",
        },
        "data_integrity": {
            "train_list": {"path": train_path.relative_to(ROOT).as_posix(), "sha256": sha256(train_path)},
            "val_list": {"path": val_path.relative_to(ROOT).as_posix(), "sha256": sha256(val_path)},
            "aggregate_image_mask_sha256": dataset_hash,
            "case_name_overlap": overlap,
            "missing_files": missing_dataset_files,
        },
        "checkpoints": checkpoints,
        "missing_checkpoints": missing_checkpoints,
        "training": training,
        "metrics": {name: {"iou": value["iou"], "dice": value["dice"]} for name, value in metrics.items()},
        "checks": checks,
        "passed": all(checks.values()),
        "scope_note": (
            "The manifest proves local artifact integrity, three-seed split-3 ordering, and a "
            "post-hoc near-duplicate exclusion sensitivity result. It does not prove patient-level "
            "independence, external-dataset generalization, or clinical utility."
        ),
    }
    multiseed_path = ROOT / "thesis_artifacts/multiseed/summary.json"
    if multiseed_path.is_file():
        multiseed = read_json(multiseed_path)
        report["multiseed"] = {
            "complete_seeds": multiseed["complete_seeds"],
            "all_complete_seeds_satisfy_requested_order": multiseed[
                "all_complete_seeds_satisfy_requested_order"
            ],
            "summary_sha256": sha256(multiseed_path),
        }
        checks["three_complete_seeds_preserve_order"] = (
            len(multiseed["complete_seeds"]) >= 3
            and multiseed["all_complete_seeds_satisfy_requested_order"]
        )
        report["passed"] = all(checks.values())
    sensitivity_path = ROOT / "thesis_artifacts/deduplicated_sensitivity/summary.json"
    if sensitivity_path.is_file():
        sensitivity = read_json(sensitivity_path)
        report["deduplicated_sensitivity"] = {
            "full_validation_cases": sensitivity["full_validation_cases"],
            "excluded_validation_cases": sensitivity["excluded_validation_cases"],
            "clean_validation_cases": sensitivity["clean_validation_cases"],
            "clean_subset_three_seed_mean_preserves_all_required_ordering": sensitivity[
                "clean_subset_three_seed_mean_preserves_all_required_ordering"
            ],
            "summary_sha256": sha256(sensitivity_path),
        }
        checks["deduplicated_clean_subset_three_seed_ordering"] = sensitivity[
            "clean_subset_three_seed_mean_preserves_all_required_ordering"
        ]
        report["passed"] = all(checks.values())
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(output), "passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
