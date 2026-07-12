import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE_EVALUATIONS = {
    "U": "runs/thesis_ch1/U/evaluation/per_case_metrics.csv",
    "UA": "runs/thesis_ch1/UA_seed41_e60/evaluation/per_case_metrics.csv",
    "UB": "runs/thesis_ch1/UB_seed41_e60/evaluation/per_case_metrics.csv",
    "UAB": "runs/thesis_ch1/UAB_seed41_e60/evaluation/per_case_metrics.csv",
    "UABC": "runs/thesis_ch2/UABC_seed41_e60/evaluation/per_case_metrics.csv",
    "UABCD": "runs/thesis_ch2/UABCD_seed41_e60/evaluation/per_case_metrics.csv",
}
MODELS = tuple(REFERENCE_EVALUATIONS)
SEEDS = (7, 41, 73)


def evaluation_paths(seed):
    if seed == 41:
        return REFERENCE_EVALUATIONS
    root = "runs/thesis_multiseed/seed{}".format(seed)
    paths = {"U": REFERENCE_EVALUATIONS["U"]}
    paths.update({
        model: "{}/{}/evaluation/per_case_metrics.csv".format(root, model)
        for model in MODELS if model != "U"
    })
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate thesis model ordering after excluding cross-split near duplicates"
    )
    parser.add_argument("--audit", default="runs/busi_split3_audit.json")
    parser.add_argument("--base_dir", default="data/busi")
    parser.add_argument("--val_file", default="busi_val3.txt")
    parser.add_argument("--output_dir", default="thesis_artifacts/deduplicated_sensitivity")
    return parser.parse_args()


def read_cases(path):
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def read_metrics(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        row["case"]: {"iou": float(row["iou"]), "dice": float(row["dice"])}
        for row in rows
    }


def mean_metrics(values, cases):
    return {
        metric: float(np.mean([values[case][metric] for case in cases]))
        for metric in ("iou", "dice")
    }


def ordering_checks(metrics):
    return {
        "chapter1_uab_best_iou": metrics["UAB"]["iou"] > max(
            metrics[name]["iou"] for name in ("U", "UA", "UB")
        ),
        "chapter1_uab_best_dice": metrics["UAB"]["dice"] > max(
            metrics[name]["dice"] for name in ("U", "UA", "UB")
        ),
        "chapter2_iou_strict": (
            metrics["UAB"]["iou"] < metrics["UABC"]["iou"] < metrics["UABCD"]["iou"]
        ),
        "chapter2_dice_strict": (
            metrics["UAB"]["dice"] < metrics["UABC"]["dice"] < metrics["UABCD"]["dice"]
        ),
    }


def main():
    args = parse_args()
    audit_path = ROOT / args.audit
    with audit_path.open("r", encoding="utf-8") as file:
        audit = json.load(file)
    val_cases = read_cases(ROOT / args.base_dir / args.val_file)
    val_set = set(val_cases)
    pairs = audit["cross_split_likely_near_image_pairs"]
    exposed_cases = sorted({
        case
        for pair in pairs
        for case in (pair["left"], pair["right"])
        if case in val_set
    })
    clean_cases = sorted(val_set - set(exposed_cases))
    if not exposed_cases or not clean_cases:
        raise RuntimeError("Near-duplicate sensitivity analysis requires both exposed and clean cases")

    per_seed_model = {}
    for seed in SEEDS:
        per_seed_model[seed] = {
            name: read_metrics(ROOT / path)
            for name, path in evaluation_paths(seed).items()
        }
        for name, values in per_seed_model[seed].items():
            missing = val_set - set(values)
            if missing:
                raise RuntimeError(
                    "seed {} {} is missing {} validation cases".format(seed, name, len(missing))
                )

    subsets = {
        "full": sorted(val_set),
        "clean_without_cross_split_near_duplicates": clean_cases,
        "near_duplicate_exposed": exposed_cases,
    }
    per_seed_metrics = {
        seed: {
            subset: {
                name: mean_metrics(values, cases)
                for name, values in models.items()
            }
            for subset, cases in subsets.items()
        }
        for seed, models in per_seed_model.items()
    }
    aggregate_metrics = {
        subset: {
            name: {
                metric: float(np.mean([
                    per_seed_metrics[seed][subset][name][metric] for seed in SEEDS
                ]))
                for metric in ("iou", "dice")
            }
            for name in MODELS
        }
        for subset in subsets
    }
    per_seed_checks = {
        str(seed): {
            subset: ordering_checks(metrics)
            for subset, metrics in seed_metrics.items()
        }
        for seed, seed_metrics in per_seed_metrics.items()
    }
    aggregate_checks = {
        subset: ordering_checks(metrics)
        for subset, metrics in aggregate_metrics.items()
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for subset, metrics in per_seed_metrics[seed].items():
            for name, values in metrics.items():
                rows.append({
                    "seed": seed,
                    "subset": subset,
                    "samples": len(subsets[subset]),
                    "model": name,
                    "iou": values["iou"],
                    "dice": values["dice"],
                    "iou_delta_vs_full": values["iou"] - per_seed_metrics[seed]["full"][name]["iou"],
                    "dice_delta_vs_full": values["dice"] - per_seed_metrics[seed]["full"][name]["dice"],
                })
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "excluded_validation_cases.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(exposed_cases) + "\n")

    report = {
        "audit_path": args.audit,
        "definition": {
            "phash_distance_max": audit["phash_distance_threshold"],
            "thumbnail_correlation_min": audit["thumbnail_correlation_threshold"],
            "cross_split_high_confidence_pairs": len(pairs),
        },
        "full_validation_cases": len(val_cases),
        "excluded_validation_cases": len(exposed_cases),
        "clean_validation_cases": len(clean_cases),
        "excluded_case_names": exposed_cases,
        "seeds": list(SEEDS),
        "per_seed_metrics": per_seed_metrics,
        "three_seed_mean_metrics": aggregate_metrics,
        "per_seed_ordering_checks": per_seed_checks,
        "three_seed_mean_ordering_checks": aggregate_checks,
        "clean_subset_three_seed_mean_preserves_all_required_ordering": all(
            aggregate_checks["clean_without_cross_split_near_duplicates"].values()
        ),
        "scope_note": (
            "This post-hoc sensitivity analysis removes validation images with high-confidence "
            "cross-split visual near duplicates. It does not reconstruct patient identities, retrain "
            "on a patient-level split, or establish external generalization."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["clean_subset_three_seed_mean_preserves_all_required_ordering"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
