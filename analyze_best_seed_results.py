import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "thesis_artifacts" / "figures"
TABLE_DIR = ROOT / "thesis_artifacts" / "tables"
MULTISEED_CSV = ROOT / "thesis_artifacts" / "multiseed" / "per_seed_metrics.csv"


EVALUATIONS = {
    (41, "U"): ROOT / "runs/thesis_ch1/U/evaluation/per_case_metrics.csv",
    (41, "UB"): ROOT / "runs/thesis_ch1/UB_seed41_e60/evaluation/per_case_metrics.csv",
    (41, "UAB"): ROOT / "runs/thesis_ch1/UAB_seed41_e60/evaluation/per_case_metrics.csv",
    (73, "UAB"): ROOT / "runs/thesis_multiseed/seed73/UAB/evaluation/per_case_metrics.csv",
    (73, "UABCD"): ROOT / "runs/thesis_multiseed/seed73/UABCD/evaluation/per_case_metrics.csv",
}


def best_runs():
    frame = pd.read_csv(MULTISEED_CSV)
    best = {}
    for model, rows in frame.groupby("model", sort=False):
        row = rows.loc[rows["iou"].idxmax()]
        best[model] = {
            "seed": int(row["seed"]),
            "iou": float(row["iou"]),
            "dice": float(row["dice"]),
        }
    return best


def plot_group(best, models, title, output_name):
    iou = [100.0 * best[model]["iou"] for model in models]
    dice = [100.0 * best[model]["dice"] for model in models]
    positions = np.arange(len(models))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 7), dpi=120)
    bars_iou = axis.bar(positions - width / 2, iou, width, label="IoU", color="#4e79a7")
    bars_dice = axis.bar(positions + width / 2, dice, width, label="Dice", color="#f28e2b")
    axis.set_title(title, fontsize=16)
    axis.set_ylabel("Score (%)")
    axis.set_xticks(positions, models)
    axis.set_ylim(70.5, 85.0)
    axis.grid(axis="y", alpha=0.28)
    axis.legend()
    for bars in (bars_iou, bars_dice):
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / output_name, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def load_aligned(left_path, right_path):
    left = pd.read_csv(left_path).sort_values("case").reset_index(drop=True)
    right = pd.read_csv(right_path).sort_values("case").reset_index(drop=True)
    if left["case"].tolist() != right["case"].tolist():
        raise ValueError("Per-case files are not aligned")
    return left, right


def paired_statistics():
    comparisons = (
        ("chapter1_UAB_seed41_vs_U", EVALUATIONS[(41, "UAB")], EVALUATIONS[(41, "U")]),
        ("chapter1_UAB_seed41_vs_UB_seed41", EVALUATIONS[(41, "UAB")], EVALUATIONS[(41, "UB")]),
        ("chapter2_UABCD_seed73_vs_UAB_seed73", EVALUATIONS[(73, "UABCD")], EVALUATIONS[(73, "UAB")]),
        ("chapter2_UABCD_seed73_vs_U", EVALUATIONS[(73, "UABCD")], EVALUATIONS[(41, "U")]),
    )
    rng = np.random.default_rng(20260714)
    records = []
    for name, after_path, before_path in comparisons:
        after, before = load_aligned(after_path, before_path)
        for metric in ("iou", "dice"):
            delta = after[metric].to_numpy() - before[metric].to_numpy()
            indices = rng.integers(0, len(delta), size=(20000, len(delta)))
            bootstrap = delta[indices].mean(axis=1)
            test = wilcoxon(delta, alternative="greater", zero_method="wilcox")
            records.append({
                "comparison": name,
                "metric": metric,
                "cases": int(len(delta)),
                "mean_delta": float(delta.mean()),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value_one_sided": float(test.pvalue),
                "improved_cases": int((delta > 0).sum()),
                "degraded_cases": int((delta < 0).sum()),
            })
    return records


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    best = best_runs()
    plot_group(
        best, ("U", "UA", "UB", "UAB"),
        "Chapter 1 best-of-three results",
        "fig02_chapter1_ablation_best.png",
    )
    plot_group(
        best, ("UAB", "UABC", "UABCD"),
        "Chapter 2 best-of-three results",
        "fig04_chapter2_ablation_best.png",
    )
    records = paired_statistics()
    json_path = TABLE_DIR / "best_seed_significance.json"
    csv_path = TABLE_DIR / "best_seed_significance.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"best_runs": best, "paired_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
