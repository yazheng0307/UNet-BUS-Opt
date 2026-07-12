import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from scipy.stats import wilcoxon

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.ThesisFourStageUNet import ThesisFourStageUNet
from train_unet import ValTransform


OUTPUT_DIR = "./thesis_artifacts"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

EVALUATIONS = {
    "U": "./runs/thesis_ch1/U/evaluation/per_case_metrics.csv",
    "UA": "./runs/thesis_ch1/UA_seed41_e60/evaluation/per_case_metrics.csv",
    "UB": "./runs/thesis_ch1/UB_seed41_e60/evaluation/per_case_metrics.csv",
    "UAB": "./runs/thesis_ch1/UAB_seed41_e60/evaluation/per_case_metrics.csv",
    "UABC": "./runs/thesis_ch2/UABC_seed41_e60/evaluation/per_case_metrics.csv",
    "UABCD": "./runs/thesis_ch2/UABCD_seed41_e60/evaluation/per_case_metrics.csv",
}

HISTORIES = {
    "UA": "./runs/thesis_ch1/UA_seed41_e60/history.csv",
    "UB": "./runs/thesis_ch1/UB_seed41_e60/history.csv",
    "UAB": "./runs/thesis_ch1/UAB_seed41_e60/history.csv",
    "UABC": "./runs/thesis_ch2/UABC_seed41_e60/history.csv",
    "UABCD": "./runs/thesis_ch2/UABCD_seed41_e60/history.csv",
}

CHECKPOINTS = {
    "U": "./runs/busi_opt_ch1/screen100_A0/best_model.pth",
    "UAB": "./runs/thesis_ch1/UAB_seed41_e60/best_model.pth",
    "UABC": "./runs/thesis_ch2/UABC_seed41_e60/best_model.pth",
    "UABCD": "./runs/thesis_ch2/UABCD_seed41_e60/best_model.pth",
}

COLORS = {
    "U": "#777777", "UA": "#4C78A8", "UB": "#59A14F",
    "UAB": "#E15759", "UABC": "#F28E2B", "UABCD": "#B07AA1",
}


def setup():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })


def load_metrics():
    return {name: pd.read_csv(path).sort_values("case").reset_index(drop=True)
            for name, path in EVALUATIONS.items()}


def mean_table(metrics):
    rows = []
    for name, frame in metrics.items():
        rows.append({
            "model": name,
            "iou": frame["iou"].mean(),
            "dice": frame["dice"].mean(),
            "iou_std": frame["iou"].std(ddof=1),
            "dice_std": frame["dice"].std(ddof=1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLE_DIR, "all_model_metrics.csv"), index=False)
    return table.set_index("model")


def grouped_bars(table, names, title, output_name):
    values_iou = [100.0 * table.loc[name, "iou"] for name in names]
    values_dice = [100.0 * table.loc[name, "dice"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars1 = ax.bar(x - width / 2, values_iou, width, label="IoU", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, values_dice, width, label="Dice", color="#F28E2B")
    ax.set_xticks(x, names)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_ylim(min(values_iou) - 2.0, max(values_dice) + 2.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                    "{:.2f}".format(bar.get_height()), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, output_name), bbox_inches="tight")
    plt.close(fig)


def training_curves(names, title, output_name):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for name in names:
        frame = pd.read_csv(HISTORIES[name])
        axes[0].plot(frame["epoch"], 100 * frame["val_iou"], label=name,
                     color=COLORS[name], linewidth=1.7)
        axes[1].plot(frame["epoch"], 100 * frame["val_dice"], label=name,
                     color=COLORS[name], linewidth=1.7)
    axes[0].set_title("Validation IoU")
    axes[1].set_title("Validation Dice")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score (%)")
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, output_name), bbox_inches="tight")
    plt.close(fig)


def case_distribution(metrics):
    names = ["U", "UA", "UB", "UAB", "UABC", "UABCD"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric in zip(axes, ("iou", "dice")):
        data = [100 * metrics[name][metric].to_numpy() for name in names]
        box = axis.boxplot(data, labels=names, patch_artist=True, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.4})
        for patch, name in zip(box["boxes"], names):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.72)
        axis.set_title("Per-case {} distribution".format(metric.upper()))
        axis.set_ylabel("Score (%)")
        axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig06_case_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def paired_statistics(metrics):
    comparisons = [("U", "UA"), ("U", "UB"), ("UB", "UAB"),
                   ("UAB", "UABC"), ("UABC", "UABCD")]
    rng = np.random.default_rng(41)
    rows = []
    for before, after in comparisons:
        for metric in ("iou", "dice"):
            delta = metrics[after][metric].to_numpy() - metrics[before][metric].to_numpy()
            samples = rng.integers(0, len(delta), size=(10000, len(delta)))
            bootstrap = delta[samples].mean(axis=1)
            nonzero = delta[np.abs(delta) > 1e-12]
            test = wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
            rows.append({
                "comparison": "{}-{}".format(after, before),
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value_one_sided": float(test.pvalue),
                "improved_cases": int((delta > 0).sum()),
                "degraded_cases": int((delta < 0).sum()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(TABLE_DIR, "paired_statistics.csv"), index=False)
    with open(os.path.join(TABLE_DIR, "paired_statistics.json"), "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=True)
    return frame


def delta_plot(metrics):
    pairs = [("U", "UA"), ("U", "UB"), ("UB", "UAB"),
             ("UAB", "UABC"), ("UABC", "UABCD")]
    labels = ["A vs U", "B vs U", "A on UB", "C on UAB", "D on UABC"]
    data = [100 * (metrics[after]["iou"] - metrics[before]["iou"]).to_numpy()
            for before, after in pairs]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], [COLORS["UA"], COLORS["UB"], COLORS["UAB"],
                                              COLORS["UABC"], COLORS["UABCD"]]):
        body.set_facecolor(color)
        body.set_alpha(0.68)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("Per-case IoU change (percentage points)")
    ax.set_title("Paired contribution of progressively added modules")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig07_paired_module_delta.png"), bbox_inches="tight")
    plt.close(fig)


def multiseed_plot():
    path = os.path.join(OUTPUT_DIR, "multiseed", "aggregate_metrics.csv")
    if not os.path.isfile(path):
        return
    frame = pd.read_csv(path)
    seed_count = int(frame["seeds"].max())
    names = frame["model"].tolist()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, color in (
        (axes[0], "iou", "#4C78A8"),
        (axes[1], "dice", "#F28E2B"),
    ):
        means = 100 * frame[metric + "_mean"].to_numpy()
        standard_deviations = 100 * frame[metric + "_std"].to_numpy()
        axis.errorbar(
            x, means, yerr=standard_deviations, fmt="o-", color=color,
            linewidth=1.8, markersize=6, capsize=4,
        )
        axis.set_xticks(x, names)
        axis.set_ylabel("{} (%)".format(metric.upper()))
        axis.set_title("{}-seed mean +/- standard deviation".format(seed_count))
        axis.grid(axis="y", alpha=0.22)
        for position, value in zip(x, means):
            axis.text(position, value + 0.12, "{:.2f}".format(value),
                      ha="center", va="bottom", fontsize=8)
    fig.suptitle("Random-seed replication on fixed BUSI split 3 (n={})".format(seed_count))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig09_multiseed_replication.png"), bbox_inches="tight")
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(14.5, 3.3))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    items = [
        (0.2, "Input\nultrasound", "#D9D9D9"),
        (2.1, "U-Net\nencoder", "#A0CBE8"),
        (4.0, "A: TriCAR\nchallenge routing", "#4C78A8"),
        (6.1, "B: CGR-Bridge\nglobal relation", "#59A14F"),
        (8.25, "U-Net\ndecoder", "#A0CBE8"),
        (10.1, "C: BD-CoRefine\nboundary + distance", "#F28E2B"),
        (12.15, "D: UDER\nFN/FP correction", "#B07AA1"),
    ]
    widths = [1.45, 1.45, 1.65, 1.65, 1.35, 1.65, 1.65]
    for (x, text, color), width in zip(items, widths):
        box = FancyBboxPatch((x, 1.0), width, 1.0, boxstyle="round,pad=0.04,rounding_size=0.06",
                             facecolor=color, edgecolor="#333333", linewidth=1.1)
        ax.add_patch(box)
        ax.text(x + width / 2, 1.5, text, ha="center", va="center",
                color="white" if color not in ("#D9D9D9", "#A0CBE8") else "#222222",
                fontsize=9, fontweight="bold")
    for index in range(len(items) - 1):
        start = items[index][0] + widths[index]
        end = items[index + 1][0]
        ax.annotate("", xy=(end, 1.5), xytext=(start, 1.5),
                    arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"})
    ax.text(5.35, 2.55, "Chapter 1: challenge-adaptive semantic representation",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(11.6, 2.55, "Chapter 2: geometry-uncertainty closed-loop refinement",
            ha="center", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig01_overall_architecture.png"), bbox_inches="tight")
    plt.close(fig)


def complexity_table():
    model = ThesisFourStageUNet("UABCD")
    groups = {
        "U-Net backbone": "backbone.",
        "Module A": "module_a.",
        "Module B": "module_b.",
        "Module C": ("module_c.", "boundary_head.", "distance_head."),
        "Module D": ("module_d.", "auxiliary_heads."),
    }
    rows = []
    for group, prefixes in groups.items():
        prefixes = (prefixes,) if isinstance(prefixes, str) else prefixes
        count = sum(parameter.numel() for name, parameter in model.named_parameters()
                    if any(name.startswith(prefix) for prefix in prefixes))
        rows.append({"component": group, "parameters": count, "parameters_m": count / 1e6})
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "parameter_breakdown.csv"), index=False)


def load_model(variant, checkpoint, device):
    model = ThesisFourStageUNet(variant).to(device)
    model.load_progressive_checkpoint(checkpoint)
    model.eval()
    return model


def qualitative_figure(metrics):
    delta = metrics["UABCD"]["iou"] - metrics["U"]["iou"]
    final = metrics["UABCD"]["iou"]
    selected_positions = [
        int(delta.idxmax()),
        int((delta - delta.median()).abs().idxmin()),
        int(final.idxmin()),
        int(final.idxmax()),
    ]
    dataset = MedicalDataSets(
        base_dir="./data/busi", split="val", transform=ValTransform(256),
        val_file_dir="busi_val3.txt",
    )
    case_to_index = {case: index for index, case in enumerate(dataset.sample_list)}
    selected_cases = [metrics["UABCD"].iloc[position]["case"] for position in selected_positions]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: load_model(name, CHECKPOINTS[name], device)
              for name in ("U", "UAB", "UABC", "UABCD")}
    columns = ["Image + GT", "U", "UAB", "UABC", "UABCD"]
    fig, axes = plt.subplots(len(selected_cases), len(columns), figsize=(15, 11))
    prediction_dir = os.path.join(FIGURE_DIR, "qualitative_masks")
    os.makedirs(prediction_dir, exist_ok=True)
    with torch.no_grad():
        for row_index, case in enumerate(selected_cases):
            sample = dataset[case_to_index[case]]
            image_tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
            target = sample["label"][0] >= 0.5
            image = cv2.imread(os.path.join("./data/busi/images", case + ".png"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].contour(target, levels=[0.5], colors=["#00FF66"], linewidths=1.4)
            axes[row_index, 0].set_ylabel(case, fontsize=9)
            for column_index, name in enumerate(("U", "UAB", "UABC", "UABCD"), start=1):
                logits = models[name](image_tensor)["segmentation"]
                prediction = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
                axes[row_index, column_index].imshow(image)
                axes[row_index, column_index].contour(target, levels=[0.5], colors=["#00FF66"], linewidths=1.1)
                if prediction.any():
                    axes[row_index, column_index].contour(prediction, levels=[0.5], colors=["#FF3344"], linewidths=1.1)
                case_row = metrics[name].set_index("case").loc[case]
                axes[row_index, column_index].set_xlabel(
                    "IoU {:.3f} | Dice {:.3f}".format(case_row["iou"], case_row["dice"]), fontsize=8
                )
                cv2.imwrite(os.path.join(prediction_dir, "{}_{}.png".format(case, name)),
                            prediction.astype(np.uint8) * 255)
            for column_index, title in enumerate(columns):
                if row_index == 0:
                    axes[row_index, column_index].set_title(title)
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
    fig.suptitle("Qualitative comparison (green: ground truth, red: prediction)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig08_qualitative_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(TABLE_DIR, "qualitative_cases.json"), "w", encoding="utf-8") as file:
        json.dump(selected_cases, file, indent=2, ensure_ascii=True)


def main():
    setup()
    metrics = load_metrics()
    table = mean_table(metrics)
    architecture_figure()
    grouped_bars(table, ["U", "UA", "UB", "UAB"],
                 "Chapter 1 ablation study", "fig02_chapter1_ablation.png")
    training_curves(["UA", "UB", "UAB"],
                    "Chapter 1 validation trajectories", "fig03_chapter1_curves.png")
    grouped_bars(table, ["UAB", "UABC", "UABCD"],
                 "Chapter 2 progressive refinement", "fig04_chapter2_ablation.png")
    training_curves(["UABC", "UABCD"],
                    "Chapter 2 validation trajectories", "fig05_chapter2_curves.png")
    case_distribution(metrics)
    paired_statistics(metrics)
    delta_plot(metrics)
    multiseed_plot()
    complexity_table()
    qualitative_figure(metrics)
    print("Thesis artifacts written to {}".format(os.path.abspath(OUTPUT_DIR)), flush=True)


if __name__ == "__main__":
    main()
