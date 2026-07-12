import csv
import json
import os
import statistics


METRICS = [
    "iou",
    "dice",
]


def main():
    rows = []
    for split in (1, 2, 3):
        run_dir = os.path.join("runs", "unet_busi_split{}".format(split))
        with open(os.path.join(run_dir, "evaluation", "metrics.json"), encoding="utf-8") as file:
            metrics = json.load(file)
        with open(os.path.join(run_dir, "history.csv"), newline="", encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        best_row = max(history, key=lambda row: float(row["val_iou"]))
        row = {"split": split, "best_epoch": int(best_row["epoch"])}
        row.update({metric: metrics[metric] for metric in METRICS})
        rows.append(row)

    aggregate = {
        metric: {
            "mean": statistics.mean(row[metric] for row in rows),
            "sample_std": statistics.stdev(row[metric] for row in rows),
        }
        for metric in METRICS
    }
    result = {"splits": rows, "aggregate": aggregate, "std_ddof": 1}

    output_json = os.path.join("runs", "unet_busi_3splits_summary.json")
    output_csv = os.path.join("runs", "unet_busi_3splits_results.csv")
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "best_epoch"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
