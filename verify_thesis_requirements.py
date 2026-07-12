import json
import os


METRICS = {
    "U": "./runs/thesis_ch1/U/evaluation/metrics.json",
    "UA": "./runs/thesis_ch1/UA_seed41_e60/evaluation/metrics.json",
    "UB": "./runs/thesis_ch1/UB_seed41_e60/evaluation/metrics.json",
    "UAB": "./runs/thesis_ch1/UAB_seed41_e60/evaluation/metrics.json",
    "UABC": "./runs/thesis_ch2/UABC_seed41_e60/evaluation/metrics.json",
    "UABCD": "./runs/thesis_ch2/UABCD_seed41_e60/evaluation/metrics.json",
}


def main():
    values = {}
    for name, path in METRICS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
        values[name] = {"iou": result["iou"], "dice": result["dice"]}

    checks = {
        "chapter1_uab_best_iou": values["UAB"]["iou"] > max(
            values[name]["iou"] for name in ("U", "UA", "UB")
        ),
        "chapter1_uab_best_dice": values["UAB"]["dice"] > max(
            values[name]["dice"] for name in ("U", "UA", "UB")
        ),
        "chapter2_strict_iou_order": (
            values["UAB"]["iou"] < values["UABC"]["iou"] < values["UABCD"]["iou"]
        ),
        "chapter2_strict_dice_order": (
            values["UAB"]["dice"] < values["UABC"]["dice"] < values["UABCD"]["dice"]
        ),
        "final_checkpoint_exists": os.path.isfile(
            "./runs/thesis_ch2/UABCD_seed41_e60/best_model.pth"
        ),
        "markdown_draft_exists": os.path.isfile("./博士学位论文初稿.md"),
        "word_draft_exists": os.path.isfile("./博士学位论文初稿.docx"),
        "at_least_eight_figures": len([
            name for name in os.listdir("./thesis_artifacts/figures")
            if name.lower().endswith(".png") and name.startswith("fig")
        ]) >= 8,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": values,
        "scope_note": (
            "This verifies the requested model ordering and draft artifacts on BUSI split 3. "
            "It does not prove multi-seed or external-dataset generalization."
        ),
    }
    os.makedirs("./thesis_artifacts", exist_ok=True)
    with open("./thesis_artifacts/verification.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
