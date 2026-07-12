import json
import os
import zipfile

import torch

from src.network.conv_based.ThesisFourStageUNet import ThesisFourStageUNet


METRICS = {
    "U": "./runs/thesis_ch1/U/evaluation/metrics.json",
    "UA": "./runs/thesis_ch1/UA_seed41_e60/evaluation/metrics.json",
    "UB": "./runs/thesis_ch1/UB_seed41_e60/evaluation/metrics.json",
    "UAB": "./runs/thesis_ch1/UAB_seed41_e60/evaluation/metrics.json",
    "UABC": "./runs/thesis_ch2/UABC_seed41_e60/evaluation/metrics.json",
    "UABCD": "./runs/thesis_ch2/UABCD_seed41_e60/evaluation/metrics.json",
}


def identity_transition_differences():
    torch.manual_seed(41)
    image = torch.randn(1, 3, 64, 64)
    transitions = (("U", "UA"), ("U", "UB"), ("UB", "UAB"),
                   ("UAB", "UABC"), ("UABC", "UABCD"))
    differences = {}
    with torch.inference_mode():
        for before_name, after_name in transitions:
            before = ThesisFourStageUNet(before_name, base_channels=8).eval()
            after = ThesisFourStageUNet(after_name, base_channels=8).eval()
            after.load_state_dict(before.state_dict(), strict=True)
            before_output = before(image)["segmentation"]
            after_output = after(image)["segmentation"]
            differences["{}->{}".format(before_name, after_name)] = float(
                (before_output - after_output).abs().max()
            )
    return differences


def docx_media_count(path):
    with zipfile.ZipFile(path) as archive:
        return len([
            name for name in archive.namelist()
            if name.startswith("word/media/") and not name.endswith("/")
        ])


def main():
    values = {}
    for name, path in METRICS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
        values[name] = {"iou": result["iou"], "dice": result["dice"]}

    with open("./thesis_artifacts/reproducibility_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    with open("./thesis_artifacts/literature_evidence.json", encoding="utf-8") as file:
        literature = json.load(file)
    with open(
        "./runs/thesis_multiseed/seed73/UABCD/evaluation/metrics.json", encoding="utf-8"
    ) as file:
        seed73_best = json.load(file)
    with open(
        "./thesis_artifacts/deduplicated_sensitivity/summary.json", encoding="utf-8"
    ) as file:
        deduplicated = json.load(file)
    identity_differences = identity_transition_differences()

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
        "overall_best_seed73_checkpoint_exists": os.path.isfile(
            "./runs/thesis_multiseed/seed73/UABCD/best_model.pth"
        ),
        "overall_best_seed73_iou_exceeds_primary": (
            seed73_best["iou"] > values["UABCD"]["iou"]
        ),
        "markdown_draft_exists": os.path.isfile("./博士学位论文初稿.md"),
        "word_draft_exists": os.path.isfile("./博士学位论文初稿.docx"),
        "word_draft_contains_ten_images": docx_media_count(
            "./博士学位论文初稿.docx"
        ) >= 10,
        "at_least_ten_figures": len([
            name for name in os.listdir("./thesis_artifacts/figures")
            if name.lower().endswith(".png") and name.startswith("fig")
        ]) >= 10,
        "reproducibility_manifest_passed": manifest["passed"],
        "four_recent_sources_verified": (
            len(literature["records"]) >= 4
            and all(record.get("doi") and record.get("authoritative_url")
                    for record in literature["records"])
        ),
        "three_complete_seeds_preserve_order": (
            len(manifest.get("multiseed", {}).get("complete_seeds", [])) >= 3
            and manifest.get("multiseed", {}).get(
                "all_complete_seeds_satisfy_requested_order", False
            )
        ),
        "deduplicated_clean_subset_has_175_cases": (
            deduplicated["clean_validation_cases"] == 175
        ),
        "deduplicated_three_seed_mean_preserves_order": deduplicated[
            "clean_subset_three_seed_mean_preserves_all_required_ordering"
        ],
        "all_progressive_transitions_identity_initialized": all(
            difference == 0.0 for difference in identity_differences.values()
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": values,
        "identity_transition_max_abs_difference": identity_differences,
        "scope_note": (
            "This verifies the requested model ordering and draft artifacts on BUSI split 3. "
            "It verifies three optimization seeds on one fixed split, but does not prove "
            "patient-level independence or external-dataset generalization."
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
