import csv
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from thop import profile

from src.network.conv_based.ThesisFourStageUNet import ThesisFourStageUNet


def active_parameters(model):
    prefixes = ["backbone."]
    if model.use_a:
        prefixes.append("module_a.")
    if model.use_b:
        prefixes.append("module_b.")
    if model.use_c:
        prefixes.extend(("module_c.", "boundary_head.", "distance_head."))
    if model.use_d:
        prefixes.extend(("module_d.", "auxiliary_heads."))
    return sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    )


def latency_ms(model, device):
    image = torch.randn(1, 3, 256, 256, device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(20):
            model(image)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            model(image)
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / 100.0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency benchmarking")
    rows = []
    for variant in ThesisFourStageUNet.VARIANTS:
        cpu_model = ThesisFourStageUNet(variant).eval()
        macs, _ = profile(cpu_model, inputs=(torch.randn(1, 3, 256, 256),), verbose=False)
        rows.append({
            "variant": variant,
            "active_parameters": active_parameters(cpu_model),
            "active_parameters_m": active_parameters(cpu_model) / 1e6,
            "gmacs_256": macs / 1e9,
            "latency_ms_batch1": latency_ms(ThesisFourStageUNet(variant), torch.device("cuda")),
        })
    os.makedirs("./thesis_artifacts/tables", exist_ok=True)
    path = "./thesis_artifacts/tables/model_complexity.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
