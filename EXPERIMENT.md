# U-Net BUSI Experiment

## Environment

- Conda environment: `E:\anaconda3\envs\my_pytorch`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB)
- PyTorch: 2.2.0+cu118
- Dataset: BUSI benign and malignant cases (647 images)
- Dataset archive MD5: `39A3F05935D85C5BFC2572353C463EEA`

## Dataset Preparation

```powershell
E:\anaconda3\envs\my_pytorch\python.exe prepare_busi.py `
  --archive .\data\busi.zip `
  --output_dir .\data\busi `
  --split_dir .\data
```

The preparation script merges multiple masks belonging to the same case. Each of
the three repository splits contains 452 training and 195 validation cases.

## Training

All three splits were trained with the benchmark settings (SGD, learning rate
0.01, polynomial decay, 300 epochs, 256x256 images, batch size 8) and AMP
enabled.

Split 3 is the default data partition for subsequent training and evaluation
experiments. Running `train_unet.py` or `evaluate_unet.py` without split-related
arguments uses `busi_train3.txt` and `busi_val3.txt`.

```powershell
E:\anaconda3\envs\my_pytorch\python.exe train_unet.py `
  --base_dir .\data\busi `
  --train_file busi_train3.txt `
  --val_file busi_val3.txt `
  --output_dir .\runs\unet_busi_split3 `
  --epochs 300 `
  --batch_size 8 `
  --workers 4
```

Each split was trained independently for 300 epochs. The best checkpoint for
each run was selected by validation IoU.

| Split | Best epoch | IoU | Dice |
| --- | ---: | ---: | ---: |
| 1 | 138 | 0.626725 | 0.759638 |
| 2 | 176 | 0.640260 | 0.770721 |
| 3 | 159 | 0.695479 | 0.813824 |
| Mean +/- sample std | - | 0.654155 +/- 0.036422 | 0.781395 +/- 0.028626 |

The best weights and detailed histories are stored in the corresponding
`runs/unet_busi_split1`, `runs/unet_busi_split2`, and `runs/unet_busi_split3`
directories. Aggregate results are written to
`runs/unet_busi_3splits_summary.json`.

## Evaluation

```powershell
E:\anaconda3\envs\my_pytorch\python.exe evaluate_unet.py `
  --model_path .\runs\unet_busi_split3\U_Net_model.pth `
  --base_dir .\data\busi `
  --val_file busi_val3.txt `
  --output_dir .\runs\unet_busi_split3\evaluation
```

The default split 3 epoch history is in `runs/unet_busi_split3/history.csv`.
Independent evaluation metrics are in
`runs/unet_busi_split3/evaluation/metrics.json`.

## Strict Optimization Metrics

New experiments select checkpoints by mean per-image IoU rather than the
original repository's batch-averaged metric. Only mean per-image IoU and Dice
at a fixed threshold of 0.5 are reported.

| Model | Epoch budget | Best epoch | IoU | Dice |
| --- | ---: | ---: | ---: | ---: |
| A0 residual U-Net | 100 | 97 | 0.72445 | 0.80718 |
| TriCAR only (A2R) | 100 | 83 | 0.74340 | 0.82640 |
| CGR-Bridge only (A2G) | 100 | 88 | 0.75102 | 0.83559 |
| TriCAR + CGR (A2) | 100 | 91 | 0.74935 | 0.83234 |
| Legacy SFS + TriCAR + CGR (A3) | 100 | 63 | 0.75444 | 0.84112 |
| FG, initialized from A2G | 80 | 74 | 0.75855 | 0.84139 |
| Cached-pretrained ResNet50 U-Net | 100 | 55 | 0.76748 | 0.84653 |
| ResNet50 continued fine-tuning control (seed 3407) | 60 | 60 | 0.77526 | 0.85211 |
| ResNet50 + CGR + BD-CoDec + UDER (seed 41) | 60 | 9 | 0.77001 | 0.84699 |
| ResNet50 + CGR + BD-CoDec + UDER (seed 3407) | 60 | 4 | 0.77839 | 0.85577 |
| ResNet50 + UDER-only (seed 3407) | 20 | 4 | **0.77875** | **0.85594** |

The best strict split-3 per-image result so far is the non-destructive
refinement of the ResNet50 U-Net at seed 3407. Against a fair continued
fine-tuning control with the same initialization, seed, backbone learning rate,
augmentation, and epoch budget, it gains 0.00313 IoU and 0.00366 Dice. This run contains
CGR-Bridge, BD-CoDec auxiliary geometry supervision, and UDER, but not TriCAR;
it is a strong-backbone transfer experiment rather than the complete
four-module thesis model.

The best refined checkpoint was selected at epoch 4 and independently reloaded
at a fixed threshold of 0.5. Its IoU is 0.77839 and Dice is 0.85577. The 195
per-case rows are stored in
`runs/resnet50_four_module_split3_seed3407/evaluation`.

## Strong-Backbone Component Screening

All variants start from the same ResNet50 checkpoint, use seed 3407 and a
20-epoch cosine schedule, and have exactly identical initial segmentation
outputs. The same-schedule fine-tuning control reaches 0.77348 IoU.

| Variant | IoU | Dice | Gain over control (IoU) |
| --- | ---: | ---: | ---: |
| Fine-tuning control | 0.77348 | 0.85049 | - |
| Deep supervision only | 0.77485 | 0.85269 | +0.00137 |
| CGR-Bridge only | 0.77640 | 0.85449 | +0.00291 |
| BD-CoDec only | 0.77650 | 0.85461 | +0.00301 |
| UDER + deep supervision | **0.77875** | **0.85594** | **+0.00527** |

Compared with deep supervision alone, UDER adds 0.00390 IoU and 0.00324 Dice.
This is the clearest isolated module effect in the strong-backbone experiments.

Fixed horizontal-flip TTA and largest-connected-component post-processing were
also tested as supplementary diagnostics. They reduce per-image IoU to 0.76481
and 0.76754 respectively (0.76343 combined), so neither is retained.

## Rejected and Revised Pilots

| Pilot | Result | Decision |
| --- | ---: | --- |
| Legacy SFS-Cal A1, 100 epochs | 0.71458 IoU | rejected |
| Shrinkage SFS-Cal v2, 100 epochs | 0.71329 IoU | rejected |
| LSSG speckle guard, stopped at epoch 44 | 0.6689 IoU | rejected early |
| B0 deep-supervision control, 60 epochs | 0.70385 IoU | screening control |
| Boundary-only B1, 60 epochs | 0.69682 IoU | revise |
| Boundary+distance B1D, 60 epochs | 0.69649 IoU | revise |
| Stable-init B1D, 60 epochs | 0.70211 IoU | near tie, not accepted |
| UDER B2, 60 epochs | 0.69974 IoU | revise |
| UDER v2 B2V2, 60 epochs | 0.69623 IoU | rejected |
| Combined B3, 60 epochs | 0.69348 IoU | rejected from scratch |
| ConvNeXt-Tiny U-Net, 80 epochs | 0.76793 IoU, 0.84744 Dice | rejected |
| ResNet101 U-Net, 80 epochs | 0.77149 IoU, 0.84817 Dice | rejected |
| ResNet50 384px with frozen encoder BN, 40 epochs | 0.75614 IoU, 0.83605 Dice | rejected |
| UDER low-LR stabilization, 30 epochs | 0.77528 IoU, 0.85358 Dice | rejected |

BD-CoDec and UDER become positive only after the semantic backbone is
pretrained: FG initialized from A2G reaches 0.75855 IoU, +0.00753 over A2G.

## Split-3 Duplicate Audit

Run:

```powershell
E:\anaconda3\envs\my_pytorch\python.exe audit_busi_split.py
```

The split has no overlapping case names and no byte-identical image across
train/validation, but has 24 high-confidence cross-split near-duplicate pairs
at pHash distance <= 4 and thumbnail correlation >= 0.98. Their resized mask
annotations have mean pairwise IoU 0.83261 (range 0.11752 to 0.93912), showing
substantial annotation inconsistency for nearly identical inputs. The full
report is `runs/busi_split3_audit.json`.
