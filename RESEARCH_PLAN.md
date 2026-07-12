# BUSI U-Net Optimization Research Plan

## Evaluation Contract

- Use `busi_train3.txt` for training and `busi_val3.txt` for model selection and
  evaluation.
- Do not train, tune post-processing, or select hyperparameters on validation
  masks outside an explicitly logged experiment.
- Report per-image mean and global/micro metrics together. Always report the
  threshold and whether test-time augmentation or an ensemble is used.
- Keep the original U-Net result (IoU 69.55%, Dice 81.38%) as the fixed baseline.
- The split-3 audit found no cross-split byte-identical images, but found 24
  high-confidence cross-split near-duplicate pairs at pHash distance <= 4 and
  normalized thumbnail correlation >= 0.98. Results on the requested split 3
  must therefore be labeled as image-level rather than patient-deduplicated
  validation. The complete evidence is in `runs/busi_split3_audit.json`.
- A 95% IoU implies at least 97.44% Dice for the same binary masks because
  `Dice = 2 * IoU / (1 + IoU)`. The 95% target is an aspirational target, not a
  reason to change the split or leak validation labels.

## Evidence From Recent Work

1. SF-RecSAM (ECCV 2024) identifies low contrast and blurred boundaries as key
   BUS problems. Its spatial-frequency fusion combines Haar high-frequency
   components with multi-scale spatial convolutions, and its dual false
   corrector uses prediction uncertainty to correct FP/FN regions. It reports
   78.58% mIoU and 87.14% mDice on an 8:1:1 BUSI split.
2. CAU-Net (Pattern Recognition 2025) explicitly models noise interference,
   small tumors, and blurred boundaries with challenge-specific encoders,
   adaptive aggregation, graph reasoning, and deep supervision. It reports
   79.13% IoU and 87.27% Dice on BUSI.
3. Boundary regression and signed-distance supervision are useful because
   overlap losses weakly penalize small boundary displacements, especially for
   small lesions and blurred ultrasound boundaries.
4. The published BUSI results above remain far below 95% IoU under their own
   protocols. Any result near 95% must therefore receive extra leakage,
   duplicate-patient, threshold-selection, and metric-implementation checks.
5. PBNet (Medical Physics 2025) extracts semantic boundaries with pooling-based
   dilation/erosion, uses them to gate high/low-level feature fusion, and adds
   multilevel boundary supervision. This supports the boundary-guided decoder,
   but its reported gain over the second-best BUSI method is below one Dice
   point, so it does not support a 95% expectation.
6. A 2025 uncertainty study identifies duplicated BUSI images with inconsistent
   annotations and recommends radiologist-curated deduplication. It also uses a
   residual-encoder U-Net, deep supervision, MC dropout, and ensembles, but
   emphasizes calibration and cross-domain failure rather than inflated
   in-domain overlap.
7. A 2026 preprint weights boundary BCE by binary predictive entropy. It reduces
   false positives on normal BUSI images and calibration error, but does not
   significantly improve lesion Dice (0.7624 versus 0.7616). Because the current
   647-case benchmark excludes normal images, its specificity claim cannot be
   transferred directly; only its uncertainty-weighting mechanism is adapted.

Primary sources:

- [SF-RecSAM paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03309.pdf)
- [SF-RecSAM code](https://github.com/dodooo1/SFRecSAM)
- [CAU-Net paper](https://doi.org/10.1016/j.patcog.2025.111851)
- [A2DMN](https://doi.org/10.1109/ISBI56570.2024.10635867)
- [Boundary-regression multi-task network](https://doi.org/10.1016/j.bspc.2025.108828)
- [PBNet](https://doi.org/10.1002/mp.17647)
- [BUSI dataset correction](https://doi.org/10.1016/j.dib.2023.109247)
- [Trustworthy BUS uncertainty study](https://arxiv.org/abs/2508.17768)
- [Entropy-guided boundary supervision](https://arxiv.org/abs/2606.22308)

## Thesis Chapter 1: Challenge-Adaptive Semantic Relation Modeling

### Module 1: TriCAR - Tri-Challenge Adaptive Routing

Problem: noise interference, small tumors, and blurred boundaries require
different receptive fields; a single fixed encoder treats every sample alike.

Design: replace three full encoders with three lightweight residual experts:
noise suppression, small-lesion preservation, and boundary/context modeling.
A sample- and scale-dependent router mixes the experts. A lightweight channel
relation block at the bottleneck provides global consistency.

Hypothesis: TriCAR selects the right local processing path for each image and
scale, avoiding the cost and two-stage training of three complete encoders.

### Module 2: CGR-Bridge - Channel-Graph Relational Bridge

Problem: local challenge experts can recover lesion evidence but cannot ensure
that distant channels encode a globally consistent lesion representation.

Design: project the bottleneck to compact channel nodes, construct a normalized
cosine-relation graph in FP32, propagate node evidence, and return it through a
bounded learnable residual. FP32 graph construction and bounded scaling avoid
the AMP overflow observed in the first A3 run.

Hypothesis: CGR-Bridge aggregates complementary expert evidence into a coherent
global semantic representation before decoding. It is deliberately lightweight
so its gain cannot be attributed to a large parameter increase.

## Thesis Chapter 2: Boundary-Uncertainty Closed-Loop Refinement

### Module 3: BD-CoDec - Boundary-Distance Cooperative Decoder

Problem: region-only BCE/Dice supervision permits jagged contours, holes, and
large relative errors on small lesions.

Design: each decoder stage predicts region, morphological boundary, and signed
distance maps. Boundary gates filter skip features before fusion. Deep
supervision combines region overlap, boundary BCE, distance regression, and a
topology-aware term.

Hypothesis: BD-CoDec converts the challenge-aware representation into a
geometrically consistent coarse mask and explicitly exposes ambiguous boundary
features to Module 4.

### Module 4: UDER - Uncertainty-Driven Dual Error Refinement

Problem: the remaining errors are asymmetric FP and FN regions near shadowed
or ambiguous tissue.

Design: estimate uncertainty from stochastic feature perturbations or
flip/scale-consistent predictions. Two residual heads independently predict FN
addition and FP removal from the image, coarse logit, distance map, and
uncertainty map. The corrected logit is `coarse + FN - FP` and is trained
end-to-end rather than using a hand-tuned intensity rule.

Hypothesis: UDER closes the loop: uncertainty produced by disagreement is used
to correct only unreliable regions while preserving high-confidence interiors.

## Logical Chain

`TriCAR -> CGR-Bridge -> BD-CoDec -> UDER`

1. TriCAR selects how local evidence should be interpreted for each challenge.
2. CGR-Bridge enforces global semantic consistency across expert channels.
3. BD-CoDec converts semantic evidence into boundary-consistent geometry.
4. UDER detects and corrects the remaining FP/FN errors.

## Required Ablations

| ID | TriCAR | CGR-Bridge | BD-CoDec | UDER | Purpose |
| --- | --- | --- | --- | --- | --- |
| A0 | | | | | Reproduced U-Net baseline |
| A2R | X | | | | Local challenge routing contribution |
| A2G | | X | | | Global graph bridge contribution |
| A2 | X | X | | Chapter 1 complete model |
| B1 | | | X | | Boundary-gating contribution |
| B1D | | | X | | Add signed-distance supervision inside BD-CoDec |
| B2 | | | | X | Uncertainty correction contribution |
| B3 | | | X | X | Chapter 2 complete model |
| F | X | X | X | X | Full model |

Every ablation uses the same split, seed policy, augmentation, epoch budget,
checkpoint rule, and evaluation script. Improvements must be confirmed on both
global and per-image metrics, with special attention to small, low-contrast,
and boundary-complexity subsets.

## Completed Chapter-1 Pilot (100 Epochs, Seed 41)

| ID | Best epoch | IoU | Dice | Outcome |
| --- | ---: | ---: | ---: | --- |
| A0 | 97 | 0.72445 | 0.80718 | chapter baseline |
| A2R (TriCAR) | 83 | 0.74340 | 0.82640 | +1.90 IoU points |
| A2G (CGR-Bridge) | 88 | 0.75102 | 0.83559 | +2.66 IoU points |
| A2 (TriCAR + CGR) | 91 | 0.74935 | 0.83234 | +2.49 IoU points |

The sequential chapter-1 chain is positive: A0 (0.72445) -> TriCAR (0.74340)
-> TriCAR+CGR (0.74935). CGR alone is stronger (0.75102), so interaction tuning
remains necessary and both independent and sequential results must be reported.

## Rejected Chapter-1 Pilots

- Legacy SFS-Cal: 0.71458 mIoU after 100 epochs, negative versus A0.
- Shrinkage SFS-Cal v2: 0.71329 mIoU after 100 epochs, also negative.
- LSSG local-statistics speckle guard: stopped at epoch 44 after reaching only
  0.6689 while the A0 trajectory had already reached 0.6840 by epoch 39.

These pilots are retained in code and run histories for reproducibility but are
not presented as thesis contributions.

## Chapter-2 Screening Status

At a common 60-epoch screening budget, the deep-supervision control B0 reaches
0.70385 mIoU. B1 (0.69682), B1D (0.69649), B2 (0.69974), and B3 (0.69348) are
negative when trained from scratch. Stable identity initialization improves
B1D to 0.70211 but does not exceed B0. UDER v2 falls to 0.69623 and is rejected.

The modules become useful in their intended sequence after semantic pretraining:
FG initialized from A2G reaches 0.75855 IoU and 0.84139 Dice, improving A2G
by 0.00753 IoU. This supports the semantic-to-geometry dependency claimed by
the four-module chain, while the negative from-scratch results must remain in
the ablation report.

## Strong-Backbone Transfer Check

A non-destructive ResNet50 experiment initializes from the best split-3
ResNet50 U-Net and adds CGR-Bridge, auxiliary boundary/distance supervision,
and zero-output BD-CoDec/UDER corrections. Initialization was verified exactly
(`max_abs_diff = 0`). Two 60-epoch seeds reach 0.77001 and 0.77839 per-image
IoU. The best seed reaches 0.85577 Dice.

A continued fine-tuning control at the same seed, initialization, backbone
learning rate, augmentation, and epoch budget reaches 0.77526 IoU and
0.85211 Dice. The net module effect is therefore +0.00313 IoU and +0.00366 Dice. This is
evidence of better case-level balance, not a universal overlap improvement. It
is not the complete four-module chain because TriCAR is absent. Strong-backbone
component ablations and at least one more seed are required before attributing
the gain to a specific module.

## Strong-Backbone Ablation Update

At a matched 20-epoch budget, the fine-tuning control reaches 0.77348 IoU and
0.85049 Dice. CGR-only reaches 0.77640/0.85449, BD-CoDec-only reaches
0.77650/0.85461, deep supervision reaches 0.77485/0.85269, and UDER with deep
supervision reaches 0.77875/0.85594. UDER therefore contributes +0.00390 IoU
and +0.00324 Dice beyond deep supervision alone and becomes the current primary
optimization branch.

Increasing backbone capacity is not effective on this split: ConvNeXt-Tiny
U-Net reaches 0.76793/0.84744 and frozen-BN ResNet101 U-Net reaches
0.77149/0.84817. A 384-pixel frozen-BN adaptation reaches only
0.75614/0.83605. These routes are rejected rather than merged into the thesis
model.

A lower-learning-rate UDER run (encoder 1e-5, modules 5e-5) reaches only
0.77528 IoU and 0.85358 Dice. The smoother trajectory does not compensate for
weaker early adaptation, so the original differential learning rates are kept.

The current honest single-model best is 0.77875 IoU and 0.85594 Dice. The 0.90
target remains a research objective, not an achieved or guaranteed result.
