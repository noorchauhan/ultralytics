# Stereo 3D Detection Experiments Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Experiment Timeline](#3-experiment-timeline)
4. [Results Summary](#4-results-summary)
5. [Detailed Experiment Notes](#5-detailed-experiment-notes)
6. [Key Findings](#6-key-findings)
7. [Error Analysis](#7-error-analysis)
8. [Failed Approaches](#8-failed-approaches)
9. [Current Best Configuration](#9-current-best-configuration)

---

## 1. Project Overview

This project implements stereo 3D object detection on KITTI by extending the Ultralytics YOLO framework. The model takes a 6-channel stereo image (left RGB + right RGB) and predicts 3D bounding boxes with class, dimensions, orientation, and depth.

**Task**: Predict 3D bounding boxes (center, dimensions, orientation) from calibrated stereo image pairs.

**Dataset**: KITTI stereo — 3 classes (Car, Pedestrian, Cyclist), ~3,700 training / ~3,769 validation images.

**Primary metric**: AP3D@0.5 (3D Average Precision at 0.5 IoU threshold).

**Hardware**: 8x GPU training, batch size 128 (n/s) or 32 (m), image size 384x1248.

---

## 2. Architecture

### 2.1 Backbone: Group-Conv Stem

The backbone uses a **groups=2 group-convolution stem** for the first two layers (stride 2 and stride 4). This processes left and right image features independently in early layers, allowing the network to learn view-specific low-level features before merging at layer 2 (C3k2).

```
Layer 0: Conv(6→64, k=3, s=2, groups=2)   # P1/2 — left/right independent
Layer 1: Conv(64→128, k=3, s=2, groups=2) # P2/4 — left/right independent
Layer 2: C3k2(128→256)                     # Features merge here
Layer 3+: Standard YOLO11 backbone         # P3/8, P4/16, P5/32
```

### 2.2 Stereo Cost Volume (v17+)

The `StereoCostVolume` module computes sparse dot-product correlation between left and right features at 24 discrete disparity offsets (0 to 48 feature pixels at stride 4):

```python
class StereoCostVolume(nn.Module):
    # Takes groups=2 features from backbone layer 1 (stride 4)
    # Splits into left (c//2) and right (c//2) channels
    # Computes normalized dot-product at each disparity offset
    # Refines: num_bins → c2 channels, downsamples stride 4 → stride 8
    # Output: [B, 64, H/8, W/8]
```

**Key design decision (v20+)**: The cost volume is routed **only to depth branches** (lr_distance, depth) in the head, NOT merged into P3. This avoids the 2D-3D task conflict where stereo correlation features dilute 2D detection quality.

**Parameter overhead**: ~75K params for s-size model (+0.7%).

### 2.3 Detection Head: Stereo3DDetHeadYOLO11

Multi-scale head (P3/P4/P5) with auxiliary branches for 3D prediction:

| Branch | Channels | Target | Encoding |
|--------|----------|--------|----------|
| `lr_distance` | 1 | Stereo disparity | Log-space: `log(cx_left - cx_right)` |
| `depth` | 16 | Direct depth | DFL bins: `linspace(log(2m), log(80m), 16)`, decoded via softmax weighted sum |
| `dimensions` | 3 | Object size (H,W,L) | Normalized offset: `(dim - mean) / std` |
| `orientation` | 2 | Heading angle | `[sin(alpha), cos(alpha)]` |

Plus standard YOLO Detect outputs: boxes (xyxy), class scores, DFL features.

The cost volume (when present) is concatenated **only with P3 features for depth branches** (lr_distance, depth). The 2D detection branches (box, cls) and geometric branches (dimensions, orientation) see clean P3/P4/P5 features.

### 2.4 Loss Function: Stereo3DDetLossYOLO11

6-component loss vector: `[box, cls, lr_dist, depth, dims, orient]`

- **2D Detection**: Standard YOLO box + cls losses via Task-Aligned Assigner (TAL)
- **Auxiliary 3D**: DFLoss for depth bins; smooth L1 for lr_distance, dimensions, orientation — all on TAL-assigned positive locations
- **Loss weights** (from YAML): `lr_distance=2.0, depth=1.0, dims=1.0, orient=1.0`

### 2.5 Depth Decoding (Inference)

Dual depth sources fused via geometric mean:

```
z_from_disparity = (fx * baseline) / (exp(lr_log) * input_w / letterbox_scale)
z_from_direct    = exp(depth_log)
z_final          = sqrt(z_disp * z_direct)   # geometric mean
```

### 2.6 Post-Processing Pipeline

1. **NMS** on 2D Detect outputs
2. **Aux map sampling** at kept indices for 3D attributes
3. **3D decoding**: depth, dimensions, orientation → Box3D
4. **(Optional) Geometric construction**: Gauss-Newton refinement with 7 constraint equations
5. **(Optional) Dense alignment**: Photometric patch matching for sub-pixel depth refinement

### 2.7 Siamese/Weight-Shared Backbone (v36+)

Alternative to the groups=2 stem: a standard 3-channel backbone processes left and right images separately via a **batch dimension trick**, with 100% shared weights.

```
Forward pass:
1. Split 6ch input → left [B,3,H,W], right [B,3,H,W]
2. Stack as batch: stereo = cat([left, right], dim=0) → [2B, 3, H, W]
3. Run backbone layers 0..tap_layer on [2B, 3, H, W] (shared weights)
4. Split at tap: left_feat = out[:B], right_feat = out[B:]
5. Pass (left_feat, right_feat) tuple to StereoCostVolume
6. Continue layers tap_layer+1..end with LEFT-ONLY features [B, C, H, W]
7. Head sees clean left-only P3/P4/P5 + cost_vol — identical interface
```

**Benefits**:
- **100% pretrained weight compatible** — standard 3ch YOLO26 backbone, no groups=2 stem modification needed
- **Clean left-only features** — 2D detection sees only left image, matching mono YOLO evaluation
- **Same cost volume quality** — L/R features computed by identical (shared) weights at stride 4

**Implementation**: `_predict_once` override in `Stereo3DDetModel` (model.py). `StereoCostVolume.forward` accepts both `(left, right)` tuple (siamese) and single tensor (groups=2) for backward compatibility. Model built with `ch=3` when `siamese: true` in YAML.

---

## 3. Experiment Timeline

### Phase 1: Baseline & Training Recipe (v3–v7)

| Version | Changes | Optimizer | Epochs | Aug | Sizes |
|---------|---------|-----------|--------|-----|-------|
| v3 | Cosine LR, patience=300 | AdamW lr=0.005 | 300 | No | n,s,m,l,x |
| v4 | Higher LR (lr0=0.01) | AdamW | 300 | No | n,s,m |
| v5 | Short schedule + cosine (best of v2+v3) | AdamW lr=0.005 | 100 | No | n,s,m,l,x |
| v6 | Long schedule (v5 start + v3 tail) | AdamW lr=0.005 | 300 | No | n,s,m |
| v7 | 2-stage: v5 weights → SGD finetune | SGD lr=0.01 | 200 | No | n,s,m,l,x |

### Phase 2: Loss & Module Experiments (v8–v11)

| Version | Changes | Result vs v5 |
|---------|---------|-------------|
| v8 | Laplacian uncertainty loss (2-ch depth branches) | Neutral (n), catastrophic (s) |
| v8b | Laplacian with balanced weights, tighter clamp | Slightly worse |
| v9 | StereoCorrelation + Laplacian + distance-aware loss | n +2pp, s -7pp, m -6pp |
| v10 | Corr + smooth_l1 + sigma supervision | n -6pp (all changes hurt) |
| v11 | Correlation module ONLY (original loss) | Neutral (n -1pp, s -1pp) |

### Phase 3: Augmentation & Depth Rebalancing (v12–v14)

| Version | Changes | Sizes |
|---------|---------|-------|
| v12 | Enable augmentation (fliplr, HSV) + rebalance depth weights (lr_dist 0.2→2.0, depth 0.1→1.0) + 200ep | n,s |
| v13 | SGD finetune from v12 checkpoints | n,s |
| v14 | SGD from scratch (300ep, no AdamW warmup) | n,s |

### Phase 4: Architecture Comparison (v15–v16)

| Version | Changes | Sizes |
|---------|---------|-------|
| v15 | YOLO26 vs YOLO11 backbone comparison (200ep AdamW) | n,s |
| v16 | SGD finetune from v15 weights | n,s |

### Phase 5: Cost Volume (v17–v21)

| Version | Changes | Sizes |
|---------|---------|-------|
| v17 | StereoCostVolume (merged into P3), 200ep AdamW | n,s |
| v18 | SGD finetune from v17 weights | n,s |
| v19 | m-size cost volume (batch=32, lr=0.001) | m |
| v20 | Cost volume routed to depth branches ONLY (not merged into P3) | n,s,m |
| v21 | SGD finetune ALL cost volume models (merged vs depth-only) | n,s,m |

### Phase 6: RAFT-Stereo & Distillation — All Failed (v22–v25)

| Version | Changes | Sizes |
|---------|---------|-------|
| v22 | 7-channel depth prior (RAFT-Stereo precomputed disparity as input) | n,s |
| v23 | Disparity distillation from teacher (weight=1.0) | n,s |
| v24 | Disparity distillation (weight=0.05) | n,s |
| v25 | Feature distillation (MSE on P3 features from teacher) | n,s |

### Phase 7: Long Training & Bug Fixes (v26–v30)

| Version | Changes | Sizes |
|---------|---------|-------|
| v26 | 1000ep SGD from v21 (extended training) | s |
| v27 | Multi-scale cost volume pyramid ablations | s |
| v28 | Multi-scale + disparity loss (w=1.0) | s |
| v29 | Multi-scale + disparity loss (w=0.05) / no disp | s |
| v30 | v26 rerun with fitness function bugfix | s |

**Critical bug discovered in v30**: The fitness function used to select `best.pt` was based on **2D mAP50-95**, not AP3D@0.5. When `get_stats()` merged `{**3d_metrics, **det_metrics}`, the 2D `fitness` key overwrote the 3D one. This means all experiments prior to v30 saved `best.pt` based on 2D performance, not 3D. For v26, the true best AP3D@0.5 was 46.4% at epoch 550, but `best.pt` only had 25.5% (wrong epoch).

**Fix**: Pop `fitness` from det_metrics before merge, put 3D metrics last. Also fixed per-threshold GT matching and relaxed depth clamp.

**Multi-scale cost volume (v27-v29) — all failed**:
- v28 (ms + disp w=1.0): s=39.5% — disparity loss dominated 87% of gradient
- v29 (ms + disp w=0.05): s=39.6% — same result even with balanced loss
- v29 (ms + no disp): s=38.9% — architecture itself hurts, not just the loss
- Single-scale baseline: s=46.4% (v26/v30)
- **Conclusion**: Fine-grained stride-4 single-scale > coarser multi-scale pyramid. All multi-scale code removed.

### Phase 8: Depth Output Representation (v31–v33)

| Version | Changes | s-size | Status |
|---------|---------|--------|--------|
| v31 | DFL-style depth bins (16 bins in log-space) | **53.0%** | Done |
| v32 | Depth residual (16 bins + 1 regression offset) | 48.8% | Failed, reverted |
| v33a | Depth loss weight 3.0 (was 1.0) | **56.5%** | Done |
| v33b | 32 depth bins (was 16) | 56.2% | Done |

### Phase 9: YOLO26 Backbone, Evaluation & Siamese Architecture (v34–v36)

| Version | Changes | n | s | m | l |
|---------|---------|---|---|---|---|
| v34 | YOLO26 backbone + costvol, all sizes, R40 eval | 42.4% | **49.8%** | 49.0% | 47.5% |
| v35 | v34 + occlusion filter OFF (train on all objects) | — | **48.1%** | — | — |
| v36 | Siamese weight-shared backbone (batch trick) | 48.1% | 48.3% | 49.0% | **50.9%** |

All v34–v36 numbers are KITTI R40 Moderate mean AP3D@0.5.

**v34**: Switched from YOLO11 to YOLO26 backbone with cost volume. Trained all sizes (n/s/m/l) with SGD 1000ep, same recipe as v30/v31. s-size is sweet spot; m/l overfit on small KITTI dataset. Introduced KITTI-standard R40 evaluation with difficulty splits (Easy/Moderate/Hard).

**v35**: Disabled occlusion filter (`filter_occluded=False`) — trains on ALL objects including heavily occluded (DontCare). Result: Moderate -1.7pp but Hard **+3.5pp** @0.5. Car Hard@0.5: 55.7%→67.9% (+12.2pp). Better robustness to difficult cases.

**v36**: Siamese/weight-shared backbone (see §2.7). Standard 3ch YOLO26 backbone processes L/R via batch trick [2B,3,H,W]. 100% pretrained weight compatible. R40 results match v35 across all sizes; l-size (50.9%) is new best. Car-only (nc=1) training failed — depth branch collapses with single-class TAL (see §5 notes).

---

## 4. Results Summary

### 4.1 Complete Results Table

All numbers are **AP3D@0.5** (%) — best validation epoch.

| Experiment | n-size | s-size | m-size | Key Change |
|------------|--------|--------|--------|------------|
| **Phase 1: Training Recipe** |
| v5 (baseline) | 17.5 | 18.6 | 16.6 | 100ep AdamW, no aug |
| v7 (SGD finetune) | **26.7** | — | — | v5 + 200ep SGD |
| **Phase 2: Loss & Modules** |
| v8 (Laplacian) | 17.3 | 7.5 | — | Laplacian NLL → negative losses |
| v9 (corr+Laplacian+dist) | 19.6 | 12.0 | 10.7 | Too many changes at once |
| v10 (corr+fixed loss) | 11.6 | 12.0 | — | Smooth L1 + sigma, still hurt |
| v11 (corr only) | 16.5 | 17.3 | — | StereoCorrelation is neutral |
| **Phase 3: Aug + Depth Rebalancing** |
| v12 (aug+depth weights) | 18.4 | 19.0 | — | Augmentation + depth 0.1→1.0 |
| v13 (v12 + SGD) | 24.7 | **27.4** | — | SGD finetune from v12 |
| v14 (SGD scratch) | 22.3 | 26.2 | — | Pure SGD 300ep |
| **Phase 4: Architecture** |
| v15 (YOLO26 AdamW) | 17.9 | — | — | YOLO26 backbone |
| v16 (YOLO11 SGD) | — | 27.1 | — | SGD from v15 YOLO11 |
| **Phase 5: Cost Volume** |
| v17 (costvol merged) | 30.6 | 36.5 | — | StereoCostVolume in P3 |
| v18 (costvol SGD) | 31.7 | 35.6 | — | SGD finetune from v17 |
| v19 (costvol m-size) | — | — | 37.7 | m-size, batch=32, lr=0.001 |
| v20 (depth-only routing) | 28.8 | 32.7 | 33.5 | Cost vol → depth branches only |
| v21 merged SGD | 31.7 | 35.6 | 38.9 | SGD from v17/v19 |
| v21 depth-only SGD | **34.0** | **40.2** | 37.7 | SGD from v20 |
| **Phase 6: RAFT-Stereo (all failed)** |
| v22 (depth prior 7ch) | 34.5 | 39.0 | — | RAFT-Stereo as 7th channel |
| v25 (feature distill SGD) | 31.1 | 33.0 | — | MSE distillation on P3 |
| **Phase 7: Long Training & Bug Fixes** |
| v26 (1000ep SGD) | — | 46.4† | — | Extended training (best.pt wrong due to bug) |
| v28 (multi-scale + disp) | — | 39.5 | — | Multi-scale pyramid + disparity loss |
| v29 (multi-scale no disp) | — | 38.9 | — | Multi-scale pyramid hurts even without disp loss |
| v30 (v26 + fitness fix) | — | **46.4** | — | Correct best.pt selection by AP3D@0.5 |
| **Phase 8: Depth Output Representation** |
| v31 (DFL depth bins) | 41.3 | **53.0** | — | 16 bins in log-depth, DFLoss |
| v32 (depth residual) | — | 48.8 | — | 16 bins + 1 residual → regression hurt bins |
| v33a (depth weight 3.0) | — | **56.5** | — | Depth loss 1.0 → 3.0 |
| v33b (32 bins) | — | 56.2 | — | 16 → 32 DFL bins (≈same as 16) |
| **Phase 9: YOLO26 + Siamese** |
| v34 (YOLO26 costvol, R40) | 42.4 | **49.8** | 49.0 | YOLO26 backbone, R40 eval |
| v35 (filter OFF, R40) | — | **48.1** | — | Occlusion filter disabled |
| v36 (siamese, R40) | 48.1 | 48.3 | 49.0 | Weight-shared 3ch backbone |
| v36-l (siamese, R40) | — | — | — | l-size: **50.9%** (new best R40) |

†v26 best AP3D@0.5 was 46.4% at ep550, but best.pt saved wrong epoch due to fitness bug.

### 4.2 Top 5 Results (R40 Moderate Mean)

| Rank | Experiment | AP3D@0.5 | AP3D@0.7 | Eval | Best Epoch |
|------|-----------|----------|----------|------|------------|
| 1 | **l_yolo26_siamese_v36** | **50.9%** | **31.6%** | R40 | — |
| 2 | s_yolo26_costvol_v34 | 49.8% | 30.7% | R40 | — |
| 3 | m_yolo26_siamese_v36 | 49.0% | 29.1% | R40 | — |
| 4 | s_yolo26_siamese_v36 | 48.3% | 29.4% | R40 | — |
| 5 | s_yolo26_costvol_v35 (filter OFF) | 48.1% | 30.1% | R40 | 210 |

**Note**: R40 and non-R40 metrics are not directly comparable. R40 uses KITTI-standard 40-point interpolation with difficulty splits; non-R40 uses COCO-style 101-point interpolation without difficulty filtering. R40 numbers are generally lower.

### 4.3 Progression Chart (s-size AP3D@0.5)

```
AP3D@0.5 (s-size, non-R40 unless noted)

 57% ┤                                                                          ■ v33a depth_w=3
     │                                                                          ■ v33b 32 bins
 55% ┤                                                                    ■ v31 depth bins
     │
 50% ┤                                                              ✗ v32 depth residual (reverted)
     │                                                                               ■ v34 R40*
     │                                                                               ■ v35 R40*
 45% ┤                                                        ■ v30 fitness fix
     │
 40% ┤                                     ■ v21 depth-only SGD
     │                                  ■ v22 depth prior
     │                               ✗ v28 multi-scale
 35% ┤                           ■ v17 costvol merged
     │                              ■ v18 SGD
     │                    ■ v25 feat distill
 30% ┤
     │             ■ v13 SGD finetune
 25% ┤                ■ v14 SGD scratch
     │
 20% ┤  ■ v5 baseline
     │  ■ v12 aug+depth
 15% ┤
     │      ■ v9 correlation
 10% ┤      ■ v10 fixed loss
     │   ■ v8 Laplacian
     ┼──────────────────────────────────────────────────────────────────────────────
       v5  v8-11 v12-14 v17-21 v22-25 v26-30 v31-33          v34-v36
                                                         *R40 not comparable
```

---

## 5. Detailed Experiment Notes

### v5 — Baseline (100ep AdamW, No Augmentation)

**Goal**: Establish baseline with short, efficient schedule.

**Config**: AdamW lr=0.005, cosine decay to lr*0.01, 100 epochs, no augmentation, groups=2 stem.

**Results**: n=17.5%, s=18.6%, m=16.6%

**Notes**: Fast convergence. Serves as control for all subsequent experiments. Model sizes beyond s don't help much — suggests the bottleneck is not model capacity but depth estimation quality.

---

### v7 — SGD Finetune (200ep SGD from v5)

**Goal**: Test 2-stage training (AdamW warmup → SGD refinement).

**Config**: Load v5 best weights → SGD lr=0.01, cosine, 200 epochs.

**Results**: n=26.7% (+9.2pp over v5)

**Key insight**: SGD finetuning is consistently the single most impactful training recipe change. The AdamW stage provides fast initial convergence; SGD refines with better generalization.

---

### v8 — Laplacian Uncertainty Loss

**Goal**: Let the network learn per-sample uncertainty to down-weight hard/ambiguous depth predictions.

**Changes**:
- Doubled channels for lr_distance (1→2) and depth (1→2) — extra channel predicts log_sigma
- Replaced smooth_l1 with Laplacian NLL: `loss = |target - pred| * exp(-log_sigma) + log_sigma`

**Results**: n=17.3% (neutral), s=7.5% (catastrophic)

**Failure analysis**: The Laplacian NLL loss went negative from epoch 4 onwards. The network learned to predict very large log_sigma values, effectively setting loss to 0 and stopping learning for depth branches. The log_sigma clamp range was too wide.

**v8b** attempted tighter clamping [-2, 2] and reduced weights but still underperformed baseline.

**Lesson**: Laplacian NLL is unstable for this task. The smooth_l1 loss is more robust.

---

### v9 — StereoCorrelation + Uncertainty Fusion + Distance-Aware Loss

**Goal**: Address depth bottleneck with explicit stereo correlation module.

**Changes** (3 simultaneous):
1. `StereoCorrelation` module: group-wise correlation at 24 disparity levels after stem
2. Uncertainty-weighted depth fusion (precision-weighted average) in decode
3. Laplacian NLL + distance-aware weighting (closer objects weighted higher)

**Results**: n=19.6% (+2.1pp), s=12.0% (-6.6pp), m=10.7% (-5.9pp)

**Failure analysis**: Combined too many changes. The Laplacian NLL was broken (same issue as v8). Nano model improved because the simpler architecture is more robust to loss instabilities, but larger models diverged.

**Lesson**: Never combine multiple experimental changes. Isolate each variable.

---

### v10 — Correlation + Fixed Loss

**Goal**: Isolate correlation benefit with stable loss function.

**Changes**: Same architecture as v9 but replaced Laplacian NLL with smooth_l1 + sigma supervision for decode-time uncertainty fusion.

**Results**: n=11.6% (-5.9pp vs v5)

**Failure analysis**: The sigma supervision and distance-aware weighting still interfered with training. Even with a stable base loss, the additional supervision signals created optimization conflicts.

---

### v11 — Correlation Only (Original Loss)

**Goal**: Isolate the pure `StereoCorrelation` module effect with zero other changes.

**Changes**: Only added StereoCorrelation after group-conv stem. Original smooth_l1 loss, geometric mean decode.

**Results**: n=16.5% (-1.0pp), s=17.3% (-1.3pp)

**Conclusion**: The `StereoCorrelation` module (inserted into the main feature pathway) is **neutral at best**. It doesn't help because the backbone already learns implicit stereo matching through the group-conv stem. This inline correlation approach was abandoned in favor of the cost volume approach (v17+).

---

### v12 — Augmentation + Depth Loss Rebalancing

**Goal**: Apply three config-only improvements simultaneously.

**Changes**:
1. Enable stereo augmentation: `fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3`
2. Rebalance depth loss weights: `lr_distance: 0.2→2.0, depth: 0.1→1.0`
3. Longer training: 100→200 epochs

**Results**: n=18.4% (+0.9pp), s=19.0% (+0.4pp)

**Notes**: Modest improvement alone, but these changes provide a better foundation for SGD finetuning (v13). The depth weight rebalancing was important — previously depth loss was 75x smaller than box loss (0.1 vs 7.5), meaning the network barely optimized for depth.

---

### v13 — SGD Finetune from v12

**Goal**: Apply proven SGD recipe to augmented + rebalanced weights.

**Config**: Load v12 best → SGD lr=0.01, cosine, 200 epochs, same augmentation.

**Results**: n=24.7% (+6.3pp over v12), s=27.4% (+8.4pp over v12)

**Key insight**: The s-size model finally overtakes n-size when combining augmentation + depth rebalancing + SGD finetune. This becomes the best non-cost-volume result.

---

### v14 — SGD from Scratch

**Goal**: Test whether pure SGD matches 2-stage (AdamW → SGD) training.

**Config**: SGD lr=0.01, cosine, 300 epochs from scratch (matching total epoch budget of v12+v13).

**Results**: n=22.3%, s=26.2%

**Conclusion**: Pure SGD is ~2pp worse than 2-stage training. The AdamW warmup provides a better starting point for SGD refinement. Two-stage training is the preferred recipe.

---

### v15 — YOLO26 vs YOLO11

**Goal**: Compare YOLO26 (newer) vs YOLO11 backbone architectures.

**Changes**: YOLO26 uses shortcut=True in neck C3k2, enhanced SPPF, C2PSA. Same head class.

**Results**: YOLO26 n=17.9% vs YOLO11 n=16.7% (200ep AdamW)

**Conclusion**: YOLO26 provides marginal improvement (~1pp) in the AdamW stage. Not enough to justify the architecture change.

---

### v16 — SGD Finetune from v15

**Results**: YOLO11 s=27.1% after SGD finetune

**Conclusion**: After SGD finetuning, YOLO11 and YOLO26 converge to similar performance. Architecture is not the bottleneck.

---

### v17 — StereoCostVolume (Merged into P3)

**Goal**: Provide explicit stereo disparity information to the detection head via a learned cost volume.

**Architecture**: `StereoCostVolume` takes groups=2 features from backbone layer 1 (stride 4), computes dot-product correlation at 24 disparity offsets (0-48 feature pixels), refines to 64 channels, and downsamples to stride 8. Output is concatenated with P3 in the PAN neck.

**Config**: 200ep AdamW, same augmentation as v12+.

**Results**: n=30.6% (+12.2pp over v12), s=36.5% (+17.5pp over v12)

**This was the single biggest improvement in the entire experiment series.** The cost volume provides the network with explicit stereo matching cues that the backbone cannot learn efficiently on its own. Unlike the v9-v11 `StereoCorrelation` (which was inserted inline and diluted backbone features), the cost volume is a separate pathway that enriches the head with pure disparity information.

---

### v18 — Cost Volume SGD Finetune

**Results**: n=31.7% (+1.1pp over v17), s=35.6% (-0.9pp)

**Notes**: SGD provided modest gains for n-size but slightly hurt s-size. The cost volume model may already be well-optimized by AdamW for 200 epochs.

---

### v19 — m-Size Cost Volume

**Config**: m-size model, batch=32 (memory limited), lr=0.001 (lower for larger model).

**Results**: m=37.7%

**Notes**: Medium model provides +1.2pp over s-size v17, but requires 4x smaller batch size and longer training.

---

### v20 — Depth-Only Cost Volume Routing

**Goal**: Solve the 2D-3D task conflict. v19's m-size model had AP3D=37.7% but mAP50(2D)=63.7%. The cost volume features diluted P3, hurting 2D box detection.

**Architecture change**: Cost volume output is passed as a **4th input** to the head (alongside P3, P4, P5). The head concatenates it only with P3 features for the depth branches (lr_distance, depth). The 2D detection branches and geometry branches see clean P3/P4/P5.

```yaml
# Head receives [P3, P4, P5, cost_vol]
# cost_vol concatenated ONLY with P3 for lr_distance and depth branches
# box, cls, dimensions, orientation see clean P3/P4/P5
```

**Results**: n=28.8%, s=32.7%, m=33.5%

**Notes**: Initial AP3D is slightly lower than merged (v17), but mAP50(2D) recovers significantly. The hypothesis was that SGD finetuning (v21) would close the AP3D gap while maintaining better 2D detection.

---

### v21 — SGD Finetune: Merged vs Depth-Only

**Goal**: Fair comparison of merged vs depth-only cost volume after SGD finetuning.

**Results**:

| Size | Merged (v21) | Depth-Only (v21) | Winner |
|------|-------------|------------------|--------|
| n | 31.7% | **34.0%** | Depth-only +2.3pp |
| s | 35.6% | **40.2%** | Depth-only +4.6pp |
| m | **38.9%** | 37.7% | Merged +1.2pp |

**Conclusion**: Depth-only routing wins convincingly at n/s sizes. The clean P3 features give the 2D detector better anchoring, which cascades to better 3D predictions because depth branches use higher-quality TAL assignments. The s-size depth-only model at **40.2% AP3D@0.5** is the overall best result.

For m-size, merged still wins — possibly because the larger model has enough capacity to handle the mixed features.

---

### v22 — Depth Prior (7-Channel RAFT-Stereo Input)

**Goal**: Provide precomputed high-quality disparity from RAFT-Stereo as a 7th input channel.

**Architecture**: Standard 6-channel model with groups=1 first layer (no group-conv stem since the 7th channel is already a depth signal). RAFT-Stereo disparity maps precomputed offline, scaled to [0,255] uint8.

**Results**: n=34.5%, s=39.0%

**Conclusion**: Competitive with cost volume (s: 39.0% vs 40.2%) but requires offline RAFT-Stereo inference (~3s/frame). Since the learned cost volume matches or exceeds this without the external dependency, the depth prior approach was abandoned.

---

### v23–v25 — Knowledge Distillation (All Failed)

**Goal**: Transfer depth knowledge from a teacher model (trained with RAFT-Stereo depth prior) to a student model (6-channel only).

| Version | Method | n-size | s-size | vs v21 baseline |
|---------|--------|--------|--------|-----------------|
| v23 | Disparity distillation (weight=1.0) | 28.1% | 33.9% | -6pp / -6pp |
| v24 | Disparity distillation (weight=0.05) | 28.5% | 33.4% | -6pp / -7pp |
| v25 | Feature distillation (MSE on P3) | 31.1% | 33.0% | -3pp / -7pp |

**Failure analysis**: All distillation approaches significantly hurt performance. The teacher's depth prior features are fundamentally different from the student's learned stereo features. Forcing the student to mimic the teacher's representations interferes with its own stereo matching learning. The distillation loss competes with the primary 3D losses rather than complementing them.

**All RAFT-Stereo code (depth prior, distillation) was removed from the codebase** after these experiments confirmed no benefit.

---

### v26 — 1000-Epoch SGD Training

**Goal**: Test whether much longer training improves the cost volume model.

**Config**: s-size, SGD lr=0.01, cosine to 0.01, 1000 epochs from v21 checkpoint, same augmentation.

**Results**: s=46.4% AP3D@0.5 at epoch 550

**Notes**: A significant +6.2pp improvement over v21 (40.2%), confirming that the model was still underfitting at 200ep SGD. However, a critical bug meant `best.pt` was saved at the wrong epoch (see v30). The true peak was identified by manual analysis of validation logs.

---

### v27–v29 — Multi-Scale Cost Volume Pyramid (All Failed)

**Goal**: Provide multi-resolution stereo cues with cost volumes at multiple strides (1/4, 1/8, 1/16).

**Architecture**: Three cost volumes at increasing strides, each providing disparity features at different spatial resolutions. Hypothesis: coarse cost volumes capture large disparities (nearby objects), fine cost volumes capture small disparities (distant objects).

**Results**:
- v28 (ms + disp w=1.0): s=39.5% — disparity loss consumed 87% of total gradient
- v29 (ms + disp w=0.05): s=39.6% — balanced loss didn't help
- v29 (ms + no disp loss): s=38.9% — architecture itself hurts

**Why it failed**: The multi-scale pyramid introduced redundant, coarser-resolution features that diluted the fine-grained stride-4 cost volume signal. The single-scale approach preserves the highest spatial resolution where sub-pixel disparity estimation is most precise. Additionally, the added parameters and computation didn't translate to better gradient signal — the network had more capacity but worse information quality.

**Conclusion**: Single-scale stride-4 cost volume is optimal. All multi-scale code was removed from the codebase.

---

### v30 — Fitness Function Bugfix

**Goal**: Re-run v26 setup with corrected model selection.

**Bug**: `val.py:get_stats()` merged 3D and 2D metrics dicts with `{**3d_metrics, **det_metrics}`. Both contained a `fitness` key. Because det_metrics came second, the 2D `fitness` (based on mAP50-95) **silently overwrote** the 3D `fitness` (based on AP3D@0.5). Result: `best.pt` was saved when 2D detection peaked, not when 3D detection peaked.

**Impact**: v26's saved `best.pt` had AP3D@0.5 of only 25.5%, despite the model reaching 46.4% at epoch 550.

**Fix**: Pop `fitness` from det_metrics before merge; place 3D metrics last to ensure 3D fitness takes priority. Also fixed per-threshold GT matching and relaxed depth clamp from 1m to ~0m.

**Results**: s=46.4% — matches v26's true peak, confirming the fix works correctly.

---

### v31 — DFL-Style Depth Bin Classification

**Goal**: Replace scalar depth regression with classification over depth bins, using Distribution Focal Loss (DFL).

**Motivation**: DFL is proven in YOLO's bbox regression (reg_max=16). The same principle — learn a discrete distribution over bin values, then decode via weighted sum — should apply to depth prediction. Classification avoids the gradient magnitude issues of regression on log-depth scalars.

**Architecture changes**:
- Depth branch outputs 16 channels (was 1) representing bin logits
- Bins span log-depth range: `linspace(log(2m), log(80m), 16)` — bin width ≈ 0.245 in log-space
- `DepthDFL` module decodes: `softmax(logits) → weighted sum by bin_values → log-depth`
- Loss: `DFLoss` (from YOLO's existing implementation) replaces smooth_l1 for depth
- Training exposes both raw bin logits (for DFLoss) and decoded log-depth (for downstream)
- Inference outputs only decoded log-depth — downstream pipeline unchanged

**Results**: s=53.0% (+6.6pp over v30), n=41.3% — **biggest single-change improvement since the cost volume (v17)**

**Why it works**:
1. DFLoss (cross-entropy with soft binning) provides more stable gradients than smooth_l1 on unbounded log-depth values
2. The softmax output naturally represents prediction confidence — ambiguous depths spread probability across bins rather than committing to a wrong scalar
3. 16 bins over log(2)–log(80) match the depth distribution of KITTI objects well
4. Only the final conv per scale changes (1→16 channels) — ~0% parameter overhead since most depth branch params are in the 3-layer hidden layers

**Training dynamics**: s peaked at epoch 700, n peaked at epoch 360 (both with 1000ep budget, patience=200).

---

### v32 — Depth Residual Offset (Failed)

**Goal**: Add sub-bin precision by learning a continuous residual offset on top of the coarse bin-decoded depth.

**Motivation**: With 16 bins over log(2)–log(80), bin width is ~0.245 in log-space. At 20m depth, this corresponds to ~±2.5m quantization error. A learned residual could recover this lost precision without disrupting the DFL training signal.

**Architecture changes**:
- Depth branch outputs 17 channels (16 bins + 1 residual)
- `DepthDFL` splits input: bins[:16] for softmax decode, residual[16:] added to coarse depth
- Loss: `DFLoss(bins) + smooth_l1(residual, gt_depth - coarse_depth.detach())`
- Single "depth" loss slot maintained — 6-item loss vector unchanged

**Results**: s=48.8% (**-4.2pp regression**), best at epoch 290, early stopped at epoch 490

**Why it failed**: The residual channel interferes with the bin classification training signal, even though the residual target is detached from the bin gradients. Possible mechanisms:
1. The shared hidden layers (3×Conv) must now serve two objectives (classification bins + regression residual) through the same features, creating an optimization conflict
2. The smooth_l1 residual loss provides gradients that compete with DFLoss gradients in the shared layers
3. The residual initializes randomly near zero, which is correct, but the initial noise may destabilize early bin learning when training from a pretrained checkpoint

**Decision**: Reverted. The 16-bin DFL approach (v31) remains optimal.

---

### v33a — Depth Loss Weight 3.0

**Goal**: Give depth more gradient by increasing its loss weight from 1.0 to 3.0.

**Motivation**: The box loss weight is 7.5, cls is 0.5. Depth at 1.0 may be under-represented in the total gradient, causing the shared backbone to prioritize 2D detection features over depth features. Tripling the depth weight should shift the balance.

**Config**: Same as v31 except `depth: 3.0` in YAML loss_weights. s-size, SGD 1000ep, GPUs 0-3.

**Results**: s=56.5% AP3D@0.5, 38.0% AP3D@0.7 — **+3.5pp over v31**, new best (non-R40)

**R40 results** (KITTI-standard, Moderate mean): AP3D@0.5=47.3%, AP3D@0.7=28.1%

Best at epoch 240, early stopped at epoch 440. Simple loss weight tuning outperformed all architecture changes (residual, 32 bins).

---

### v33b — 32 Depth Bins

**Goal**: Halve the bin width by doubling from 16 to 32 bins.

**Motivation**: v32's residual approach to sub-bin precision failed, but finer bins via pure DFL is the clean alternative. 32 bins over log(2)–log(80) gives bin width ≈ 0.122 in log-space (half of 16-bin). At 20m, quantization error drops from ~±2.5m to ~±1.2m.

**Config**: `DEPTH_BINS=32` in head_yolo11.py. s-size, SGD 1000ep from v31 best.pt, GPUs 4-7.

**Results**: s=56.2% AP3D@0.5, 35.9% AP3D@0.7 — +3.2pp over v31, but slightly worse than v33a

Best at epoch 220, early stopped at epoch 420. More bins ≈ same AP3D@0.5 but weaker AP3D@0.7. Not worth the complexity — 16 bins remain optimal.

---

### v34 — YOLO26 Backbone + R40 Evaluation

**Goal**: Evaluate YOLO26 backbone across all model sizes with KITTI-standard R40 evaluation.

**Changes**:
1. Switched from YOLO11 to YOLO26 backbone (same cost volume architecture)
2. Implemented KITTI R40 AP evaluation: 40-point max-precision-at-recall-threshold with difficulty splits (Easy/Moderate/Hard)
3. Trained all sizes (n/s/m/l) with SGD 1000ep, filter_occluded=True

**Results (R40 Moderate Mean AP3D)**:

| Size | Params | AP3D@0.5 | AP3D@0.7 |
|------|--------|----------|----------|
| n | 3.6M | 42.4% | 24.8% |
| **s** | **11.6M** | **49.8%** | **30.7%** |
| m | 26.7M | 49.0% | 31.2% |
| l | 31.1M | 47.5% | 30.2% |

**Conclusion**: s-size is the sweet spot. m/l overfit on KITTI's small training set (~3,700 images). The diminishing returns beyond s confirm that the bottleneck is data, not model capacity.

---

### v35 — Occlusion Filter Disabled

**Goal**: Train on ALL objects including heavily occluded (KITTI occlusion levels 0-2 + DontCare), letting the KITTI difficulty system handle evaluation filtering.

**Motivation**: With `filter_occluded=True`, ~25% of training labels were discarded (occluded objects). The KITTI R40 difficulty system already accounts for occlusion (Easy: occ==0, Moderate: occ<=1, Hard: occ<=2). Training on all objects should improve Hard difficulty without hurting Easy.

**Config**: YOLO26-s, SGD 1000ep, `filter_occluded=False`. Otherwise same as v34.

**Results (R40, per-class and difficulty)**:

| Class | Easy@0.5 | Mod@0.5 | Hard@0.5 | Easy@0.7 | Mod@0.7 | Hard@0.7 |
|-------|----------|---------|----------|----------|---------|----------|
| Car | 72.8% | 71.8% | **67.9%** | 36.8% | 41.2% | **39.6%** |
| Ped | 49.4% | 42.4% | **38.5%** | 31.2% | 27.0% | **24.4%** |
| Cyc | 27.1% | 30.1% | **28.4%** | 17.5% | 22.1% | **20.8%** |

Mean Moderate: AP3D@0.5=48.1%, AP3D@0.7=30.1%
Mean Hard: AP3D@0.5=44.9%, AP3D@0.7=28.3%

**vs v34 (filter ON)**: Moderate -1.7pp @0.5, but Hard **+3.5pp @0.5, +3.0pp @0.7**. Car Hard@0.5: 55.7%→67.9% (+12.2pp). Much better robustness to occluded/truncated objects.

Best at epoch 210, early stopped at epoch 410.

---

### v36 — Siamese Weight-Shared Backbone

**Goal**: Replace groups=2 stem with a siamese/weight-shared backbone for 100% pretrained weight compatibility and clean left-only 2D features.

**Architecture** (see §2.7): Standard 3ch YOLO26 backbone processes L/R images via batch trick [2B,3,H,W]. Features are split at the tap point (backbone layer 1, stride 4) for cost volume computation. Only left features continue through the remaining backbone and head.

**Config**: YOLO26 + siamese, SGD 1000ep, `filter_occluded=False`, 6 GPUs. Same recipe as v35.

**R40 Results (KITTI-standard, 3-class multi-class)**:

| Size | Params | Mean Mod @0.5 | Mean Mod @0.7 | Car Mod @0.5 | Car Mod @0.7 |
|------|--------|---------------|---------------|--------------|--------------|
| n | 3.6M | 48.1% | 29.9% | 70.7% | 42.4% |
| s | 11.6M | 48.3% | 29.4% | 71.7% | 42.3% |
| m | 26.7M | 49.0% | 29.1% | 71.1% | 39.1% |
| l | 31.2M | **50.9%** | **31.6%** | 71.3% | 41.8% |

**vs v35 (groups=2, R40 Moderate)**:

| Metric | v35-s | v36-s | v36-l | Δ (s) | Δ (l) |
|--------|-------|-------|-------|-------|-------|
| Mean @0.5 | 48.1% | 48.3% | **50.9%** | +0.2 | **+2.8** |
| Mean @0.7 | 30.1% | 29.4% | **31.6%** | -0.7 | **+1.5** |
| Car @0.5 | 71.8% | 71.7% | 71.3% | -0.1 | -0.5 |
| Car @0.7 | 41.2% | 42.3% | 41.8% | +1.1 | +0.6 |

**Key finding**: Siamese matches groups=2 at R40 level across all sizes. l-size (50.9%) is new best Mean Moderate AP3D@0.5, unlike groups=2 where l overfits. The siamese shared backbone may regularize better at larger model sizes.

**Car-only (nc=1) training FAILED**: s=2.87% AP3D@0.5, n=2.16% after 280 epochs. Depth DFLoss stuck at ~5.0 (3-class converges to 2.5 by ep10). Model predicts all depths at ~17-18m (depth branch collapsed). Root cause: TAL with nc=1 binary classifier provides insufficient gradient diversity for auxiliary branches. **Workaround**: use per-class Car results from multi-class model for benchmark tables.

**Speed benchmarks** (single Blackwell RTX PRO 6000, batch=1, fp16, 384×1248):

| Size | Params | GFLOPs | Inference | FPS |
|------|--------|--------|-----------|-----|
| n | 3.6M | 7.3G | 7.9ms | 127 |
| s | 11.6M | 18.5G | 8.2ms | 122 |
| m | 26.8M | 60.9G | 9.0ms | 111 |
| l | 31.2M | — | 13.9ms | 72 |

All sizes are well within real-time (>30 FPS). n-size at 127 FPS is attractive for deployment.

**Status**: Complete. All 4 sizes evaluated with R40. Car-only nc=1 approach abandoned (use multi-class per-class results).

---

## 6. Key Findings

### 6.1 What Worked

1. **StereoCostVolume (+18pp)**: The single biggest architectural improvement. Explicit disparity correlation at 24 offsets provides stereo matching cues that the backbone cannot learn implicitly. Cost: only +0.7% parameters.

2. **DFL-Style Depth Bins (+6.6pp)**: Replacing scalar depth regression with 16-bin classification using Distribution Focal Loss. Provides more stable gradients and natural confidence calibration. The second-largest single-change improvement.

3. **Long SGD Training (+6pp)**: Extending from 200 to 1000 SGD epochs yielded +6.2pp for s-size. The model was significantly underfitting at 200 epochs.

4. **Depth-Only Cost Volume Routing (+2-5pp over merged)**: Separating 2D detection from 3D depth estimation in the head avoids task conflict. P3 stays clean for 2D boxes; cost volume enriches only depth branches.

5. **SGD Finetuning (+6-8pp)**: Consistently improves over AdamW-only training. Two-stage recipe (200ep AdamW → 200ep SGD) is better than pure SGD.

6. **Loss Weight Rebalancing (+1-2pp)**: Increasing depth loss weights from 0.1 to 1.0 (and lr_distance from 0.2 to 2.0) ensures the network actually optimizes for depth, which was previously dwarfed by the box loss (7.5).

7. **Log-Space Depth Targets (+14pp)**: Converting lr_distance targets from raw disparity [0.01, 0.22] to log-space [-4.6, -1.5] dramatically improved training stability. AP3D@0.5 tripled from 7.4% to 21.1%.

8. **Augmentation (+1-2pp)**: Stereo-safe augmentations (horizontal flip with view swap, HSV) provide modest regularization benefit.

9. **Siamese Backbone (v36)**: Weight-shared 3ch backbone via batch trick matches groups=2 quality with 100% pretrained weight compatibility. n-size matches s-size on KITTI — 3.2x fewer params for equal accuracy.

10. **Disabling Occlusion Filter (v35, +3.5pp Hard)**: Training on all objects including heavily occluded improves Hard difficulty by +3.5pp @0.5 with only -1.7pp Moderate regression. Car Hard@0.5: +12.2pp.

### 6.2 What Didn't Work

1. **Laplacian NLL Loss (v8)**: Loss went negative from epoch 4 — the network learned to predict infinite uncertainty to zero out the loss.

2. **Inline StereoCorrelation (v9-v11)**: Inserting correlation into the main feature pathway is neutral. Unlike the cost volume (separate pathway to depth branches), inline correlation dilutes backbone features.

3. **Distance-Aware Loss Weighting (v9-v10)**: Weighting closer objects higher didn't help — the TAL assignment already biases toward well-matched (often closer) objects.

4. **Sigma/Uncertainty Supervision (v10)**: Training the network to predict its own uncertainty for decode-time fusion created optimization conflicts.

5. **RAFT-Stereo Depth Prior (v22)**: Competitive but adds offline RAFT-Stereo dependency (~3s/frame). Not worth the complexity vs the learned cost volume.

6. **Knowledge Distillation (v23-v25)**: All variants hurt performance by 3-7pp. Teacher and student learn fundamentally different representations.

7. **YOLO26 Backbone (v15-v16)**: Marginal improvement (~1pp) over YOLO11. Architecture is not the bottleneck.

8. **Multi-Scale Cost Volume (v27-v29)**: All variants worse than single-scale. The coarser resolution loses the sub-pixel precision that matters for depth. Architecture itself hurt even without disparity loss.

9. **Car-Only Single-Class Training (v36)**: nc=1 models completely fail — depth branch collapses (DFLoss stuck at 5.0, all predictions at ~17m). TAL with binary classifier provides insufficient gradient diversity for auxiliary 3D branches. Use per-class results from multi-class models instead.

9. **Depth Residual Offset (v32)**: Adding a regression channel alongside DFL bins hurt by -4.2pp. The shared hidden layers can't serve both classification and regression well. Sub-bin precision is not the current bottleneck.

10. **Fitness Function Bug (v30)**: Silently saving best.pt by 2D fitness instead of 3D fitness caused all prior experiments to report wrong best checkpoints. Always verify that your selection metric matches your evaluation metric.

11. **32 Depth Bins (v33b)**: 32 bins ≈ same AP3D@0.5 as 16 bins but slightly worse @0.7. Not worth the added complexity.

### 6.3 Lessons Learned

- **Isolate variables**: v9 combined 3 changes and was impossible to interpret. v11 (single change) clearly showed StereoCorrelation is neutral.
- **Depth is the #1 bottleneck**: Oracle analysis shows +17pp headroom from perfect depth (at current v31 level). Every successful experiment directly improved depth estimation.
- **2D-3D task conflict is real**: Routing cost volume features only to depth branches (v20) outperforms merging into P3 (v17) by 2-5pp.
- **Classification > regression for depth**: DFL bins (+6.6pp) succeeded where all regression enhancements (Laplacian, uncertainty, residual) failed. Discrete distributions are more trainable than unbounded scalars.
- **Don't add complexity to working losses**: Residual, uncertainty, distance-weighting — every attempt to "improve" the depth loss made it worse. The simplest loss (DFLoss for bins, smooth_l1 for other aux) is the best.
- **s-size is the sweet spot (groups=2), n-size for siamese**: With groups=2 backbone, s-size outperforms n by 3-6pp. But with siamese backbone (v36), n matches s — 3.2x fewer params for equal accuracy. KITTI's small dataset limits larger models.
- **Train longer**: v26/v30 showed +6pp from 200→1000 epochs. v31 peaked at ep700. Don't underestimate the value of patience.
- **Simple weight tuning > architecture changes**: v33a (depth loss 1.0→3.0) gained +3.5pp with zero code changes, beating depth residual (v32, -4.2pp) and 32 bins (v33b, +3.2pp).

---

## 7. Error Analysis

Oracle ablation analysis (substituting one predicted component with ground truth) on the v12 checkpoint:

| Oracle Component | AP3D@0.5 | AP3D@0.7 | Change vs Baseline |
|-----------------|----------|----------|--------------------|
| Baseline (no oracle) | 28.5% | 17.5% | — |
| **Oracle Depth** | **54.5%** | **49.6%** | **+26.0pp / +32.1pp** |
| Oracle Dimensions | 28.7% | 18.3% | +0.2pp / +0.8pp |
| Oracle Orientation | 24.2% | 14.0% | -4.3pp / -3.5pp |
| Oracle All | 49.1% | 48.9% | +20.6pp / +31.4pp |

### Per-Component Error Statistics

| Component | Mean Error | Median Error | Relative Error |
|-----------|-----------|-------------|----------------|
| Depth | 0.81m | 0.66m | 5.1% |
| Dimensions (L) | 0.20m | — | — |
| Dimensions (W) | 0.05m | — | — |
| Dimensions (H) | 0.05m | — | — |
| Orientation | 16.4° | 6.6° | — |

### Interpretation

- **Depth is overwhelmingly the bottleneck**: Perfect depth would nearly double AP3D@0.5. This motivated the cost volume experiments (v17+).
- **Dimensions are solved**: Oracle dims provide only +0.2pp. The normalized offset prediction with class-specific priors works well.
- **Orientation oracle hurts**: The -4.3pp drop suggests a matching artifact — oracle orientation changes the 3D IoU in unexpected ways due to the interaction between box center and heading angle.

---

## 8. Failed Approaches

### 8.1 Laplacian NLL (v8)

**Hypothesis**: Per-sample uncertainty would down-weight hard examples and improve depth estimation.

**Why it failed**: The loss function `L = |y - pred| * exp(-log_sigma) + log_sigma` has a pathological minimum at `log_sigma → ∞`, which drives the loss to `-∞`. Without careful regularization, the network exploits this by predicting maximum uncertainty for all samples.

### 8.2 StereoCorrelation Module (v9-v11)

**Hypothesis**: Explicit left-right correlation features would improve depth.

**Why it failed**: When inserted into the main backbone pathway, correlation features compete with appearance features needed for 2D detection and classification. The cost volume approach (v17) succeeds because it provides correlation as a **separate, additive signal** only to depth branches.

### 8.3 RAFT-Stereo Distillation (v23-v25)

**Hypothesis**: A student model could learn to predict RAFT-quality depth without the RAFT-Stereo network at inference time.

**Why it failed**: RAFT-Stereo produces dense, high-precision disparity maps using iterative refinement — a fundamentally different representation than the sparse, per-anchor predictions of the YOLO head. Forcing alignment between these representations through MSE or L1 distillation creates conflicting gradients with the primary 3D detection loss.

### 8.4 Combined Changes (v9, v10)

**Hypothesis**: Multiple improvements should compound.

**Why it failed**: Optimization dynamics are nonlinear. Each change shifts the loss landscape, and combinations can create interference patterns. v9 combined 3 changes and only improved n-size while catastrophically hurting s/m. The lesson: always test changes in isolation first.

### 8.5 Multi-Scale Cost Volume Pyramid (v27-v29)

**Hypothesis**: Cost volumes at multiple resolutions (stride 4, 8, 16) would capture disparities at different scales.

**Why it failed**: The coarser cost volumes (stride 8, 16) lose spatial precision without providing useful additional information. Depth estimation is fundamentally a high-resolution matching task — coarse features add noise, not signal. Even without the disparity loss (v29 no-disp), the multi-scale architecture alone hurt by -7.5pp vs single-scale.

### 8.6 Depth Residual Offset (v32)

**Hypothesis**: A learned residual on top of DFL bin-decoded depth would provide sub-bin precision.

**Why it failed**: The 17th channel (residual) shares the same 3-layer hidden network as the 16 bin logits. The smooth_l1 regression gradient from the residual competes with the DFL classification gradient from the bins in the shared layers, degrading both tasks. The residual target (gt - coarse_depth.detach()) was correctly computed without gradient flow, but the shared features still suffered from the multi-task conflict. Result: -4.2pp regression.

---

## 9. Current Best Configuration

### Model

```yaml
# yolo26s-stereo3ddet-siamese.yaml (v36 — recommended)
Architecture: YOLO26 backbone + siamese weight-shared + StereoCostVolume
Input: 6 channels (left RGB + right RGB), processed as [2B,3,H,W] batch trick
Cost volume: 24 disparity bins, 64 output channels, routed to depth branches only
Head: Stereo3DDetHeadYOLO11 with P3/P4/P5 + cost_vol input
Pretrained: 100% compatible with standard YOLO26 weights
```

Alternative (v35): `yolo26s-stereo3ddet-costvol.yaml` with groups=2 stem — slightly higher Moderate AP but no pretrained compatibility.

### Training Recipe

```
SGD lr=0.01, cosine decay to lr*0.01, 1000 epochs
Augmentation: fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3
Batch: 128 (n), 64 (s), 32 (m), 16 (l)
Image size: 384x1248
Patience: 200 (early stopping based on AP3D@0.5 fitness)
filter_occluded: False (train on all objects including heavily occluded)
```

### Depth Branch

```
Output: 16 channels (DFL bins in log-depth space)
Bin range: linspace(log(2m), log(80m), 16) — bin width ≈ 0.245 log-units
Decode: softmax(logits) → weighted sum → log-depth scalar
Loss: DFLoss (cross-entropy with soft binning)
Architecture: 3×Conv deep branch per scale, cost volume concat at P3
```

### Loss Weights

```yaml
lr_distance: 2.0   # log-space disparity
depth: 3.0          # DFL bin classification (log-space direct depth)
dimensions: 1.0     # normalized offset
orientation: 1.0    # sin/cos encoding
box: 7.5            # standard YOLO box loss
cls: 0.5            # standard YOLO cls loss
```

### Best Results

**R40 evaluation (KITTI-standard, v36 siamese)**:

| Size | Mean Mod @0.5 | Mean Mod @0.7 | Car Mod @0.5 | Car Mod @0.7 | FPS |
|------|---------------|---------------|--------------|--------------|-----|
| n | 48.1% | 29.9% | 70.7% | 42.4% | 127 |
| s | 48.3% | 29.4% | 71.7% | 42.3% | 122 |
| m | 49.0% | 29.1% | 71.1% | 39.1% | 111 |
| l | **50.9%** | **31.6%** | 71.3% | 41.8% | 72 |

### Remaining Headroom

Oracle analysis (from v12 baseline) showed +26pp from perfect depth. Remaining depth error is primarily in feature extraction and stereo matching quality, not the output representation.

**Potential future directions**:
- Wider cost volume disparity coverage (currently 24 offsets × 2px = 48px at stride 4 → max ~192px full-res)
- Higher resolution training (384×1248 → 480×1560)
- Attention-based cost volume (replace dot-product with cross-attention)
- Test pretrained weight loading with siamese backbone (main benefit not yet exploited)
- Curriculum learning or multi-task pretraining to improve auxiliary branch gradient quality
