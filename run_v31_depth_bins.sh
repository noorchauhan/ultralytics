#!/bin/bash
# v31: Depth bin classification (DFL-style) replacing scalar depth regression
#
# Key change:
#   - Depth branch outputs 16 bins in log-depth space [log(2m), log(80m)]
#   - DFLoss (cross-entropy with soft binning) replaces smooth_l1
#   - DFL decode (softmax → weighted sum) converts bins to scalar log-depth
#   - Only 6 params change per scale (final conv 1→16), rest of depth branch preserved
#
# Baseline (v30 = v26 + fitness fix):
#   s: 46.4% AP3D@0.5 (SGD 1000ep from v21 checkpoint)
#
# Starting from v30 best.pt with partial weight transfer (depth final conv re-initialized).
# Using same SGD 1000ep setup as v26/v30 for direct comparison.

DATA="/home/rick/datasets/kitti_raw/dataset.yaml"
EPOCHS=1000
IMGSZ="384,1248"
DEVICES="0,1,2,3,4,5,6,7"
PATIENCE=200
VAL_PERIOD=10

AUG="fliplr=0.5 hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 scale=0.0 crop_fraction=0.0"

yolo settings wandb=true

# ── s-size: SGD from v30 checkpoint (pretrained= for partial weight transfer) ──
# Depth branch final conv re-initialized (1→16 bins), all other weights transferred.
CKPT="runs/stereo3ddet/s_costvol_v30_fitness_fix/weights/best.pt"

echo "============================================"
echo "v31: s-size SGD depth bins from v30  $(date)"
echo "============================================"

yolo train \
    task=stereo3ddet \
    model=yolo11s-stereo3ddet-costvol.yaml \
    pretrained=${CKPT} \
    data=${DATA} \
    epochs=${EPOCHS} batch=128 imgsz=${IMGSZ} device=${DEVICES} \
    patience=${PATIENCE} val_period=${VAL_PERIOD} \
    optimizer=SGD lr0=0.01 lrf=0.01 \
    cos_lr=True \
    name=s_costvol_v31_depth_bins \
    ${AUG}

echo "Finished s_costvol_v31_depth_bins  $(date)"
echo ""

# ── n-size: SGD from scratch ──
echo "============================================"
echo "v31: n-size SGD depth bins from scratch  $(date)"
echo "============================================"

yolo train \
    task=stereo3ddet \
    model=yolo11n-stereo3ddet-costvol.yaml \
    data=${DATA} \
    epochs=${EPOCHS} batch=128 imgsz=${IMGSZ} device=${DEVICES} \
    patience=${PATIENCE} val_period=${VAL_PERIOD} \
    optimizer=SGD lr0=0.01 lrf=0.01 \
    cos_lr=True \
    name=n_costvol_v31_depth_bins \
    ${AUG}

echo "Finished n_costvol_v31_depth_bins  $(date)"
echo ""

echo "All v31 runs complete.  $(date)"
echo ""
echo "=== v31 RESULTS ==="
echo "  s: runs/stereo3ddet/s_costvol_v31_depth_bins/"
echo "  n: runs/stereo3ddet/n_costvol_v31_depth_bins/"
echo ""
echo "Baseline (v30): s=46.4% AP3D@0.5"
