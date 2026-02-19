#!/bin/bash
# v20: Cost volume feeds ONLY depth branches (not merged into P3)
#
# Fixes 2D-3D task conflict: v19 m-size had AP3D=36.1% but mAP50(2D)=74.9%
# vs s_yolo26_v16 which had AP3D=28.6% but mAP50(2D)=76.7%.
# Cost volume diluted P3 features, hurting 2D detection.
#
# New architecture: cost volume bypasses P3, concatenated only with P3
# for depth branches (lr_distance, depth). 2D detection sees clean P3/P4/P5.
#
# BASELINES (corrected, full val):
#   m_costvol_v19 (merged):     AP3D=36.1%, mAP50(2D)=74.9%
#   s_costvol_v17 (merged):     AP3D=32.2%, mAP50(2D)=61.5%
#   s_yolo26_v16  (no costvol): AP3D=28.6%, mAP50(2D)=76.7%

DATA="/home/rick/datasets/kitti_raw/dataset.yaml"
EPOCHS=200
IMGSZ="384,1248"
DEVICES="0,1,2,3,4,5,6,7"
PATIENCE=200
VAL_PERIOD=10

AUG="fliplr=0.5 hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 scale=0.0 crop_fraction=0.0"

for SIZE in n s m; do
    MODEL="yolo11${SIZE}-stereo3ddet-costvol.yaml"
    NAME="${SIZE}_costvol_v20_depthonly"

    # Adjust batch size for model size
    case $SIZE in
        n) BATCH=128 ;;
        s) BATCH=128 ;;
        m) BATCH=32  ;;
    esac

    echo "============================================"
    echo "Training ${NAME}  $(date)"
    echo "============================================"
    yolo settings wandb=true
    yolo train \
        task=stereo3ddet \
        model=${MODEL} \
        data=${DATA} \
        epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} device=${DEVICES} \
        patience=${PATIENCE} val_period=${VAL_PERIOD} \
        optimizer=AdamW lr0=0.001 lrf=0.01 \
        cos_lr=True \
        name=${NAME} \
        ${AUG}

    echo "Finished ${NAME}  $(date)"
    echo ""
done

echo "All v20 runs complete.  $(date)"
echo ""
echo "=== COMPARISON ==="
echo "v19 merged costvol:     m=36.1% AP3D, 74.9% mAP50(2D)"
echo "v20 depth-only costvol: runs/stereo3ddet/{n,s,m}_costvol_v20_depthonly/"
