#!/bin/bash
# v21: SGD finetune ALL cost volume models — merged vs depth-only comparison
#
# Fair comparison: 200ep SGD finetune on both architectures at all sizes.
# Tests whether v20 depth-only just needs more training to surpass v19 merged.
#
# Base weights (200ep AdamW):
#   Merged:     n_costvol_v173 (29.5%), s_costvol_v173 (32.2%), m_costvol_v19 (36.1%)
#   Depth-only: n_costvol_v20  (28.8%), s_costvol_v20  (32.7%), m_costvol_v20 (33.5%)
#
# v18 (merged n/s SGD) already ran but results were mixed. Re-running all with
# identical settings for a clean apples-to-apples comparison.

DATA="/home/rick/datasets/kitti_raw/dataset.yaml"
EPOCHS=200
IMGSZ="384,1248"
DEVICES="0,1,2,3,4,5,6,7"
PATIENCE=200
VAL_PERIOD=10

AUG="fliplr=0.5 hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 scale=0.0 crop_fraction=0.0"

# ── Merged (v17/v19) SGD finetune ──
declare -A MERGED_WEIGHTS
MERGED_WEIGHTS[n]="runs/stereo3ddet/n_costvol_v173/weights/best.pt"
MERGED_WEIGHTS[s]="runs/stereo3ddet/s_costvol_v173/weights/best.pt"
MERGED_WEIGHTS[m]="runs/stereo3ddet/m_costvol_v19/weights/best.pt"

# ── Depth-only (v20) SGD finetune ──
declare -A DEPTHONLY_WEIGHTS
DEPTHONLY_WEIGHTS[n]="runs/stereo3ddet/n_costvol_v20_depthonly/weights/best.pt"
DEPTHONLY_WEIGHTS[s]="runs/stereo3ddet/s_costvol_v20_depthonly/weights/best.pt"
DEPTHONLY_WEIGHTS[m]="runs/stereo3ddet/m_costvol_v20_depthonly/weights/best.pt"

for SIZE in n s m; do
    # Adjust batch size for model size
    case $SIZE in
        n) BATCH=128 ;;
        s) BATCH=128 ;;
        m) BATCH=32  ;;
    esac

    # ── Run 1: Merged SGD finetune ──
    WEIGHTS="${MERGED_WEIGHTS[$SIZE]}"
    NAME="${SIZE}_costvol_v21_merged_sgd"
    echo "============================================"
    echo "Training ${NAME} from ${WEIGHTS}  $(date)"
    echo "============================================"
    yolo settings wandb=true
    yolo train \
        task=stereo3ddet \
        model=${WEIGHTS} \
        data=${DATA} \
        epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} device=${DEVICES} \
        patience=${PATIENCE} val_period=${VAL_PERIOD} \
        optimizer=SGD lr0=0.01 lrf=0.01 \
        cos_lr=True \
        name=${NAME} \
        ${AUG}
    echo "Finished ${NAME}  $(date)"
    echo ""

    # ── Run 2: Depth-only SGD finetune ──
    WEIGHTS="${DEPTHONLY_WEIGHTS[$SIZE]}"
    NAME="${SIZE}_costvol_v21_depthonly_sgd"
    echo "============================================"
    echo "Training ${NAME} from ${WEIGHTS}  $(date)"
    echo "============================================"
    yolo settings wandb=true
    yolo train \
        task=stereo3ddet \
        model=${WEIGHTS} \
        data=${DATA} \
        epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} device=${DEVICES} \
        patience=${PATIENCE} val_period=${VAL_PERIOD} \
        optimizer=SGD lr0=0.01 lrf=0.01 \
        cos_lr=True \
        name=${NAME} \
        ${AUG}
    echo "Finished ${NAME}  $(date)"
    echo ""
done

echo "All v21 runs complete.  $(date)"
echo ""
echo "=== COMPARISON ==="
echo "Base (200ep AdamW):"
echo "  Merged:     n=29.5%  s=32.2%  m=36.1%  AP3D@0.5"
echo "  Depth-only: n=28.8%  s=32.7%  m=33.5%  AP3D@0.5"
echo ""
echo "v21 SGD finetune results → runs/stereo3ddet/{n,s,m}_costvol_v21_{merged,depthonly}_sgd/"
