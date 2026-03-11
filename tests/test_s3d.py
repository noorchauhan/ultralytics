# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for Stereo 3D Detection (s3d) task.

Covers model creation, training, validation, prediction, and export,
plus critical unit tests for 3D IoU and KITTI R40 metrics.
"""

import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.utils.metrics import compute_3d_iou

MODEL = "yolo26n-s3d.yaml"
DATA = "kitti-stereo8.yaml"


# --- Integration tests (train/val/predict/export) ---


def test_model_creation():
    """Test s3d model creation from YAML for all scales."""
    for scale in ("n", "s", "m", "l"):
        model = YOLO(f"yolo26{scale}-s3d.yaml")
        assert model.task == "s3d"


def test_train():
    """Test s3d training for 2 epochs on mini dataset."""
    model = YOLO(MODEL)
    model.train(data=DATA, epochs=2, imgsz=[384, 1248], batch=2, val=False)


def test_val():
    """Test s3d validation on mini dataset."""
    model = YOLO(MODEL)
    model.val(data=DATA, imgsz=[384, 1248], batch=2)


def test_predict(tmp_path):
    """Test s3d prediction on synthetic stereo pair."""
    import cv2

    # Create synthetic stereo pair
    left_img = tmp_path / "left.png"
    right_img = tmp_path / "right.png"
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(left_img), img)
    cv2.imwrite(str(right_img), img)

    model = YOLO(MODEL)
    results = model.predict(source=[(str(left_img), str(right_img))], imgsz=[384, 1248])
    assert len(results) >= 0  # Should complete without error


def test_export_onnx():
    """Test ONNX export for s3d model."""
    model = YOLO(MODEL)
    path = model.export(format="onnx", imgsz=[384, 1248])
    assert path.endswith(".onnx")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="TensorRT requires CUDA")
def test_export_engine():
    """Test TensorRT engine export for s3d model."""
    model = YOLO(MODEL)
    path = model.export(format="engine", imgsz=[384, 1248])
    assert path.endswith(".engine")


# --- Unit tests: 3D IoU ---


def test_3d_iou_identical():
    """Test that identical 3D boxes have IoU of 1.0."""
    box = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                class_label="Car", class_id=0, confidence=0.95)
    assert abs(compute_3d_iou(box, box) - 1.0) < 1e-6


def test_3d_iou_no_overlap():
    """Test that non-overlapping boxes have IoU of 0.0."""
    box1 = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    box2 = Box3D(center_3d=(100.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    assert compute_3d_iou(box1, box2) == 0.0


def test_3d_iou_partial_overlap():
    """Test IoU for partially overlapping 3D boxes."""
    box1 = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    box2 = Box3D(center_3d=(11.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    iou = compute_3d_iou(box1, box2)
    assert 0.0 < iou < 1.0


# --- Unit tests: KITTI R40 metrics ---


def test_classify_difficulty():
    """Test KITTI difficulty classification."""
    from ultralytics.models.yolo.s3d.metrics import (
        DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_MODERATE, classify_difficulty,
    )

    assert classify_difficulty(truncated=0.0, occluded=0, bbox_height_2d=50.0) == DIFFICULTY_EASY
    assert classify_difficulty(truncated=0.2, occluded=1, bbox_height_2d=30.0) == DIFFICULTY_MODERATE
    assert classify_difficulty(truncated=0.4, occluded=2, bbox_height_2d=28.0) == DIFFICULTY_HARD
    assert classify_difficulty(truncated=0.0, occluded=3, bbox_height_2d=50.0) == -1  # Don't care
    assert classify_difficulty(truncated=0.0, occluded=0, bbox_height_2d=20.0) == -1  # Too small


def test_compute_ap_r40():
    """Test R40 AP computation with perfect and zero predictions."""
    from ultralytics.models.yolo.s3d.metrics import compute_ap_r40

    # Perfect predictions
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert abs(compute_ap_r40(recall, precision) - 1.0) < 1e-6

    # No true positives
    recall = np.array([0.0])
    precision = np.array([0.0])
    assert compute_ap_r40(recall, precision) == 0.0


def test_metrics_process():
    """Test Stereo3DDetMetrics end-to-end with a perfect match."""
    from ultralytics.models.yolo.s3d.metrics import DIFFICULTY_EASY, Stereo3DDetMetrics

    metrics = Stereo3DDetMetrics(names={0: "Car"})

    gt_box = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                   class_label="Car", class_id=0, confidence=1.0, truncated=0.0, occluded=0)
    pred_box = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                     class_label="Car", class_id=0, confidence=0.95)

    iou_matrix = np.array([[compute_3d_iou(pred_box, gt_box)]])
    metrics.update_stats({
        "pred_boxes": [pred_box],
        "gt_boxes": [gt_box],
        "iou_matrix": iou_matrix,
        "gt_difficulties": np.array([DIFFICULTY_EASY]),
        "pred_heights_2d": np.array([50.0]),
    })
    metrics.process()

    assert metrics.ap3d[0.5][DIFFICULTY_EASY][0] > 0.0
    assert metrics.maps3d_50 > 0.0
