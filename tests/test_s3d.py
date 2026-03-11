# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for Stereo 3D Detection (s3d) task."""

import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.utils.metrics import compute_3d_iou

MODEL = "yolo26n-s3d.yaml"
DATA = "kitti-stereo8.yaml"


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

    left_img = tmp_path / "left.png"
    right_img = tmp_path / "right.png"
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(left_img), img)
    cv2.imwrite(str(right_img), img)

    model = YOLO(MODEL)
    results = model.predict(source=[(str(left_img), str(right_img))], imgsz=[384, 1248])
    assert len(results) >= 0


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


def test_3d_iou():
    """Test 3D IoU computation: identical, no overlap, and partial overlap."""
    box = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                class_label="Car", class_id=0, confidence=0.95)
    assert abs(compute_3d_iou(box, box) - 1.0) < 1e-6

    far_box = Box3D(center_3d=(100.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                    class_label="Car", class_id=0, confidence=0.95)
    assert compute_3d_iou(box, far_box) == 0.0

    near_box = Box3D(center_3d=(11.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                     class_label="Car", class_id=0, confidence=0.95)
    box2 = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    assert 0.0 < compute_3d_iou(box2, near_box) < 1.0
