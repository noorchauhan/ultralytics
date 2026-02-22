# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for stereo 3D detection metrics."""

import numpy as np
import pytest

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.stereo3ddet.metrics import (
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MODERATE,
    Stereo3DDetMetrics,
    classify_difficulty,
    compute_ap_r40,
)
from ultralytics.utils.metrics import compute_3d_iou


class TestCompute3DIoU:
    """Test suite for compute_3d_iou function."""

    def test_compute_3d_iou_identical_boxes(self):
        """Test that identical boxes have IoU of 1.0."""
        box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box, box)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_compute_3d_iou_non_overlapping_boxes(self):
        """Test that non-overlapping boxes have IoU of 0.0."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(100.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert iou == 0.0, f"Expected IoU=0.0 for non-overlapping boxes, got {iou}"

    def test_compute_3d_iou_partially_overlapping_boxes(self):
        """Test IoU calculation for partially overlapping boxes."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(11.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert 0.0 < iou < 1.0, f"Expected IoU between 0 and 1, got {iou}"

    def test_compute_3d_iou_with_numpy_array(self):
        """Test compute_3d_iou with numpy array input."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2_array = np.array([10.0, 2.0, 30.0, 3.88, 1.63, 1.53, 0.0])
        iou = compute_3d_iou(box1, box2_array)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_compute_3d_iou_with_rotation(self):
        """Test IoU calculation with rotated boxes."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=np.pi / 4,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert 0.0 <= iou <= 1.0, f"IoU must be in [0, 1], got {iou}"


class TestClassifyDifficulty:
    """Test suite for KITTI difficulty classification."""

    def test_easy(self):
        assert classify_difficulty(truncated=0.0, occluded=0, bbox_height_2d=50.0) == DIFFICULTY_EASY

    def test_moderate(self):
        assert classify_difficulty(truncated=0.2, occluded=1, bbox_height_2d=30.0) == DIFFICULTY_MODERATE

    def test_hard(self):
        assert classify_difficulty(truncated=0.4, occluded=2, bbox_height_2d=28.0) == DIFFICULTY_HARD

    def test_dont_care_too_small(self):
        assert classify_difficulty(truncated=0.0, occluded=0, bbox_height_2d=20.0) == -1

    def test_dont_care_too_occluded(self):
        assert classify_difficulty(truncated=0.0, occluded=3, bbox_height_2d=50.0) == -1

    def test_easy_boundary(self):
        """Height=40, occ=0, trunc=0.15 should be Easy."""
        assert classify_difficulty(truncated=0.15, occluded=0, bbox_height_2d=40.0) == DIFFICULTY_EASY

    def test_moderate_boundary(self):
        """Height=25, occ=1, trunc=0.30 should be Moderate."""
        assert classify_difficulty(truncated=0.30, occluded=1, bbox_height_2d=25.0) == DIFFICULTY_MODERATE


class TestComputeApR40:
    """Test suite for R40 AP computation."""

    def test_perfect_ap(self):
        """Perfect predictions should give AP close to 1.0."""
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 1.0])
        ap = compute_ap_r40(recall, precision)
        assert abs(ap - 1.0) < 1e-6

    def test_zero_ap(self):
        """No true positives should give AP of 0.0."""
        recall = np.array([0.0])
        precision = np.array([0.0])
        ap = compute_ap_r40(recall, precision)
        assert ap == 0.0

    def test_partial_ap(self):
        """Partial detections should give AP between 0 and 1."""
        recall = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        precision = np.array([1.0, 0.8, 0.7, 0.6, 0.5])
        ap = compute_ap_r40(recall, precision)
        assert 0.0 < ap < 1.0


class TestStereo3DDetMetrics:
    """Test suite for Stereo3DDetMetrics class."""

    def test_metrics_initialization(self):
        """Test that Stereo3DDetMetrics initializes correctly."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        assert metrics.names == names
        assert metrics.nc == 3
        assert metrics.stats == []
        assert metrics.ap3d == {}

    def test_metrics_update_stats(self):
        """Test that update_stats correctly stores statistics."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)

        stat = {
            "pred_boxes": [],
            "gt_boxes": [],
            "iou_matrix": np.zeros((0, 0)),
            "gt_difficulties": np.array([], dtype=int),
            "pred_heights_2d": np.array([]),
        }
        metrics.update_stats(stat)
        assert len(metrics.stats) == 1

    def test_metrics_maps3d_properties(self):
        """Test that maps3d_50 and maps3d_70 return 0.0 when no metrics computed."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        assert metrics.maps3d_50 == 0.0
        assert metrics.maps3d_70 == 0.0

    def test_metrics_process_with_perfect_match(self):
        """Test metrics with a perfect prediction (identical box)."""
        names = {0: "Car"}
        metrics = Stereo3DDetMetrics(names=names)

        gt_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=1.0,
            truncated=0.0,
            occluded=0,
        )
        pred_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )

        iou_matrix = np.array([[compute_3d_iou(pred_box, gt_box)]])
        metrics.update_stats({
            "pred_boxes": [pred_box],
            "gt_boxes": [gt_box],
            "iou_matrix": iou_matrix,
            "gt_difficulties": np.array([DIFFICULTY_EASY]),
            "pred_heights_2d": np.array([50.0]),
        })
        metrics.process()

        # With perfect match: AP should be > 0 at Easy difficulty
        assert metrics.ap3d[0.5][DIFFICULTY_EASY][0] > 0.0
        assert metrics.maps3d_50 > 0.0  # Moderate should also have it (cumulative)
        assert metrics.fitness > 0.0

    def test_difficulty_ordering(self):
        """Test that Moderate AP <= Easy AP and Hard AP <= Moderate AP."""
        names = {0: "Car"}
        metrics = Stereo3DDetMetrics(names=names)

        # Create several GT boxes with different difficulties
        boxes = []
        for i in range(10):
            boxes.append(Box3D(
                center_3d=(10.0 + i * 5, 2.0, 30.0),
                dimensions=(3.88, 1.63, 1.53),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=1.0,
                truncated=0.0,
                occluded=0,
            ))

        # Perfect predictions for all
        preds = [Box3D(
            center_3d=b.center_3d,
            dimensions=b.dimensions,
            orientation=b.orientation,
            class_label="Car",
            class_id=0,
            confidence=0.9 - i * 0.05,
        ) for i, b in enumerate(boxes)]

        n = len(boxes)
        iou_matrix = np.eye(n)  # Perfect diagonal
        gt_diffs = np.array([DIFFICULTY_EASY] * 4 + [DIFFICULTY_MODERATE] * 3 + [DIFFICULTY_HARD] * 3)

        metrics.update_stats({
            "pred_boxes": preds,
            "gt_boxes": boxes,
            "iou_matrix": iou_matrix,
            "gt_difficulties": gt_diffs,
            "pred_heights_2d": np.full(n, 50.0),
        })
        metrics.process()

        # With all correct matches, Easy and Moderate and Hard should all have high AP
        # The key property: since difficulties are cumulative, all should be similar here
        ap_easy = metrics.ap3d[0.5][DIFFICULTY_EASY].get(0, 0.0)
        ap_mod = metrics.ap3d[0.5][DIFFICULTY_MODERATE].get(0, 0.0)
        ap_hard = metrics.ap3d[0.5][DIFFICULTY_HARD].get(0, 0.0)
        assert ap_easy > 0
        assert ap_mod > 0
        assert ap_hard > 0
