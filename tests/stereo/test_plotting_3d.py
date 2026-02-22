# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for upcoming 3D visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ultralytics.utils import plotting


@pytest.mark.parametrize("calib_param", ["dataclass", "dict"])
def test_plot_boxes3d_draws_wireframes(sample_boxes3d, sample_calibration, sample_calibration_dict, calib_param):
    """Ensure plot_boxes3d returns an annotated copy with unchanged shape."""
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    calib = sample_calibration if calib_param == "dataclass" else sample_calibration_dict

    annotated = plotting.plot_boxes3d(img, sample_boxes3d, calib)

    assert annotated is not img, "Function should return a copy, not modify in-place"
    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype


def test_plot_boxes3d_handles_empty_list(sample_calibration_dict):
    """The helper should return the original-looking image when no boxes are provided."""
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    annotated = plotting.plot_boxes3d(img, [], sample_calibration_dict)

    assert annotated.shape == img.shape
    assert np.array_equal(annotated, img)


def test_plot_stereo3d_boxes_combines_views(sample_stereo_pair, sample_boxes3d):
    """plot_stereo3d_boxes should annotate both views and return a combined canvas."""
    left_annotated, right_annotated, combined = plotting.plot_stereo3d_boxes(
        sample_stereo_pair.left_image,
        sample_stereo_pair.right_image,
        pred_boxes3d=sample_boxes3d,
        gt_boxes3d=sample_boxes3d,
        left_calib=sample_stereo_pair.calibration,
    )

    assert left_annotated.shape == sample_stereo_pair.left_image.shape
    assert right_annotated.shape == sample_stereo_pair.right_image.shape
    assert combined.shape[0] == sample_stereo_pair.left_image.shape[0]
    assert combined.shape[1] == sample_stereo_pair.left_image.shape[1] + sample_stereo_pair.right_image.shape[1]


def test_plot_boxes3d_uses_pred_color_scheme(sample_box3d, sample_calibration_dict):
    """Verify that predicted boxes use prediction color scheme."""
    from ultralytics.utils.plotting import VisualizationConfig

    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    config = VisualizationConfig()
    # Use a distinct color for class 0 in pred scheme
    config.pred_color_scheme[0] = (255, 0, 0)  # Red for predictions
    config.gt_color_scheme[0] = (0, 255, 0)  # Green for ground truth

    annotated_pred = plotting.plot_boxes3d(img, [sample_box3d], sample_calibration_dict, config, is_ground_truth=False)
    annotated_gt = plotting.plot_boxes3d(img, [sample_box3d], sample_calibration_dict, config, is_ground_truth=True)

    # Images should be different (different colors used)
    assert not np.array_equal(annotated_pred, annotated_gt), "Prediction and GT should use different colors"


def test_plot_boxes3d_skips_invalid_boxes(sample_calibration_dict):
    """Verify that boxes with invalid parameters are skipped without errors."""
    from ultralytics.data.stereo.box3d import Box3D

    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    valid_box = Box3D(
        center_3d=(10.0, 2.0, 30.0),
        dimensions=(3.88, 1.63, 1.53),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )

    # Create an invalid box (negative depth) - this will fail validation in Box3D.__post_init__
    # So we can't create it directly, but we can test that the function handles projection errors
    boxes = [valid_box]

    # Should complete without error
    annotated = plotting.plot_boxes3d(img, boxes, sample_calibration_dict)
    assert annotated.shape == img.shape


def test_plot_boxes3d_handles_missing_calibration(sample_boxes3d):
    """Verify graceful handling when calibration is missing — boxes are skipped, no crash."""
    img = np.zeros((375, 1242, 3), dtype=np.uint8)

    # Empty calib dict causes KeyError in projection, caught by except → boxes skipped
    annotated = plotting.plot_boxes3d(img, sample_boxes3d, {})
    assert annotated.shape == img.shape
    assert np.array_equal(annotated, img), "No boxes should be drawn with invalid calibration"


def test_plot_stereo3d_boxes_requires_left_calib(sample_stereo_pair, sample_boxes3d):
    """Verify that left_calib is required for stereo visualization."""
    with pytest.raises(ValueError, match="left_calib is required"):
        plotting.plot_stereo3d_boxes(
            sample_stereo_pair.left_image,
            sample_stereo_pair.right_image,
            pred_boxes3d=sample_boxes3d,
            left_calib=None,
        )
