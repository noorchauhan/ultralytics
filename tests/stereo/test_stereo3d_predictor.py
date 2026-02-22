# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for Stereo3DDetPredictor and related functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.models.yolo.stereo3ddet.predict import Stereo3DDetPredictor, load_stereo_pair


@pytest.fixture
def sample_left_image(tmp_path):
    """Create a sample left image file."""
    import cv2

    img_path = tmp_path / "left.jpg"
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_right_image(tmp_path):
    """Create a sample right image file."""
    import cv2

    img_path = tmp_path / "right.jpg"
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_calibration():
    """Create a sample KITTI-style calibration."""
    return CalibrationParameters(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        baseline=0.54,
        image_width=1242,
        image_height=375,
    )


class TestLoadStereoPair:
    """Test load_stereo_pair function."""

    def test_load_stereo_pair_valid(self, sample_left_image, sample_right_image):
        """Test loading valid stereo pair."""
        left_img, right_img = load_stereo_pair(sample_left_image, sample_right_image)

        assert isinstance(left_img, np.ndarray)
        assert isinstance(right_img, np.ndarray)
        assert left_img.shape == (375, 1242, 3)
        assert right_img.shape == (375, 1242, 3)
        assert left_img.shape == right_img.shape

    def test_load_stereo_pair_missing_left(self, sample_right_image, tmp_path):
        """Test loading with missing left image."""
        missing_left = tmp_path / "missing_left.jpg"
        with pytest.raises(FileNotFoundError, match="Left image not found"):
            load_stereo_pair(missing_left, sample_right_image)

    def test_load_stereo_pair_missing_right(self, sample_left_image, tmp_path):
        """Test loading with missing right image."""
        missing_right = tmp_path / "missing_right.jpg"
        with pytest.raises(FileNotFoundError, match="Right image not found"):
            load_stereo_pair(sample_left_image, missing_right)

    def test_load_stereo_pair_size_mismatch(self, tmp_path):
        """Test loading with size mismatch."""
        import cv2

        left_path = tmp_path / "left.jpg"
        right_path = tmp_path / "right.jpg"

        left_img = np.zeros((375, 1242, 3), dtype=np.uint8)
        right_img = np.zeros((376, 1243, 3), dtype=np.uint8)  # Different size

        cv2.imwrite(str(left_path), left_img)
        cv2.imwrite(str(right_path), right_img)

        with pytest.raises(ValueError, match="Image size mismatch"):
            load_stereo_pair(left_path, right_path)


class TestStereo3DDetPredictor:
    """Test Stereo3DDetPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        # Use minimal overrides to avoid cfg parsing issues
        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "imgsz": 640}
        predictor = Stereo3DDetPredictor(overrides=overrides)

        assert predictor.args.task == "stereo3ddet"
        assert predictor.model is None  # Not set up yet
        assert hasattr(predictor, "calib_params")

    def test_preprocess_stereo_pair(self, sample_left_image, sample_right_image):
        """Test preprocessing stereo pair to 6-channel tensor."""
        # Load actual images
        left_img, right_img = load_stereo_pair(sample_left_image, sample_right_image)
        
        # Stack to create 6-channel stereo image
        stereo_img = np.concatenate([left_img, right_img], axis=2)  # [H, W, 6]

        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "imgsz": 384}
        predictor = Stereo3DDetPredictor(overrides=overrides)
        predictor.device = torch.device("cpu")
        predictor.imgsz = 384

        # Mock model
        predictor.model = MagicMock()
        predictor.model.fp16 = False
        predictor.model.stride = 32

        # Preprocess stereo pair (list of 6-channel images)
        stereo_imgs = [stereo_img]
        processed = predictor.preprocess(stereo_imgs)

        assert isinstance(processed, torch.Tensor)
        # Should be 6 channels (3 left + 3 right)
        assert processed.shape[1] == 6
        assert processed.shape[0] == 1  # Batch size 1

    @pytest.mark.skip(reason="Requires full model setup")
    def test_full_prediction_workflow(self):
        """Test full prediction workflow with single stereo pair."""
        # This test requires a trained model and actual stereo images
        pass

    @pytest.mark.skip(reason="Requires full model setup")
    def test_batch_prediction(self):
        """Test batch prediction with multiple stereo pairs."""
        # This test requires a trained model and actual stereo images
        pass

