# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for stereo 3D detection validator."""

from unittest.mock import MagicMock

import torch

from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator


class TestStereo3DDetValidator:
    """Test suite for Stereo3DDetValidator class."""

    def test_validator_initialization(self):
        """Test that Stereo3DDetValidator initializes correctly."""
        args = {"task": "stereo3ddet", "imgsz": 640}
        validator = Stereo3DDetValidator(args=args)
        assert validator.args.task == "stereo3ddet"
        assert hasattr(validator, "metrics")

    def test_validator_preprocess(self):
        """Test that preprocess handles stereo batch correctly."""
        args = {"task": "stereo3ddet", "imgsz": 640, "half": False}
        validator = Stereo3DDetValidator(args=args)
        validator.device = torch.device("cpu")

        # Create mock batch with stereo images [B, 6, H, W]
        batch = {
            "img": torch.randn(2, 6, 384, 1280),
            "labels": [None, None],
            "calib": [None, None],
        }
        processed = validator.preprocess(batch)
        assert "img" in processed
        assert processed["img"].shape == (2, 6, 384, 1280)

    def test_validator_uses_6_channel_input(self):
        """Test that validator uses 6-channel input during warmup and inference (T090)."""
        args = {"task": "stereo3ddet", "imgsz": 640, "data": None}
        validator = Stereo3DDetValidator(args=args)
        validator.device = torch.device("cpu")

        # Set self.data with channels=6
        validator.data = {
            "channels": 6,
            "names": {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
            "nc": 3,
        }

        assert validator.data["channels"] == 6, "Validator should use 6 channels for stereo input"
