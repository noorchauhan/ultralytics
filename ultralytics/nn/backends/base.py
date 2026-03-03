# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseBackend(ABC):
    """Base class for all inference backends.

    This abstract class defines the interface that all inference backends must implement.
    It provides common functionality for model loading, inference, and device management.

    Attributes:
        model: The underlying inference model.
        device (torch.device): The device to run inference on.
        fp16 (bool): Whether to use FP16 precision.
        nhwc (bool): Whether the model expects NHWC input format.
        stride (int): Model stride, typically 32 for YOLO models.
        names (dict): Dictionary mapping class indices to class names.
        task (str): The task type (detect, segment, classify, pose, obb).
        batch (int): Batch size.
        imgsz (tuple): Input image size.
        ch (int): Number of input channels.
        end2end (bool): Whether the model has end-to-end NMS.
        dynamic (bool): Whether the model supports dynamic input shapes.
    """

    def __init__(self, weights: str | Path, device: torch.device | str, fp16: bool = False, **kwargs: Any):
        """Initialize the base backend.

        Args:
            weights: Path to the model weights file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional backend-specific arguments.
        """
        self.weights = Path(weights) if isinstance(weights, str) else weights
        self.device = torch.device(device) if isinstance(device, str) else device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.imgsz = (640, 640)
        self.ch = 3
        self.end2end = False
        self.dynamic = False
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from weights."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on the input image.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments for inference.

        Returns:
            Model output tensor(s).
        """
        raise NotImplementedError

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert NumPy array to torch tensor on the model device.

        Args:
            x: NumPy array or tensor.

        Returns:
            Torch tensor on self.device.
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def check_class_names(self, names: list | dict) -> dict[int, str]:
        """Check and convert class names to dict format.

        Args:
            names: Class names as list or dict.

        Returns:
            Dictionary mapping class indices to class names.
        """
        if isinstance(names, list):
            names = dict(enumerate(names))
        if isinstance(names, dict):
            names = {int(k): str(v) for k, v in names.items()}
        return names

    def default_class_names(self, data: str | Path | None = None) -> dict[int, str]:
        """Load class names from YAML or return default names.

        Args:
            data: Path to YAML file with class names.

        Returns:
            Dictionary mapping class indices to class names.
        """
        if data:
            from ultralytics.utils import YAML, check_yaml

            try:
                return YAML.load(check_yaml(data))["names"]
            except Exception:
                pass
        return {i: f"class{i}" for i in range(999)}
