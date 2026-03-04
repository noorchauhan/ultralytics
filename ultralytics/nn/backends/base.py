# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseBackend(ABC):
    """Base class for all inference backends.

    This abstract class defines the interface that all inference backends must implement. It provides common
    functionality for model loading, inference, and device management.

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
        metadata (dict | None): Model metadata dictionary or None if not available.
    """

    def __init__(self, weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False, **kwargs: Any):
        """Initialize the base backend.

        Args:
            weight: Path to the model weight or a torch.nn.Module instance.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional backend-specific arguments.
        """
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.ch = 3
        self.end2end = False
        self.dynamic = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)

    @abstractmethod
    def load_model(self, weight: str | torch.nn.Module) -> None:
        """Load the model from weight."""
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

    def apply_metadata(self, metadata: dict | None) -> None:
        """Process and apply metadata to backend attributes.

        Handles type conversions for common metadata fields and applies them
        to the backend instance.

        Args:
            metadata: Dictionary containing metadata key-value pairs.
        """
        if not metadata:
            return

        # Store raw metadata
        self.metadata = metadata

        # Process type conversions
        for k, v in metadata.items():
            if k in {"stride", "batch", "channels"}:
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                metadata[k] = ast.literal_eval(v)

        # Apply to backend attributes
        for k, v in metadata.items():
            setattr(self, k, v)
