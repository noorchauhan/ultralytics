# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.utils.checks import check_requirements
from ultralytics.nn.backends.base import BaseBackend


class TritonBackend(BaseBackend):
    """Triton Inference Server backend.

    Supports loading and inference with models hosted on NVIDIA Triton Inference Server.
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize Triton backend.

        Args:
            weights: Triton model URL (triton://model_name).
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.fp16 &= fp16  # Triton supports FP16

    def load_model(self) -> None:
        """Load the Triton remote model."""
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(str(self.weights))

        # Copy metadata from Triton model
        if hasattr(self.model, "metadata"):
            self.metadata = self.model.metadata

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run Triton inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()

        im_np = im.cpu().numpy()
        y = self.model(im_np)

        if isinstance(y, (list, tuple)):
            return [self.from_numpy(x) for x in y]
        return self.from_numpy(y)
