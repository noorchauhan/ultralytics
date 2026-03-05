# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseBackend
from ultralytics.utils.checks import check_requirements


class TritonBackend(BaseBackend):
    """Triton Inference Server backend.

    Supports loading and inference with models hosted on NVIDIA Triton Inference Server.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the Triton remote model."""
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(weight)

        # Copy metadata from Triton model
        if hasattr(self.model, "metadata"):
            self.apply_metadata(self.model.metadata)

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run Triton inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor(s).
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im.cpu().numpy())
