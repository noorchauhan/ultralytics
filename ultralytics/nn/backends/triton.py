# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


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

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run Triton inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output as list of numpy arrays.
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im.cpu().numpy())
