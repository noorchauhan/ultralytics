# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip
from ultralytics.nn.backends.base import BaseBackend


class RKNNBackend(BaseBackend):
    """RKNN inference backend.

    Supports loading and inference with RKNN models on Rockchip devices.
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize RKNN backend.

        Args:
            weights: Path to the RKNN model file or directory.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.rknn = True
        self.nhwc = True
        self.rknn_model = None

    def load_model(self) -> None:
        """Load the RKNN model."""
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")
            
        LOGGER.info(f"Loading {self.weights} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(self.weights)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.rknn_model = RKNNLite()
        ret = self.rknn_model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
            
        ret = self.rknn_model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML
            self.metadata = YAML.load(metadata_file)

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run RKNN inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        im_np = (im.cpu().numpy() * 255).astype("uint8")
        im_np = im_np if isinstance(im_np, (list, tuple)) else [im_np]
        y = self.rknn_model.inference(inputs=im_np)

        if isinstance(y, (list, tuple)):
            return [self.from_numpy(x) for x in y]
        return self.from_numpy(y)
