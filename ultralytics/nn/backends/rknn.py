# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip


class RKNNBackend(BaseBackend):
    """RKNN inference backend.

    Supports loading and inference with RKNN models on Rockchip devices.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the RKNN model."""
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
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

            self.apply_metadata(YAML.load(metadata_file))

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
