# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class CoreMLBackend(BaseBackend):
    """CoreML inference backend.

    Supports loading and inference with CoreML models (.mlpackage files).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the CoreML model."""
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        LOGGER.info(f"Loading {weight} for CoreML inference...")
        self.model = ct.models.MLModel(weight)
        self.dynamic = self.model.get_spec().description.input[0].type.HasField("multiArrayType")

        # Load metadata
        self.apply_metadata(dict(self.model.user_defined_metadata))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run CoreML inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output as numpy array(s).
        """
        im = im.cpu().numpy()
        h, w = im.shape[1:3]

        im = im.transpose(0, 3, 1, 2) if self.dynamic else Image.fromarray((im[0] * 255).astype("uint8"))
        y = self.model.predict({"image": im})
        if "confidence" in y:  # NMS included
            from ultralytics.utils.ops import xywh2xyxy

            box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
            cls = y["confidence"].argmax(1, keepdims=True)
            y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
        else:
            y = list(y.values())
        if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
            y = list(reversed(y))
        return y
