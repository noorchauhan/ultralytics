# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.nn.backends.base import BaseBackend


class CoreMLBackend(BaseBackend):
    """CoreML inference backend.

    Supports loading and inference with CoreML models (.mlpackage files).
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize CoreML backend.

        Args:
            weights: Path to the .mlpackage model file.
            device: Device to run inference on (always CPU for CoreML).
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.coreml = True
        self.nhwc = True
        self.device = torch.device("cpu")  # CoreML uses CPU

    def load_model(self) -> None:
        """Load the CoreML model."""
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        LOGGER.info(f"Loading {self.weights} for CoreML inference...")
        self.model = ct.models.MLModel(self.weights)
        self.dynamic = self.model.get_spec().description.input[0].type.HasField("multiArrayType")
        
        # Load metadata
        metadata = dict(self.model.user_defined_metadata)
        if metadata:
            self.metadata = metadata

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run CoreML inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments (includes h, w for coordinate scaling).

        Returns:
            Model output tensor(s).
        """
        im_np = im.cpu().numpy()
        h, w = kwargs.get("h", im.shape[2]), kwargs.get("w", im.shape[3])
        
        if self.dynamic:
            im_np = im_np.transpose(0, 3, 1, 2)
            y = self.model.predict({"image": im_np})
        else:
            im_pil = Image.fromarray((im_np[0] * 255).astype("uint8"))
            y = self.model.predict({"image": im_pil})

        if "confidence" in y:  # NMS included
            from ultralytics.utils.ops import xywh2xyxy
            
            box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
            cls = y["confidence"].argmax(1, keepdims=True)
            y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
            return self.from_numpy(y)
        else:
            y = list(y.values())
            
        if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
            y = list(reversed(y))
            
        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])
