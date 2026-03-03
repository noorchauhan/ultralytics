# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.nn.backends.base import BaseBackend


class MNNBackend(BaseBackend):
    """MNN inference backend.

    Supports loading and inference with MNN models (.mnn files).
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize MNN backend.

        Args:
            weights: Path to the .mnn model file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.mnn = True
        self.rt = None
        self.net = None

    def load_model(self) -> None:
        """Load the MNN model."""
        LOGGER.info(f"Loading {self.weights} for MNN inference...")
        check_requirements("MNN")
        import MNN

        config = {
            "precision": "low",
            "backend": "CPU",
            "numThread": (os.cpu_count() + 1) // 2
        }
        self.rt = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(str(self.weights), [], [], runtime_manager=self.rt, rearrange=True)
        
        # Load metadata from bizCode
        info = self.net.get_info()
        if "bizCode" in info:
            try:
                self.metadata = json.loads(info["bizCode"])
            except json.JSONDecodeError:
                pass

    def torch_to_mnn(self, x: torch.Tensor):
        """Convert PyTorch tensor to MNN tensor.
        
        Args:
            x: PyTorch tensor.
            
        Returns:
            MNN tensor.
        """
        import MNN
        return MNN.expr.const(x.data_ptr(), x.shape)

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run MNN inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        input_var = self.torch_to_mnn(im)
        output_var = self.net.onForward([input_var])
        y = [x.read() for x in output_var]

        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])
