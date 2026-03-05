# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path

import MNN
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class MNNBackend(BaseBackend):
    """MNN inference backend.

    Supports loading and inference with MNN models (.mnn files).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the MNN model."""
        LOGGER.info(f"Loading {weight} for MNN inference...")
        check_requirements("MNN")
        # import MNN

        config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
        rt = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(weight, [], [], runtime_manager=rt, rearrange=True)

        # Load metadata from bizCode
        info = self.net.get_info()
        if "bizCode" in info:
            try:
                self.apply_metadata(json.loads(info["bizCode"]))
            except json.JSONDecodeError:
                pass

    def torch_to_mnn(self, x: torch.Tensor):
        """Convert PyTorch tensor to MNN tensor.

        Args:
            x: PyTorch tensor.

        Returns:
            MNN tensor.
        """
        return MNN.expr.const(x.data_ptr(), x.shape)

    def forward(self, im: torch.Tensor) -> list:
        """Run MNN inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output as list of MNN tensors.
        """
        input_var = self.torch_to_mnn(im)
        output_var = self.net.onForward([input_var])
        return [x.read() for x in output_var]
