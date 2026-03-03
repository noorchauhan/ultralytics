# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import IS_JETSON, LOGGER, is_jetson
from ultralytics.nn.backends.base import BaseBackend


class PyTorchBackend(BaseBackend):
    """PyTorch inference backend.

    Supports loading and inference with native PyTorch models (.pt files).
    """

    def __init__(
        self,
        weights: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ):
        """Initialize PyTorch backend.

        Args:
            weights: Path to the .pt model file or nn.Module instance.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            fuse: Whether to fuse Conv2D + BatchNorm layers.
            verbose: Whether to print verbose messages.
            **kwargs: Additional arguments.
        """
        self.nn_module = isinstance(weights, nn.Module)
        if self.nn_module:
            super().__init__("", device, fp16, **kwargs)
            self._model_instance = weights
        else:
            super().__init__(weights, device, fp16, **kwargs)
            self._model_instance = None
        self.fuse = fuse
        self.verbose = verbose

    def load_model(self) -> None:
        """Load the PyTorch model."""
        from ultralytics.nn.tasks import load_checkpoint

        if self.nn_module:
            model = self._model_instance
            if self.fuse and hasattr(model, "fuse"):
                if IS_JETSON and is_jetson(jetpack=5):
                    model = model.to(self.device)
                model = model.fuse(verbose=self.verbose)
            model = model.to(self.device)
        else:
            model, _ = load_checkpoint(self.weights, device=self.device, fuse=self.fuse)

        # Extract model attributes
        if hasattr(model, "kpt_shape"):
            self.kpt_shape = model.kpt_shape
        self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
        self.ch = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3

        if self.fp16:
            model.half()
        else:
            model.float()

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.end2end = getattr(model, "end2end", False)

    def forward(
        self, im: torch.Tensor, augment: bool = False, visualize: bool = False, embed: list | None = None, **kwargs: Any
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run PyTorch inference.

        Args:
            im: Input image tensor in BCHW format.
            augment: Whether to use test-time augmentation.
            visualize: Whether to visualize features.
            embed: Layers to extract embeddings from.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)


class TorchScriptBackend(BaseBackend):
    """TorchScript inference backend.

    Supports loading and inference with TorchScript models (.torchscript files).
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize TorchScript backend.

        Args:
            weights: Path to the .torchscript model file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.metadata = None

    def load_model(self) -> None:
        """Load the TorchScript model."""
        import torchvision  # noqa
        import json

        LOGGER.info(f"Loading {self.weights} for TorchScript inference...")
        extra_files = {"config.txt": ""}
        self.model = torch.jit.load(self.weights, _extra_files=extra_files, map_location=self.device)

        if self.fp16:
            self.model.half()
        else:
            self.model.float()

        if extra_files["config.txt"]:
            self.metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run TorchScript inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor.
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im)
