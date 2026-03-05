# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import IS_JETSON, LOGGER, is_jetson

from .base import BaseBackend


class PyTorchBackend(BaseBackend):
    """PyTorch inference backend.

    Supports loading and inference with native PyTorch models (.pt files).
    """

    def __init__(
        self,
        weight: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize PyTorch backend.

        Args:
            weight: Path to the .pt model file or nn.Module instance.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            fuse: Whether to fuse Conv2D + BatchNorm layers.
            verbose: Whether to print verbose messages.
        """
        self.fuse = fuse
        self.verbose = verbose
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | torch.nn.Module) -> None:
        """Load the PyTorch model."""
        from ultralytics.nn.tasks import load_checkpoint

        if isinstance(weight, torch.nn.Module):
            if self.fuse and hasattr(weight, "fuse"):
                if IS_JETSON and is_jetson(jetpack=5):
                    weight = weight.to(self.device)
                weight = weight.fuse(verbose=self.verbose)
            model = weight.to(self.device)
        else:
            model, _ = load_checkpoint(weight, device=self.device, fuse=self.fuse)

        # Extract model attributes
        if hasattr(model, "kpt_shape"):
            self.kpt_shape = model.kpt_shape
        self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
        self.ch = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
        model.half() if self.fp16 else model.float()

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

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False):
        """Initialize TorchScript backend.

        Args:
            weight: Path to the .torchscript model file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
        """
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str) -> None:
        """Load the TorchScript model."""
        import torchvision  # noqa
        import json

        LOGGER.info(f"Loading {weight} for TorchScript inference...")
        extra_files = {"config.txt": ""}
        self.model = torch.jit.load(weight, _extra_files=extra_files, map_location=self.device)
        self.model.half() if self.fp16 else self.model.float()

        if extra_files["config.txt"]:
            self.apply_metadata(json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items())))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run TorchScript inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor.
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im)
