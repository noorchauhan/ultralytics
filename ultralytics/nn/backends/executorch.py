# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_executorch_requirements


class ExecuTorchBackend(BaseBackend):
    """ExecuTorch inference backend.

    Supports loading and inference with ExecuTorch models (.pte files).
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize ExecuTorch backend.

        Args:
            weights: Path to the .pte model file or directory.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.program = None
        self.model = None

    def load_model(self) -> None:
        """Load the ExecuTorch model."""
        LOGGER.info(f"Loading {self.weights} for ExecuTorch inference...")
        check_executorch_requirements()

        from executorch.runtime import Runtime

        w = Path(self.weights)
        if w.is_dir():
            model_file = next(w.rglob("*.pte"))
            metadata_file = w / "metadata.yaml"
        else:
            model_file = w
            metadata_file = w.parent / "metadata.yaml"

        self.program = Runtime.get().load_program(str(model_file))
        self.model = self.program.load_method("forward")

        # Load metadata
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run ExecuTorch inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        y = self.model.execute([im])

        if isinstance(y, (list, tuple)):
            return [self.from_numpy(x) for x in y] if not isinstance(y[0], torch.Tensor) else y
        return self.from_numpy(y) if not isinstance(y, torch.Tensor) else y
