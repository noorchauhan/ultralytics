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

    def load_model(self, weight: str | Path) -> None:
        """Load the ExecuTorch model."""
        LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
        check_executorch_requirements()

        from executorch.runtime import Runtime

        w = Path(weight)
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

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run ExecuTorch inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor(s).
        """
        return self.model.execute([im])
