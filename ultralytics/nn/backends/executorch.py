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
