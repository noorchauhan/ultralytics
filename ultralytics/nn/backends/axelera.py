# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class AxeleraBackend(BaseBackend):
    """Axelera inference backend.

    Supports loading and inference with Axelera models on Axelera hardware.
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize Axelera backend.

        Args:
            weights: Path to the Axelera model directory.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.model = None

    def load_model(self) -> None:
        """Load the Axelera model."""
        if not os.environ.get("AXELERA_RUNTIME_DIR"):
            LOGGER.warning(
                "Axelera runtime environment is not activated.\n"
                "Please run: source /opt/axelera/sdk/latest/axelera_activate.sh\n\n"
                "If this fails, verify driver installation: "
                "https://docs.ultralytics.com/integrations/axelera/#axelera-driver-installation"
            )

        try:
            from axelera.runtime import op
        except ImportError:
            check_requirements(
                "axelera_runtime2==0.1.2",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi",
            )
            from axelera.runtime import op

        w = Path(self.weights)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        self.model = op.load(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run Axelera inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        y = self.model(im.cpu())

        if isinstance(y, (list, tuple)):
            return [self.from_numpy(x) for x in y] if not isinstance(y[0], torch.Tensor) else y
        return self.from_numpy(y) if not isinstance(y, torch.Tensor) else y
