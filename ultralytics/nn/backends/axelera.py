# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class AxeleraBackend(BaseBackend):
    """Axelera inference backend.

    Supports loading and inference with Axelera models on Axelera hardware.
    """

    def load_model(self, weight: str | Path) -> None:
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

        w = Path(weight)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        self.model = op.load(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run Axelera inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor(s).
        """
        return self.model(im.cpu())
