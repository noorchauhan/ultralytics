# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class NCNNBackend(BaseBackend):
    """NCNN inference backend.

    Supports loading and inference with NCNN models (*_ncnn_model/ directories).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the NCNN model."""
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()

        # Setup Vulkan if available
        cuda = self.device.type != "cpu" and torch.cuda.is_available()
        if isinstance(self.device, str) and self.device.startswith("vulkan"):
            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(int(self.device.split(":")[1]))
            self.device = torch.device("cpu")
        elif cuda:
            self.net.opt.use_vulkan_compute = True

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.param"))

        self.net.load_param(str(w))
        self.net.load_model(str(w.with_suffix(".bin")))

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run NCNN inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor(s).
        """
        mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            # Sort output names as temporary fix for pnnx issue
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])
