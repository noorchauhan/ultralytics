# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import ARM64, LOGGER
from ultralytics.utils.checks import check_requirements


class PaddleBackend(BaseBackend):
    """PaddlePaddle inference backend.

    Supports loading and inference with PaddlePaddle models (*_paddle_model/ directories).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the PaddlePaddle model."""
        cuda = torch.cuda.is_available()

        LOGGER.info(f"Loading {weight} for PaddlePaddle inference...")
        if cuda:
            check_requirements("paddlepaddle-gpu>=3.0.0,!=3.3.0")
        elif ARM64:
            check_requirements("paddlepaddle==3.0.0")
        else:
            check_requirements("paddlepaddle>=3.0.0,!=3.3.0")

        import paddle.inference as pdi

        w = Path(weight)
        model_file, params_file = None, None

        if w.is_dir():
            model_file = next(w.rglob("*.json"), None)
            params_file = next(w.rglob("*.pdiparams"), None)
        elif w.suffix == ".pdiparams":
            model_file = w.with_name("model.json")
            params_file = w

        if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
            raise FileNotFoundError(f"Paddle model not found in {w}. Both .json and .pdiparams files are required.")

        config = pdi.Config(str(model_file), str(params_file))
        if torch.cuda.is_available() and self.device.type != "cpu":
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)

        self.predictor = pdi.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_names = self.predictor.get_output_names()

        # Load metadata
        metadata_file = w / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run PaddlePaddle inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        self.input_handle.copy_from_cpu(im.cpu().numpy().astype(np.float32))
        self.predictor.run()
        y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])
