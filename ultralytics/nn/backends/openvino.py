# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class OpenVINOBackend(BaseBackend):
    """OpenVINO inference backend.

    Supports loading and inference with OpenVINO IR models (*_openvino_model/ directories).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load the OpenVINO model."""
        LOGGER.info(f"Loading {weight} for OpenVINO inference...")
        check_requirements("openvino>=2024.0.0")
        import openvino as ov

        self.ov = ov.Core()
        device_name = "AUTO"

        if isinstance(self.device, str) and self.device.startswith("intel"):
            device_name = self.device.split(":")[1].upper()
            self.device = torch.device("cpu")
            if device_name not in self.ov.available_devices:
                LOGGER.warning(f"OpenVINO device '{device_name}' not available. Using 'AUTO' instead.")
                device_name = "AUTO"

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.xml"))

        ov_model = self.ov.read_model(model=str(w), weights=w.with_suffix(".bin"))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))
            # OpenVINO-specific: use batch and dynamic for inference mode selection
            self.batch = self.metadata.get("batch", 1)
            self.dynamic = self.metadata.get("args", {}).get("dynamic", self.dynamic)

        # Set inference mode
        self.inference_mode = "CUMULATIVE_THROUGHPUT" if self.dynamic and self.batch > 1 else "LATENCY"

        self.ov_compiled_model = self.ov.compile_model(
            ov_model,
            device_name=device_name,
            config={"PERFORMANCE_HINT": self.inference_mode},
        )
        LOGGER.info(
            f"Using OpenVINO {self.inference_mode} mode for batch={self.batch} inference on "
            f"{', '.join(self.ov_compiled_model.get_property('EXECUTION_DEVICES'))}..."
        )
        self.input_name = self.ov_compiled_model.input().get_any_name()

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run OpenVINO inference.

        Args:
            im: Input image tensor in BCHW format.

        Returns:
            Model output tensor(s).
        """
        im = im.cpu().numpy().astype(np.float32)

        if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
            # Async inference for larger batch sizes
            n = im.shape[0]
            results = [None] * n

            def callback(request, userdata):
                """Place result in preallocated list using userdata index."""
                results[userdata] = request.results

            async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
            async_queue.set_callback(callback)

            for i in range(n):
                async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)
            async_queue.wait_all()

            y = [list(r.values()) for r in results]
            y = [np.concatenate(x) for x in zip(*y)]
        else:
            # Sync inference for LATENCY mode
            y = list(self.ov_compiled_model(im).values())
        return y
