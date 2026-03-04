# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class ONNXBackend(BaseBackend):
    """ONNX Runtime inference backend.

    Supports loading and inference with ONNX models (.onnx files).
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, dnn: bool = False, **kwargs: Any):
        """Initialize ONNX backend.

        Args:
            weights: Path to the .onnx model file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            dnn: Use OpenCV DNN module instead of ONNX Runtime.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.dnn = dnn  # Keep this to distinguish DNN vs ONNX Runtime
        self.session = None
        self.net = None
        self.output_names = None
        self.io = None
        self.bindings = None
        self.use_io_binding = False

    def load_model(self) -> None:
        """Load the ONNX model."""
        cuda = self.device.type != "cpu" and torch.cuda.is_available()

        if self.dnn:
            # OpenCV DNN
            LOGGER.info(f"Loading {self.weights} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(self.weights)
        else:
            # ONNX Runtime
            LOGGER.info(f"Loading {self.weights} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            # Select execution provider
            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                    self.device = torch.device("cpu")

            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with "
                f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )

            self.session = onnxruntime.InferenceSession(self.weights, providers=providers)
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Get metadata
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))

            # Check if dynamic shapes
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type

            # Setup IO binding for CUDA
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(
                        self.device
                    )
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=self.device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run ONNX inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()

        if self.dnn:
            # OpenCV DNN
            im_np = im.cpu().numpy()
            self.net.setInput(im_np)
            y = self.net.forward()
            return self.from_numpy(y)

        # ONNX Runtime
        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
            self.io.bind_input(
                name="images",
                device_type=im.device.type,
                device_id=im.device.index if im.device.type == "cuda" else 0,
                element_type=np.float16 if self.fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            return self.bindings
        else:
            im_np = im.cpu().numpy()
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im_np})
            return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])


class ONNXIMXBackend(ONNXBackend):
    """ONNX IMX (i.MX) inference backend.

    Supports inference on NXP i.MX devices with quantized models.
    """

    def __init__(self, weights: str | Path, device: torch.device, fp16: bool = False, **kwargs: Any):
        """Initialize IMX backend.

        Args:
            weights: Path to the IMX model directory.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            **kwargs: Additional arguments.
        """
        super().__init__(weights, device, fp16, **kwargs)
        self.device = torch.device("cpu")  # IMX always uses CPU

    def load_model(self) -> None:
        """Load the IMX model."""
        check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort  # noqa - register custom NMS ops

        w = Path(self.weights)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False

        self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run IMX inference with task-specific output formatting.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        im_np = im.cpu().numpy()
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im_np})

        task = kwargs.get("task", self.task)
        if task == "detect":
            # boxes, conf, cls
            y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif task == "pose":
            # boxes, conf, kpts
            y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif task == "segment":
            y = (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )

        if isinstance(y, tuple):
            return [self.from_numpy(x) for x in y]
        return self.from_numpy(y)
