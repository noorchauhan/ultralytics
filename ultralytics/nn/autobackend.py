# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import attempt_download_asset, is_url

# Import all backends
from ultralytics.nn.backends import (
    AxeleraBackend,
    CoreMLBackend,
    ExecuTorchBackend,
    MNNBackend,
    NCNNBackend,
    ONNXBackend,
    ONNXIMXBackend,
    OpenVINOBackend,
    PaddleBackend,
    PyTorchBackend,
    RKNNBackend,
    TFLiteBackend,
    TensorFlowBackend,
    TensorRTBackend,
    TorchScriptBackend,
    TritonBackend,
)


def check_class_names(names: list | dict) -> dict[int, str]:
    """Check class names and convert to dict format if needed.

    Args:
        names (list | dict): Class names as list or dict format.

    Returns:
        (dict): Class names in dict format with integer keys and string values.

    Raises:
        KeyError: If class indices are invalid for the dataset size.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            from ultralytics.utils import ROOT, YAML

            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """Load class names from a YAML file or return numerical class names.

    Args:
        data (str | Path, optional): Path to YAML file containing class names.

    Returns:
        (dict): Dictionary mapping class indices to class names.
    """
    if data:
        try:
            from ultralytics.utils import YAML, check_yaml

            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """Handle dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |
            | Triton Inference      | triton://model    |
            | ExecuTorch            | *.pte             |
            | Axelera               | *_axelera_model/  |

    Attributes:
        backend (BaseBackend): The loaded inference backend instance.
        format (str): The model format (e.g., 'pt', 'onnx', 'engine').
        model: The underlying model (nn.Module for PyTorch backends, backend instance otherwise).
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        task (str): The type of task the model performs (detect, segment, classify, pose).
        names (dict): A dictionary of class names that the model can detect.
        stride (int): The model stride, typically 32 for YOLO models.
        fp16 (bool): Whether the model uses half-precision (FP16) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.

    Methods:
        forward: Run inference on an input image.
        from_numpy: Convert NumPy arrays to tensors on the model device.
        warmup: Warm up the model with a dummy input.
        _model_type: Determine the model type from file path.

    Examples:
        >>> model = AutoBackend(model="yolo26n.pt", device="cuda")
        >>> results = model(img)
    """

    # NHWC formats (vs torch BCHW)
    _NHWC_FORMATS = {"coreml", "saved_model", "pb", "tflite", "edgetpu", "rknn"}
    _BACKEND_MAP = {
        "pt": PyTorchBackend,
        "jit": TorchScriptBackend,
        "onnx": ONNXBackend,
        "dnn": ONNXBackend,  # Special case: ONNX with DNN
        "xml": OpenVINOBackend,
        "engine": TensorRTBackend,
        "coreml": CoreMLBackend,
        "saved_model": TensorFlowBackend,
        "pb": TensorFlowBackend,
        "tflite": TFLiteBackend,
        "edgetpu": TFLiteBackend,
        "paddle": PaddleBackend,
        "mnn": MNNBackend,
        "ncnn": NCNNBackend,
        "imx": ONNXIMXBackend,
        "rknn": RKNNBackend,
        "triton": TritonBackend,
        "pte": ExecuTorchBackend,
        "axelera": AxeleraBackend,
    }

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize the AutoBackend for inference.

        Args:
            model (str | torch.nn.Module): Path to the model weights file or a module instance.
            device (torch.device): Device to run the model on.
            dnn (bool): Use OpenCV DNN module for ONNX inference.
            data (str | Path, optional): Path to the additional data.yaml file containing class names.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Enable verbose logging.
        """
        super().__init__()
        nn_module = isinstance(model, nn.Module)

        # Determine model format from path/URL
        self.format = "nn_module" if nn_module else self._model_type(model, dnn)

        # Check if format supports FP16
        fp16_supported = self.format in {"pt", "jit", "onnx", "xml", "engine", "triton"} or nn_module
        fp16 &= fp16_supported

        # Set device
        cuda = isinstance(device, torch.device) and torch.cuda.is_available() and device.type != "cpu"
        if cuda and self.format not in {"pt", "jit", "engine", "onnx", "paddle"} and not nn_module:
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        w = attempt_download_asset(model) if self.format == "pt" else model

        # Select and initialize the appropriate backend
        backend_kwargs = {"device": device, "fp16": fp16}

        if self.format not in self._BACKEND_MAP:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. "
                f"Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )
        if nn_module or self.format == "pt":
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif self.format == "dnn":
            backend_kwargs["dnn"] = True
        elif self.format == "saved_model":
            backend_kwargs["is_savedmodel"] = True
        elif self.format == "pb":
            backend_kwargs["is_savedmodel"] = False
        elif self.format == "tflite":
            backend_kwargs["edgetpu"] = False
        elif self.format == "edgetpu":
            backend_kwargs["edgetpu"] = True
        elif self.format == "tfjs":
            raise NotImplementedError("Ultralytics TF.js inference is not currently supported.")
        self.backend = self._BACKEND_MAP[self.format](w, **backend_kwargs)
        self.backend.load_model()

        # Determine NHWC based on format
        self.nhwc = self.format in self._NHWC_FORMATS

        # Apply metadata from backend
        self._apply_metadata(data)

        # Expose backend attributes for backward compatibility
        self._expose_backend_attrs()

    def _apply_metadata(self, data: str | Path | None = None):
        """Apply metadata from backend to AutoBackend.

        Args:
            data: Path to data.yaml file with class names.
        """
        # Load external metadata if needed
        metadata = getattr(self.backend, "metadata", None)
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            from ultralytics.utils import YAML

            metadata = YAML.load(metadata)

        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                    metadata[k] = ast.literal_eval(v)

            # Apply to backend
            for k, v in metadata.items():
                if hasattr(self.backend, k):
                    setattr(self.backend, k, v)
                setattr(self, k, v)  # TODO

        # Check names
        if not hasattr(self.backend, "names") or not self.backend.names:
            self.backend.names = default_class_names(data)
        self.backend.names = check_class_names(self.backend.names)

    def _expose_backend_attrs(self):
        """Expose backend attributes on AutoBackend for backward compatibility."""
        # Copy all relevant attributes from backend to self
        attrs_to_copy = [
            "device",
            "fp16",
            "stride",
            "names",
            "task",
            "batch",
            "imgsz",
            "ch",
            "end2end",
            "dynamic",
            "nhwc",
        ]

        for attr in attrs_to_copy:
            if hasattr(self.backend, attr):
                setattr(self, attr, getattr(self.backend, attr))
            else:
                setattr(self, attr, False)

        # Handle model attribute specially - for PyTorch models it should be the actual model
        if hasattr(self.backend, "model"):
            self.model = self.backend.model
        else:
            self.model = self.backend

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on an AutoBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): Whether to perform data augmentation during inference.
            visualize (bool): Whether to visualize the output predictions.
            embed (list, optional): A list of layer indices to return embeddings from.
            **kwargs (Any): Additional keyword arguments for model configuration.

        Returns:
            (torch.Tensor | list[torch.Tensor]): The raw output tensor(s) from the model.
        """
        _, _, h, w = im.shape
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        # Build forward kwargs based on backend type
        if self.format in {"pt", "nn_module"}:
            forward_kwargs = {"augment": augment, "visualize": visualize, "embed": embed, **kwargs}
        else:
            # Pass task and image dimensions for coordinate scaling (used by some backends)
            forward_kwargs = {"task": self.task, "h": h, "w": w, **kwargs}

        y = self.backend.forward(im, **forward_kwargs)

        # Handle single output
        if not isinstance(y, (list, tuple)):
            y = [y]

        # Update names if needed (for segmentation)
        if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):
            nc = y[0].shape[1] - y[1].shape[1] - 4
            self.names = {i: f"class{i}" for i in range(nc)}
            self.backend.names = self.names

        return y[0] if len(y) == 1 else y

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a NumPy array to a torch tensor on the model device.

        Args:
            x (np.ndarray | torch.Tensor): Input array or tensor.

        Returns:
            (torch.Tensor): Tensor on `self.device`.
        """
        return self.backend.from_numpy(x)

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """Warm up the model by running forward pass(es) with a dummy input.

        Args:
            imgsz (tuple[int, int, int, int]): Dummy input shape in (batch, channels, height, width) format.
        """
        # Delegate to backend's warmup method
        self.backend.warmup(imgsz)

    @staticmethod
    def _model_type(p: str = "path/to/model.pt", dnn: bool = False) -> list[bool]:
        """Take a path to a model file and return the model type.

        Args:
            p (str): Path to the model file.

        Returns:
            (list[bool]): List of booleans indicating the model type.

        Examples:
            >>> types = AutoBackend._model_type("path/to/model.onnx")
            >>> assert types[2]  # onnx
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        types[8] &= not types[9]
        format = next((f for i, f in enumerate(export_formats()["Argument"]) if types[i]), None)
        if format == "onnx" and dnn:
            format = "dnn"
        if not any(types):
            from urllib.parse import urlsplit

            url = urlsplit(p)
            if bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}:
                format = "triton"
        return format
