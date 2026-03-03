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
        model: The underlying model (nn.Module for PyTorch backends, backend instance otherwise).
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        task (str): The type of task the model performs (detect, segment, classify, pose).
        names (dict): A dictionary of class names that the model can detect.
        stride (int): The model stride, typically 32 for YOLO models.
        fp16 (bool): Whether the model uses half-precision (FP16) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.
        pt (bool): Whether the model is a PyTorch model.
        jit (bool): Whether the model is a TorchScript model.
        onnx (bool): Whether the model is an ONNX model.
        xml (bool): Whether the model is an OpenVINO model.
        engine (bool): Whether the model is a TensorRT engine.
        coreml (bool): Whether the model is a CoreML model.
        saved_model (bool): Whether the model is a TensorFlow SavedModel.
        pb (bool): Whether the model is a TensorFlow GraphDef.
        tflite (bool): Whether the model is a TensorFlow Lite model.
        edgetpu (bool): Whether the model is a TensorFlow Edge TPU model.
        tfjs (bool): Whether the model is a TensorFlow.js model.
        paddle (bool): Whether the model is a PaddlePaddle model.
        mnn (bool): Whether the model is an MNN model.
        ncnn (bool): Whether the model is an NCNN model.
        imx (bool): Whether the model is an IMX model.
        rknn (bool): Whether the model is an RKNN model.
        triton (bool): Whether the model is a Triton Inference Server model.
        pte (bool): Whether the model is a PyTorch ExecuTorch model.
        axelera (bool): Whether the model is an Axelera model.

    Methods:
        forward: Run inference on an input image.
        from_numpy: Convert NumPy arrays to tensors on the model device.
        warmup: Warm up the model with a dummy input.
        _model_type: Determine the model type from file path.

    Examples:
        >>> model = AutoBackend(model="yolo26n.pt", device="cuda")
        >>> results = model(img)
    """

    # Mapping of model type flags to backend classes
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

        # Determine model type from path/URL
        model_types = self._model_type("" if nn_module else model)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            pte,
            axelera,
            triton,
            format,
        ) = model_types

        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16

        # Set device
        cuda = isinstance(device, torch.device) and torch.cuda.is_available() and device.type != "cpu"
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        w = attempt_download_asset(model) if pt else model

        # Initialize all format flags to False
        self._init_format_flags()

        # Select and initialize the appropriate backend
        backend_kwargs = {"device": device, "fp16": fp16}

        if nn_module or pt:
            self.pt = True
            self.nn_module = nn_module
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif jit:
            self.jit = True
        elif dnn:
            self.onnx = True
            self.dnn = True
            backend_kwargs["dnn"] = True
        elif onnx:
            self.onnx = True
        elif imx:
            self.imx = True
        elif xml:
            self.xml = True
        elif engine:
            self.engine = True
        elif coreml:
            self.coreml = True
        elif saved_model:
            self.saved_model = True
            backend_kwargs["is_savedmodel"] = True
        elif pb:
            self.pb = True
            backend_kwargs["is_savedmodel"] = False
        elif tflite:
            self.tflite = True
            backend_kwargs["edgetpu"] = False
        elif edgetpu:
            self.edgetpu = True
            self.tflite = True  # Edge TPU is a type of TFLite
            backend_kwargs["edgetpu"] = True
        elif tfjs:
            self.tfjs = True
            raise NotImplementedError("Ultralytics TF.js inference is not currently supported.")
        elif paddle:
            self.paddle = True
        elif mnn:
            self.mnn = True
        elif ncnn:
            self.ncnn = True
        elif rknn:
            self.rknn = True
        elif triton:
            self.triton = True
        elif pte:
            self.pte = True
        elif axelera:
            self.axelera = True
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. "
                f"Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )
        self.backend = self._BACKEND_MAP[format](w, **backend_kwargs)
        self.backend.load_model()

        # Determine NHWC based on format
        self.nhwc = self._get_active_format() in self._NHWC_FORMATS

        # Apply metadata from backend
        self._apply_metadata(data)

        # Expose backend attributes for backward compatibility
        self._expose_backend_attrs()

    def _init_format_flags(self):
        """Initialize all format flags to False."""
        self.pt = False
        self.jit = False
        self.onnx = False
        self.dnn = False
        self.xml = False
        self.engine = False
        self.coreml = False
        self.saved_model = False
        self.pb = False
        self.tflite = False
        self.edgetpu = False
        self.tfjs = False
        self.paddle = False
        self.mnn = False
        self.ncnn = False
        self.imx = False
        self.rknn = False
        self.triton = False
        self.pte = False
        self.axelera = False
        self.nn_module = False

    def _get_active_format(self) -> str:
        """Get the active format name based on which flag is True."""
        formats = [
            ("pt", self.pt),
            ("jit", self.jit),
            ("onnx", self.onnx),
            ("xml", self.xml),
            ("engine", self.engine),
            ("coreml", self.coreml),
            ("saved_model", self.saved_model),
            ("pb", self.pb),
            ("tflite", self.tflite),
            ("edgetpu", self.edgetpu),
            ("paddle", self.paddle),
            ("mnn", self.mnn),
            ("ncnn", self.ncnn),
            ("imx", self.imx),
            ("rknn", self.rknn),
            ("triton", self.triton),
            ("pte", self.pte),
            ("axelera", self.axelera),
        ]
        for name, flag in formats:
            if flag:
                return name
        return ""

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
        ]

        for attr in attrs_to_copy:
            if hasattr(self.backend, attr):
                setattr(self, attr, getattr(self.backend, attr))

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
        if self.pt or self.nn_module:
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
        from ultralytics.utils.nms import non_max_suppression

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)
            for _ in range(2 if self.jit else 1):
                self.forward(im)
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)

    @staticmethod
    def _model_type(p: str = "path/to/model.pt") -> list[bool]:
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

        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}
            format = "triton"

        return [*types, triton, format]
