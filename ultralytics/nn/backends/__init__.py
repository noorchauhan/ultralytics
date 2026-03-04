# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Ultralytics YOLO inference backends.

This package provides modular inference backends for various frameworks.
Each backend is implemented as a separate class that can be used independently
or through the unified AutoBackend API.
"""

from .base import BaseBackend
from .pytorch import PyTorchBackend, TorchScriptBackend
from .onnx import ONNXBackend, ONNXIMXBackend
from .openvino import OpenVINOBackend
from .tensorrt import TensorRTBackend
from .coreml import CoreMLBackend
from .tensorflow import TFLiteBackend, TensorFlowBackend
from .paddle import PaddleBackend
from .mnn import MNNBackend
from .ncnn import NCNNBackend
from .rknn import RKNNBackend
from .executorch import ExecuTorchBackend
from .axelera import AxeleraBackend
from .triton import TritonBackend

__all__ = [
    "BaseBackend",
    "PyTorchBackend",
    "TorchScriptBackend",
    "ONNXBackend",
    "ONNXIMXBackend",
    "OpenVINOBackend",
    "TensorRTBackend",
    "CoreMLBackend",
    "TensorFlowBackend",
    "TFLiteBackend",
    "PaddleBackend",
    "MNNBackend",
    "NCNNBackend",
    "RKNNBackend",
    "ExecuTorchBackend",
    "AxeleraBackend",
    "TritonBackend",
]
