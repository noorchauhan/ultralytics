# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import json
import platform
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER


class TensorFlowBackend(BaseBackend):
    """TensorFlow SavedModel and GraphDef inference backend.

    Supports loading and inference with TensorFlow SavedModel and GraphDef formats.
    """

    def __init__(
        self, weight: str | Path, device: torch.device, fp16: bool = False, is_savedmodel: bool = True, **kwargs: Any
    ):
        """Initialize TensorFlow backend.

        Args:
            weight: Path to the SavedModel directory or .pb file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            is_savedmodel: Whether weight is a SavedModel (True) or GraphDef (False).
            **kwargs: Additional arguments.
        """
        self.saved_model = is_savedmodel  # Keep to distinguish SavedModel vs GraphDef
        self.pb = not is_savedmodel
        super().__init__(weight, device, fp16, **kwargs)

    def load_model(self, weight: str | Path) -> None:
        """Load the TensorFlow model."""
        import tensorflow as tf

        if self.saved_model:
            LOGGER.info(f"Loading {weight} for TensorFlow SavedModel inference...")
            self.model = tf.saved_model.load(weight)
            # Load metadata
            metadata_file = Path(weight) / "metadata.yaml"
            if metadata_file.exists():
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
        else:
            LOGGER.info(f"Loading {weight} for TensorFlow GraphDef inference...")
            from ultralytics.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap frozen graphs for deployment."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()
            with open(weight, "rb") as f:
                gd.ParseFromString(f.read())
            self.frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))

            # Try to find metadata
            try:
                metadata_file = next(
                    Path(weight).resolve().parent.rglob(f"{Path(weight).stem}_saved_model*/metadata.yaml")
                )
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
            except StopIteration:
                pass

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run TensorFlow inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments.

        Returns:
            Model output tensor(s).
        """
        import tensorflow as tf

        im_np = im.cpu().numpy()

        if self.saved_model:
            y = self.model.serving_default(im_np)
            if not isinstance(y, list):
                y = list(y.values()) if hasattr(y, "values") else [y]
        else:
            y = self.frozen_func(x=tf.constant(im_np))
            if not isinstance(y, list):
                y = [y]

        # Convert to numpy
        y = [x.numpy() if hasattr(x, "numpy") else x for x in y]

        # Handle segmentation task output ordering
        task = kwargs.get("task", self.task)
        if task == "segment":
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
            if y[1].shape[-1] == 6:  # end-to-end model
                y = [y[1]]
            else:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))

        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])


class TFLiteBackend(BaseBackend):
    """TensorFlow Lite and Edge TPU inference backend.

    Supports loading and inference with TFLite models (.tflite files) and Edge TPU models.
    """

    def __init__(
        self, weight: str | Path, device: torch.device, fp16: bool = False, edgetpu: bool = False, **kwargs: Any
    ):
        """Initialize TFLite backend.

        Args:
            weight: Path to the .tflite model file.
            device: Device to run inference on.
            fp16: Whether to use FP16 precision.
            edgetpu: Whether this is an Edge TPU model.
            **kwargs: Additional arguments.
        """
        self.edgetpu = edgetpu  # Keep to distinguish Edge TPU vs regular TFLite
        super().__init__(weight, device, fp16, **kwargs)

    def load_model(self, weight: str | Path) -> None:
        """Load the TFLite model."""
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate

            self.tf = None
        except ImportError:
            import tensorflow as tf

            self.tf = tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

        if self.edgetpu:
            device = device[3:] if str(self.device).startswith("tpu") else ":0"
            LOGGER.info(f"Loading {weight} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
            delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                platform.system()
            ]
            self.interpreter = Interpreter(
                model_path=str(weight),
                experimental_delegates=[load_delegate(delegate, options={"device": device})],
            )
            self.device = torch.device("cpu")  # Required, otherwise PyTorch will try to use the wrong device
        else:
            LOGGER.info(f"Loading {weight} for TensorFlow Lite inference...")
            self.interpreter = Interpreter(model_path=weight)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load metadata
        try:
            with zipfile.ZipFile(weight, "r") as zf:
                name = zf.namelist()[0]
                contents = zf.read(name).decode("utf-8")
                if name == "metadata.json":
                    self.apply_metadata(json.loads(contents))
                else:
                    self.apply_metadata(ast.literal_eval(contents))
        except (zipfile.BadZipFile, SyntaxError, ValueError, json.JSONDecodeError):
            pass

    def forward(self, im: torch.Tensor, **kwargs: Any) -> torch.Tensor | list[torch.Tensor]:
        """Run TFLite inference.

        Args:
            im: Input image tensor in BCHW format.
            **kwargs: Additional arguments (includes h, w for coordinate scaling).

        Returns:
            Model output tensor(s).
        """
        im = im.cpu().numpy()
        h, w = kwargs.get("h", im.shape[2]), kwargs.get("w", im.shape[3])

        details = self.input_details[0]
        is_int = details["dtype"] in {np.int8, np.int16}

        if is_int:
            scale, zero_point = details["quantization"]
            im = (im / scale + zero_point).astype(details["dtype"])

        self.interpreter.set_tensor(details["index"], im)
        self.interpreter.invoke()

        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            if is_int:
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale
            if x.ndim == 3:
                # Denormalize xywh by image size
                if x.shape[-1] == 6 or self.end2end:
                    x[:, :, [0, 2]] *= w
                    x[:, :, [1, 3]] *= h
                    if self.task == "pose":
                        x[:, :, 6::3] *= w
                        x[:, :, 7::3] *= h
                else:
                    x[:, [0, 2]] *= w
                    x[:, [1, 3]] *= h
                    if self.task == "pose":
                        x[:, 5::3] *= w
                        x[:, 6::3] *= h
            y.append(x)

        # Handle segmentation
        task = kwargs.get("task", self.task)
        if task == "segment":
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
            if y[1].shape[-1] == 6:
                y = [y[1]]
            else:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))

        return [self.from_numpy(x) for x in y] if len(y) > 1 else self.from_numpy(y[0])
