# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import types
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.torch_utils import TORCH_2_4, TORCH_2_9


def best_onnx_opset(onnx: types.ModuleType, cuda: bool = False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback.

    Args:
        onnx: The ``onnx`` module.
        cuda (bool): Whether export is targeting CUDA.

    Returns:
        (int): The recommended ONNX opset version.
    """
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if TORCH_2_9:
            opset = min(opset, 20)  # legacy TorchScript exporter caps at opset 20 in torch 2.9+
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())


def export_onnx_model(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    opset: int | None = None,
    dynamic: bool = False,
    half: bool = False,
    simplify: bool = False,
    nms: bool = False,
    fmt: str = "onnx",
    metadata: dict | None = None,
    device: torch.device | None = None,
    task: str = "detect",
    prefix: str = "",
) -> str:
    """Export a PyTorch model to ONNX format with optional simplification and metadata.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor.
        file (Path | str): Source model path used to derive the ``.onnx`` output path.
        opset (int | None): ONNX opset version. If None, auto-detected.
        dynamic (bool): Whether to use dynamic axes.
        half (bool): Whether to convert to FP16.
        simplify (bool): Whether to simplify the ONNX graph.
        nms (bool): Whether NMS is embedded in the model.
        fmt (str): Export format name (e.g. ``"onnx"``, ``"engine"``).
        metadata (dict | None): Key-value metadata to embed in the ONNX file.
        device (torch.device | None): Device the model lives on.
        task (str): Model task, e.g. ``"detect"``, ``"segment"``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ONNX file.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.export.engine import torch2onnx
    from ultralytics.utils.patches import arange_patch

    requirements = ["onnx>=1.12.0,<2.0.0"]
    if simplify:
        requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
    check_requirements(requirements)
    import onnx

    opset = opset or best_onnx_opset(onnx, cuda="cuda" in str(device))
    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")

    f = str(Path(file).with_suffix(".onnx"))
    output_names = ["output0", "output1"] if task == "segment" else ["output0"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if task == "segment":
            dynamic_axes["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
            dynamic_axes["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif task in {"detect", "pose", "obb"}:
            dynamic_axes["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
        if nms:  # only batch size is dynamic with NMS
            dynamic_axes["output0"].pop(2, None)

    with arange_patch(dynamic=dynamic, half=half, fmt=fmt):
        torch2onnx(
            model,
            im,
            f,
            opset=opset,
            input_names=["images"],
            output_names=output_names,
            dynamic=dynamic_axes,
        )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model

    # Simplify
    if simplify:
        try:
            import onnxslim

            LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
            model_onnx = onnxslim.slim(model_onnx)
        except Exception as e:
            LOGGER.warning(f"{prefix} simplifier failure: {e}")

    # Metadata
    for k, v in (metadata or {}).items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    # IR version
    if getattr(model_onnx, "ir_version", 0) > 10:
        LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
        model_onnx.ir_version = 10

    # FP16 conversion for CPU export
    if half and fmt == "onnx" and device is not None and device.type == "cpu":
        try:
            from onnxruntime.transformers import float16

            LOGGER.info(f"{prefix} converting to FP16...")
            model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
        except Exception as e:
            LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

    onnx.save(model_onnx, f)
    return f
