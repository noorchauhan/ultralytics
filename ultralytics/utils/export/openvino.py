# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch

from ultralytics.utils import LOGGER, YAML


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str | None = None,
    dynamic: bool = False,
    half: bool = False,
    int8: bool = False,
    calibration_dataset: Any | None = None,
    transform_fn: Callable | None = None,
    ignored_scope_args: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to OpenVINO format with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor.
        file (Path | str): Source model path used to derive output directory.
        dynamic (bool): Whether to use dynamic input shapes.
        half (bool): Whether to compress to FP16.
        int8 (bool): Whether to apply INT8 quantization.
        iou (float): IoU threshold for RT info.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        task (str): Model task (``"detect"``, ``"segment"``, etc.).
        model_names (dict | None): Class names dict for RT info labels.
        calibration_dataset: Dataset for nncf.Dataset (required when ``int8=True``).
        transform_fn: Transform function for calibration preprocessing.
        ignored_scope_args (dict | None): Kwargs passed to ``nncf.IgnoredScope`` for head patterns.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_openvino_model`` or ``_int8_openvino_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_2_3

    check_requirements("openvino>=2025.2.0" if _is_macos_154() else "openvino>=2024.0.0")
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")

    ov_model = ov.convert_model(model, input=None if dynamic else [im.shape], example_input=im)
    if int8:
        check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
        check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
        import nncf

        ignored_scope = None
        if ignored_scope_args:
            ignored_scope = nncf.IgnoredScope(**ignored_scope_args)

        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=nncf.Dataset(calibration_dataset, transform_fn),
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
        )

    if file is not None:
        file = Path(file)
        suffix = f"_{'int8_' if int8 else ''}openvino_model{os.sep}"
        f = str(file).replace(file.suffix, suffix)
        f_ov = str(Path(f) / file.with_suffix(".xml").name)
        ov.save_model(ov_model, f_ov, compress_to_fp16=half)
    return ov_model


def _is_macos_154() -> bool:
    """Return True if running on macOS 15.4 or newer."""
    try:
        from ultralytics.utils import MACOS, MACOS_VERSION

        return MACOS and MACOS_VERSION >= "15.4"
    except Exception:
        return False
