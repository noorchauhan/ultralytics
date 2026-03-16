# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    args,
    metadata: dict | None = None,
    task: str = "detect",
    model_names: dict | None = None,
    calibration_dataset=None,
    transform_fn=None,
    ignored_scope_args: dict | None = None,
    prefix: str = "",
) -> str:
    """Export a PyTorch model to OpenVINO format with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The model to export (may be NMS-wrapped).
        im (torch.Tensor): Example input tensor.
        file (Path | str): Source model path used to derive output directory.
        args: Export arguments (``dynamic``, ``half``, ``int8``, ``iou``, ``format``).
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        task (str): Model task (``"detect"``, ``"segment"``, etc.).
        model_names (dict | None): Class names dict for RT info labels.
        calibration_dataset: Dataset for nncf.Dataset (required when ``args.int8``).
        transform_fn: Transform function for calibration preprocessing.
        ignored_scope_args (dict | None): Kwargs passed to ``nncf.IgnoredScope`` for head patterns.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_openvino_model`` or ``_int8_openvino_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_2_1, TORCH_2_3

    check_requirements("openvino>=2025.2.0" if _is_macos_154() else "openvino>=2024.0.0")
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    from ultralytics.utils.torch_utils import TORCH_VERSION

    assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

    file = Path(file)
    model_names = model_names or {}

    ov_model = ov.convert_model(
        model,
        input=None if args.dynamic else [im.shape],
        example_input=im,
    )

    def serialize(ov_model, out_file: str) -> None:
        """Set RT info, serialize model, and save metadata YAML."""
        ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
        ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
        ov_model.set_rt_info(114, ["model_info", "pad_value"])
        ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
        ov_model.set_rt_info(args.iou, ["model_info", "iou_threshold"])
        ov_model.set_rt_info([v.replace(" ", "_") for v in model_names.values()], ["model_info", "labels"])
        if task != "classify":
            ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])
        ov.save_model(ov_model, out_file, compress_to_fp16=args.half)
        YAML.save(Path(out_file).parent / "metadata.yaml", metadata or {})

    if args.int8:
        fq = str(file).replace(file.suffix, f"_int8_openvino_model{os.sep}")
        fq_ov = str(Path(fq) / file.with_suffix(".xml").name)
        check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
        check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
        import nncf

        ignored_scope = None
        if ignored_scope_args:
            ignored_scope = nncf.IgnoredScope(**ignored_scope_args)

        quantized_ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=nncf.Dataset(calibration_dataset, transform_fn),
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
        )
        serialize(quantized_ov_model, fq_ov)
        return fq

    f = str(file).replace(file.suffix, f"_openvino_model{os.sep}")
    f_ov = str(Path(f) / file.with_suffix(".xml").name)
    serialize(ov_model, f_ov)
    return f


def _is_macos_154() -> bool:
    """Return True if running on macOS 15.4 or newer."""
    try:
        from ultralytics.utils import MACOS, MACOS_VERSION

        return MACOS and MACOS_VERSION >= "15.4"
    except Exception:
        return False
