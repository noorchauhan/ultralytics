# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ultralytics.utils import YAML


def torch2axelera(
    f_onnx: str,
    model_str: str,
    args: SimpleNamespace,
    metadata: dict | None = None,
    calibration_dataset: Any | None = None,
    transform_fn: Callable | None = None,
    prefix: str = "",
):
    """Export an ONNX model to Axelera format.

    Args:
        f_onnx (str): Path to the source ONNX file (already exported).
        model_str (str): String representation of the model (``str(model)``) for architecture detection.
        args (SimpleNamespace): Export arguments. ``args.int8`` must be ``True``.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        calibration_dataset: Dataloader for INT8 calibration.
        transform_fn: Transformation function applied to calibration batches.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_axelera_model`` directory.
    """
    import os

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    try:
        from axelera import compiler
    except ImportError:
        from ultralytics.utils.checks import check_apt_requirements, check_requirements

        check_apt_requirements(
            ["libllvm14", "libgirepository1.0-dev", "pkg-config", "libcairo2-dev", "build-essential", "cmake"]
        )
        check_requirements(
            "axelera-voyager-sdk==1.5.2",
            cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi "
            "--extra-index-url https://software.axelera.ai/artifactory/axelera-dev-pypi",
        )

    from axelera import compiler
    from axelera.compiler import CompilerConfig

    model_name = Path(f_onnx).stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    if "C2PSA" in model_str:  # YOLO11
        config = CompilerConfig(
            quantization_scheme="per_tensor_min_max",
            ignore_weight_buffers=False,
            resources_used=0.25,
            aipu_cores_used=1,
            multicore_mode="batch",
            output_axm_format=True,
            model_name=model_name,
        )
    else:  # YOLOv8
        config = CompilerConfig(
            tiling_depth=6,
            split_buffer_promotion=True,
            resources_used=0.25,
            aipu_cores_used=1,
            multicore_mode="batch",
            output_axm_format=True,
            model_name=model_name,
        )

    qmodel = compiler.quantize(
        model=f_onnx,
        calibration_dataset=calibration_dataset,
        config=config,
        transform_fn=transform_fn,
    )
    compiler.compile(model=qmodel, config=config, output_dir=export_path)

    axm_name = f"{model_name}.axm"
    axm_src = Path(axm_name)
    axm_dst = export_path / axm_name
    if axm_src.exists():
        axm_src.replace(axm_dst)

    YAML.save(export_path / "metadata.yaml", metadata or {})
    return export_path
