# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER, YAML


def _axelera_compiler_config(compiler_config, model_name: str, is_yolo11: bool):
    """Build an Axelera compiler configuration for YOLOv8/YOLO11 export."""
    if is_yolo11:
        return compiler_config(
            quantization_scheme="per_tensor_min_max",
            ignore_weight_buffers=False,
            resources_used=0.25,
            aipu_cores_used=1,
            multicore_mode="batch",
            output_axm_format=True,
            model_name=model_name,
        )
    return compiler_config(
        tiling_depth=6,
        split_buffer_promotion=True,
        resources_used=0.25,
        aipu_cores_used=1,
        multicore_mode="batch",
        output_axm_format=True,
        model_name=model_name,
    )


def onnx2axelera(
    compiler,
    compiler_config,
    model,
    onnx_path: str | Path,
    calibration_dataset,
    transform_fn,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Convert an ONNX model to Axelera format.

    Args:
        compiler: Axelera compiler module.
        compiler_config: Axelera `CompilerConfig` class used to create compiler settings.
        model: Source YOLO model used to select YOLOv8 vs YOLO11 configuration.
        onnx_path (str | Path): Input ONNX model path.
        calibration_dataset: Calibration dataloader for quantization.
        transform_fn: Calibration preprocessing transform function.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Returns:
        (Path): Path to exported Axelera model directory.
    """
    LOGGER.info(f"\n{prefix} starting export with axelera-voyager-sdk {compiler.__version__}...")

    onnx_path = Path(onnx_path)
    model_name = onnx_path.stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    config = _axelera_compiler_config(
        compiler_config=compiler_config,
        model_name=model_name,
        is_yolo11="C2PSA" in model.__str__(),
    )
    qmodel = compiler.quantize(
        model=str(onnx_path),
        calibration_dataset=calibration_dataset,
        config=config,
        transform_fn=transform_fn,
    )
    compiler.compile(model=qmodel, config=config, output_dir=export_path)

    axm_name = f"{model_name}.axm"
    axm_src = Path(axm_name)
    if axm_src.exists():
        axm_src.replace(export_path / axm_name)

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)
    return export_path
