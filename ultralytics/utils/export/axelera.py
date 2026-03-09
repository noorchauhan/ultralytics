# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import YAML


def torch2axelera(
    compiler,
    compiler_config,
    extract_ultralytics_metadata,
    model,
    file: str | Path,
    calibration_dataset,
    transform_fn,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Convert a YOLO model to Axelera format.

    Args:
        compiler: Axelera compiler module.
        compiler_config: Axelera `CompilerConfig` class used to create compiler settings.
        extract_ultralytics_metadata: Axelera metadata extraction utility.
        model: Source YOLO model for quantization.
        file (str | Path): Source model file path used to derive output names.
        calibration_dataset: Calibration dataloader for quantization.
        transform_fn: Calibration preprocessing transform function.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Returns:
        (Path): Path to exported Axelera model directory.
    """

    file = Path(file)
    model_name = file.stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    axelera_model_metadata = extract_ultralytics_metadata(model)
    config = compiler_config(
        model_metadata=axelera_model_metadata,
        model_name=model_name,
        resources_used=0.25,
        aipu_cores_used=1,
        multicore_mode="batch",
        output_axm_format=True,
    )
    qmodel = compiler.quantize(
        model=model,
        calibration_dataset=calibration_dataset,
        config=config,
        transform_fn=transform_fn,
    )
    compiler.compile(model=qmodel, config=config, output_dir=export_path)

    for artifact in [f"{model_name}.axm", "compiler_config_final.toml"]:
        artifact_path = Path(artifact)
        if artifact_path.exists():
            artifact_path.replace(export_path / artifact_path.name)

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)
    return export_path
