# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER, WINDOWS


class IOSDetectModel(nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model: nn.Module, im: torch.Tensor, mlprogram: bool = True):
        """Initialize the IOSDetectModel class with a YOLO model and example image.

        Args:
            model (nn.Module): The YOLO model to wrap.
            im (torch.Tensor): Example input tensor with shape (B, C, H, W).
            mlprogram (bool): Whether exporting to MLProgram format.
        """
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        self.mlprogram = mlprogram
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],  # broadcast (slower, smaller)
                device=next(model.parameters()).device,
            )

    def forward(self, x: torch.Tensor):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # pad class length to multiple of 80
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)
        return cls, xywh * self.normalize


def _pipeline_coreml(
    model: Any,
    output_shape: tuple,
    metadata: dict,
    args: SimpleNamespace,
    weights_dir: Path | str | None = None,
    prefix: str = "",
):
    """Create CoreML pipeline with NMS for YOLO detection models.

    Args:
        model: CoreML model.
        output_shape (tuple): Output shape tuple from the exporter.
        metadata (dict): Model metadata.
        args: Export arguments with ``iou``, ``conf``, ``agnostic_nms``, ``format`` attributes.
        weights_dir: Weights directory for MLProgram models.
        prefix (str): Prefix for log messages.

    Returns:
        CoreML pipeline model.
    """
    import coremltools as ct

    LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")

    spec = model.get_spec()
    outs = list(iter(spec.description.output))
    if args.format == "mlmodel":  # mlmodel doesn't infer shapes automatically
        outs[0].type.multiArrayType.shape[:] = output_shape[2], output_shape[1] - 4
        outs[1].type.multiArrayType.shape[:] = output_shape[2], 4

    names = metadata["names"]
    nx = spec.description.input[0].type.imageType.width
    ny = spec.description.input[0].type.imageType.height
    nc = outs[0].type.multiArrayType.shape[-1]
    if len(names) != nc:  # Hack fix for MLProgram NMS bug https://github.com/ultralytics/ultralytics/issues/22309
        names = {**names, **{i: str(i) for i in range(len(names), nc)}}

    model = ct.models.MLModel(spec, weights_dir=weights_dir)

    # Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = spec.specificationVersion
    for i in range(len(outs)):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    output_names = ["confidence", "coordinates"]
    for i, name in enumerate(output_names):
        nms_spec.description.output[i].name = name

    for i, out in enumerate(outs):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = out.type.multiArrayType.shape[-1]
        ma_type.shapeRange.sizeRanges[1].upperBound = out.type.multiArrayType.shape[-1]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
    nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
    nms.confidenceOutputFeatureName = output_names[0]
    nms.coordinatesOutputFeatureName = output_names[1]
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = args.iou
    nms.confidenceThreshold = args.conf
    nms.pickTop.perClass = not args.agnostic_nms
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    # Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=output_names,
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = spec.specificationVersion
    pipeline.spec.description.metadata.userDefined.update(
        {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
    )

    # Save the model
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
    model.input_description["image"] = "Input image"
    model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
    LOGGER.info(f"{prefix} pipeline success")
    return model


def torch2coreml(
    model: nn.Module,
    im: torch.Tensor,
    file: Path | str,
    args: SimpleNamespace,
    output_shape: tuple,
    metadata: dict | None = None,
    imgsz: list | None = None,
    prefix: str = "",
):
    """Export a PyTorch model to CoreML ``.mlpackage`` or ``.mlmodel`` format.

    Args:
        model (nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor.
        file (Path | str): Source model path used to derive the output path.
        args (SimpleNamespace): Export arguments (``format``, ``batch``, ``dynamic``, ``nms``, ``int8``, ``half``).
        output_shape (tuple): Model output shape used by the NMS pipeline.
        metadata (dict | None): Metadata to embed in the CoreML model.
        imgsz (list | None): Image size ``[h, w]``.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported CoreML model file/directory.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_1_11

    mlmodel = args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
    check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
    assert TORCH_1_11, "CoreML export requires torch>=1.11"
    if args.batch > 1:
        assert args.dynamic, (
            "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. "
            "Please retry at 'dynamic=True'."
        )
    if args.dynamic:
        assert not args.nms, (
            "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
        )
        assert model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."

    file = Path(file)
    f = file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
    if f.is_dir():
        shutil.rmtree(f)

    imgsz = imgsz or [640, 640]
    classifier_config = None
    if model.task == "classify":
        classifier_config = ct.ClassifierConfig(list(model.names.values()))
        export_model = model
    elif model.task == "detect":
        export_model = IOSDetectModel(model, im, mlprogram=not mlmodel) if args.nms else model
    else:
        if args.nms:
            LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo26n.pt'.")
        export_model = model

    ts = torch.jit.trace(export_model.eval(), im, strict=False)  # TorchScript model

    if args.dynamic:
        input_shape = ct.Shape(
            shape=(
                ct.RangeDim(lower_bound=1, upper_bound=args.batch, default=1),
                im.shape[1],
                ct.RangeDim(lower_bound=32, upper_bound=imgsz[0] * 2, default=imgsz[0]),
                ct.RangeDim(lower_bound=32, upper_bound=imgsz[1] * 2, default=imgsz[1]),
            )
        )
        inputs = [ct.TensorType("image", shape=input_shape)]
    else:
        inputs = [ct.ImageType("image", shape=im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

    ct_model = ct.convert(
        ts,
        inputs=inputs,
        classifier_config=classifier_config,
        convert_to="neuralnetwork" if mlmodel else "mlprogram",
    )
    bits, mode = (8, "kmeans") if args.int8 else (16, "linear") if args.half else (32, None)
    if bits < 32:
        if "kmeans" in mode:
            check_requirements("scikit-learn")
        if mlmodel:
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        elif bits == 8:  # mlprogram already quantized to FP16
            import coremltools.optimize.coreml as cto

            op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
            config = cto.OptimizationConfig(global_config=op_config)
            ct_model = cto.palettize_weights(ct_model, config=config)
    if args.nms and model.task == "detect":
        ct_model = _pipeline_coreml(
            ct_model,
            output_shape=output_shape,
            metadata=metadata or {},
            args=args,
            weights_dir=None if mlmodel else ct_model.weights_dir,
            prefix=prefix,
        )

    m = dict(metadata or {})  # copy to avoid mutating original
    ct_model.short_description = m.pop("description", "")
    ct_model.author = m.pop("author", "")
    ct_model.license = m.pop("license", "")
    ct_model.version = m.pop("version", "")
    ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
    if model.task == "classify":
        ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

    try:
        ct_model.save(str(f))  # save *.mlpackage
    except Exception as e:
        LOGGER.warning(
            f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
            f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
        )
        f = f.with_suffix(".mlmodel")
        ct_model.save(str(f))
    return f
