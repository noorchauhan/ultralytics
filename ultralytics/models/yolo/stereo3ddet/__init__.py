# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection module for YOLO.

This module implements stereo-based 3D object detection with CenterNet-style outputs,
including geometric construction, dense alignment, and occlusion handling.

Key Components:
    - Stereo3DDetModel: Main model class for stereo 3D detection
    - Stereo3DDetTrainer: Training logic with stereo-specific augmentation
    - Stereo3DDetValidator: Validation with KITTI AP3D metrics
    - Stereo3DDetPredictor: Inference pipeline for stereo images
"""

from .model import Stereo3DDetModel
from .train import Stereo3DDetTrainer
from .val import Stereo3DDetValidator
from .predict import Stereo3DDetPredictor
from .visualize import plot_stereo_predictions
from .metrics import Stereo3DDetMetrics

from .geometric import (
    GeometricObservations,
    CalibParams,
    GeometricConstruction,
    solve_geometric_batch,
    solve_geometric_single,
    fallback_simple_triangulation,
)

from .dense_align_optimized import (
    DenseAlignment,
    create_dense_alignment_optimized as create_dense_alignment_from_config,
)

from .occlusion import (
    classify_occlusion,
    should_skip_dense_alignment,
)

from .preprocess import (
    preprocess_stereo_batch,
    preprocess_stereo_images,
    compute_letterbox_params,
    decode_and_refine_predictions,
    get_geometric_config,
    get_dense_alignment_config,
    clear_config_cache,
)

from .utils import (
    get_paper_class_mapping,
    filter_and_remap_class_id,
    is_paper_class,
    get_paper_class_names,
)

from .augment import (
    StereoCalibration,
    StereoLabels,
    StereoHSV,
    StereoHFlip,
    StereoScale,
    StereoCrop,
    StereoLetterBox,
)

__all__ = [
    "Stereo3DDetModel",
    "Stereo3DDetTrainer",
    "Stereo3DDetValidator",
    "Stereo3DDetPredictor",
    "Stereo3DDetMetrics",
    "plot_stereo_predictions",
    "GeometricConstruction",
    "GeometricObservations",
    "CalibParams",
    "solve_geometric_batch",
    "solve_geometric_single",
    "fallback_simple_triangulation",
    "DenseAlignment",
    "create_dense_alignment_from_config",
    "classify_occlusion",
    "should_skip_dense_alignment",
    "preprocess_stereo_batch",
    "preprocess_stereo_images",
    "compute_letterbox_params",
    "decode_and_refine_predictions",
    "get_geometric_config",
    "get_dense_alignment_config",
    "clear_config_cache",
    "get_paper_class_mapping",
    "filter_and_remap_class_id",
    "is_paper_class",
    "get_paper_class_names",
    "StereoCalibration",
    "StereoLabels",
    "StereoHSV",
    "StereoHFlip",
    "StereoScale",
    "StereoCrop",
    "StereoLetterBox",
]
