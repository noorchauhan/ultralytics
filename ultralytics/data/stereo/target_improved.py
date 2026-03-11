
"""Ground truth target generator for Stereo 3D detection with YOLO backbone.

This module implements the target generation for the 7-branch stereo 3D detection head
following the paper "Stereo CenterNet based 3D Object Detection for Autonomous Driving".

Note: This generator creates targets for 3D detection branches. The 2D detection
(heatmap, offset, bbox_size) has been replaced by YOLO-style TaskAlignedAssigner
with direct bounding box regression.

The 4 branches are:
Stereo 2D Association (2 branches):
    1. lr_distance: Left-Right center distance [1, H, W]
    2. right_width: Right box width [1, H, W]

3D Components (2 branches):
    3. dimensions: 3D dimension offsets (ΔH, ΔW, ΔL) [3, H, W]
    4. orientation: Multi-Bin orientation encoding [8, H, W]

References:
    Paper: https://arxiv.org/abs/2103.11071
    Figure 4: Shows vertex ordering 0,1,2,3 at bottom of 3D box
    Figure 5: Shows geometric relationship between 2D and 3D
"""

from __future__ import annotations

import math
from typing import Any
import torch


class TargetGenerator:
    """Generate ground truth targets for stereo 3D detection with YOLO backbone.

    Creates regression targets for 4 branches (stereo association + 3D components).

    Note: This generator is used with YOLO-based detection. Heatmap-based 2D detection
    has been replaced by TaskAlignedAssigner with direct bounding box regression.
    """

    def __init__(
        self,
        mean_dims,
        std_dims,
        output_size: tuple[int, int] = (96, 320),  # Example: (96, 320) for 384×1280 input with 4x downsampling
        num_classes: int = 3,
        class_names: dict[int, str] | None = None,
    ):
        """Initialize target generator.

        Args:
            output_size: Output feature map size (H, W). Determined dynamically from model architecture.
                         The actual downsampling factor depends on the model config (e.g., P3 = 8x, P4 = 16x).
            num_classes: Number of object classes.
            mean_dims: Mean dimensions per class [L, W, H] in meters.
                       Can be integer keys (class_id) or string keys (class_name).
            std_dims: Standard deviation of dimensions per class [L, W, H] in meters.
                      Can be integer keys (class_id) or string keys (class_name).
                      Used for normalized offset prediction: (dim - mean) / std.
            class_names: Mapping from class_id to class_name (e.g., {0: "Car", 1: "Pedestrian", ...}).
                       If None, will use generic names ("Class 0", "Class 1", ...).
        """
        self.output_h, self.output_w = output_size
        self.num_classes = num_classes

        # Class name mapping (dataset-specific)
        # This maps class_id -> class_name
        if class_names is not None:
            self.class_names_map = class_names
        else:
            # Use generic names if not provided
            self.class_names_map = {i: f"Class {i}" for i in range(num_classes)}

        # Build reverse mapping (class_name -> class_id) for backward compatibility
        self.class_name_to_id = {v: k for k, v in self.class_names_map.items()}

        # Handle both integer keys (class_id) and string keys (class_name) for mean_dims
        assert mean_dims is not None, "mean_dims must be provided"
        # Check if mean_dims uses integer keys (class IDs) or string keys (class names)
        self.mean_dims = mean_dims

        # Handle both integer keys (class_id) and string keys (class_name) for std_dims
        assert std_dims is not None, "std_dims must be provided"
        # Check if std_dims uses integer keys (class IDs) or string keys (class names)
        self.std_dims = std_dims
        
    def generate_targets(
        self,
        labels: list[dict],
        input_size: tuple[int, int] = (384, 1280),
        calib: list[dict[str, float]] | None = None,
        original_size: list[tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate ground truth targets for stereo 3D detection (4 branches).

        Args:
            labels: List of label dictionaries from dataset. Each label should have:
                - class_id: int
                - left_box: dict with center_x, center_y, width, height (normalized to letterboxed input image)
                - right_box: dict with center_x, width (normalized to letterboxed input image)
                - dimensions: dict with length, width, height (meters)
                - alpha: observation angle in radians
                - location_3d (optional): dict with x, y, z (meters) - 3D center position
                  If not provided, computed from stereo disparity
            input_size: Input image size (H, W) after preprocessing (letterbox).
            calib: Camera calibration parameters dict with fx, fy, cx, cy, baseline.
                   Already transformed to letterboxed space by the dataset.
                   If None, uses default KITTI values.
            original_size: Original image size (H, W) before preprocessing.
                   Used for reference, not for coordinate transformation.
                   If None, uses KITTI default (375, 1242).

        Returns:
            Dictionary with 4 branch targets for stereo association and 3D detection,
            each [num_classes or channels, H_out, W_out]
            where H_out, W_out are determined by the model's output size (architecture-agnostic).

            Note: 2D detection targets (heatmap, offset, bbox_size) are NOT generated.
            These are replaced by YOLO-style TaskAlignedAssigner with direct bbox regression.
        """
        input_h, input_w = input_size
        
        # Scale calibration parameters to match preprocessed input size
        # Calibration parameters are typically in original image space (e.g., KITTI 1242x375)
        # We need to scale them to match the preprocessed input size
        # Scale factors from preprocessed input to feature map output
        # Initialize targets (7 branches for 3D detection with YOLO backbone)
        # Note: heatmap, offset, bbox_size are not generated - replaced by YOLO TaskAlignedAssigner
        targets = {
            "lr_distance": torch.zeros(1, self.output_h, self.output_w),
            "right_width": torch.zeros(1, self.output_h, self.output_w),
            "dimensions": torch.zeros(3, self.output_h, self.output_w),
            "orientation": torch.zeros(8, self.output_h, self.output_w),
        }

        # Process each object
        for label, calib, original_size in zip(labels, calib, original_size):
            self._process_single_label(
                label, targets, input_h, input_w, calib, original_size
            )
        return targets

    def _process_single_label(
        self,
        label: dict[str, Any],
        targets: dict[str, torch.Tensor],
        input_h: int,
        input_w: int,
        calib: dict[str, float],
        original_size: tuple[int, int],
    ) -> None:
        """Process a single label and update targets.

        Args:
            label: Single label dictionary.
            targets: Target tensors to update.
            input_h, input_w: Input image dimensions.
            calib: Camera calibration parameters.
            original_size: Original image size.
        """
        class_id = label["class_id"]
        left_box = label["left_box"]
        right_box = label["right_box"]
        dimensions = label["dimensions"]
        
        rotation_y = label["rotation_y"]
        location_3d = label["location_3d"]
        x_3d = location_3d["x"]
        z_3d = location_3d["z"]
        ray_angle = math.atan2(x_3d, z_3d)
        alpha = rotation_y - ray_angle
        

        fx = calib["fx"]
        fy = calib["fy"]
        cx = calib["cx"]
        cy = calib["cy"]
        baseline = calib["baseline"]

        # Calibration is already transformed to letterboxed space in the dataset,
        # so we don't need to scale it again here. The calibration parameters
        # (fx, fy, cx, cy) are already in the letterboxed input image space.
        # orig_h, orig_w = original_size  # Not needed for calibration scaling anymore

        # output scale factors
        scale_h = self.output_h / input_h
        scale_w = self.output_w / input_w

        # Get center coordinates in input image (pixels)
        center_x = left_box["center_x"] * input_w
        center_y = left_box["center_y"] * input_h

        # Scale to output size
        center_x_out = center_x * scale_w
        center_y_out = center_y * scale_h

        # Integer center (for sparse target assignment)
        center_x_int = int(center_x_out)
        center_y_int = int(center_y_out)

        # Skip if center is outside output bounds
        if not (0 <= center_x_int < self.output_w and 0 <= center_y_int < self.output_h):
            return

        # ============================================================
        # Stereo 2D Association (2 branches for YOLO-based detection)
        # ============================================================

        # 1. LR distance (left-right center distance for stereo association)
        # Paper Equation 4: distance between left and right object centers
        right_center_x = right_box["center_x"] * input_w
        lr_dist = center_x - right_center_x  # Disparity in pixels
        lr_distance_stored = lr_dist * scale_w
        targets["lr_distance"][0, center_y_int, center_x_int] = lr_distance_stored

        # 2. Right width (in feature map units, same scale as lr_distance)
        # Store raw value in feature map units for consistent magnitude with lr_distance
        right_w = right_box["width"] * input_w * scale_w
        targets["right_width"][0, center_y_int, center_x_int] = right_w

        # ============================================================
        # 3D Components (5 branches)
        # ============================================================

        # 6. Dimensions (normalized offset from class mean)
        # Using normalized offset: (dim - mean) / std for more stable training
        # mean_dims and std_dims are both [L, W, H] in meters
        mean_dim = self.mean_dims.get(class_id, [1.0, 1.0, 1.0])
        std_dim = self.std_dims.get(class_id, [0.2, 0.2, 0.5])
        
        # Compute normalized offsets: (dim - mean) / std
        # decoder expects [ΔH, ΔW, ΔL] order, so we need to reorder from [L, W, H]
        # std_dim is [L, W, H], so:
        #   std_dim[0] = std_L, std_dim[1] = std_W, std_dim[2] = std_H
        #   mean_dim[0] = mean_L, mean_dim[1] = mean_W, mean_dim[2] = mean_H
        dim_offset = [
            (dimensions["height"] - mean_dim[2]) / std_dim[2],   # channel 0 = (H - mean_H) / std_H
            (dimensions["width"] - mean_dim[1]) / std_dim[1],    # channel 1 = (W - mean_W) / std_W
            (dimensions["length"] - mean_dim[0]) / std_dim[0],   # channel 2 = (L - mean_L) / std_L
        ]
        targets["dimensions"][:, center_y_int, center_x_int] = torch.tensor(dim_offset)

        # 7. Orientation (Multi-Bin encoding - Paper Section 3.1)
        targets["orientation"][:, center_y_int, center_x_int] = encode_orientation(alpha)


def compute_dimension_offset(
    dims: tuple[float, float, float],
    class_id: int,
    mean_dims: dict,
    std_dims: dict,
) -> torch.Tensor:
    """Compute normalized dimension offset [ΔH, ΔW, ΔL] for 3D detection.

    The offset is computed as (dim - mean) / std for each dimension,
    following the paper's approach for stable training.

    Args:
        dims: Object dimensions as (length, width, height) in meters.
        class_id: Integer class ID for looking up mean/std values.
        mean_dims: Dict mapping class_id -> [mean_L, mean_W, mean_H].
        std_dims: Dict mapping class_id -> [std_L, std_W, std_H].

    Returns:
        Tensor of shape [3] with normalized offsets [ΔH, ΔW, ΔL].
    """
    mean_dim = mean_dims.get(class_id, [1.0, 1.0, 1.0])
    std_dim = std_dims.get(class_id, [0.2, 0.2, 0.5])
    return torch.tensor(
        [
            (dims[2] - mean_dim[2]) / std_dim[2],  # height
            (dims[1] - mean_dim[1]) / std_dim[1],  # width
            (dims[0] - mean_dim[0]) / std_dim[0],  # length
        ],
        dtype=torch.float32,
    )


def encode_orientation(alpha: float) -> torch.Tensor:
    """Encode orientation angle into Multi-Bin format.

    Multi-Bin encoding (2 bins) as described in Paper Section 3.1:
    - Bin 0: α ∈ [-π, 0], center = -π/2
    - Bin 1: α ∈ [0, π], center = +π/2

    The residual angle is encoded as (sin, cos) for each bin.

    Output format: [conf_0, conf_1, sin_0, cos_0, sin_1, cos_1, pad, pad]

    Args:
        alpha: Observation angle in radians, range [-π, π].

    Returns:
        Orientation encoding tensor of shape [8].
    """
    # Normalize alpha to [-π, π]
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))

    # Determine bin based on angle sign
    if alpha < 0:
        bin_idx = 0
        bin_center = -math.pi / 2
    else:
        bin_idx = 1
        bin_center = math.pi / 2

    # Compute residual angle within bin
    residual = alpha - bin_center

    # Initialize encoding
    encoding = torch.zeros(8, dtype=torch.float32)

    # Set bin confidence (one-hot encoding)
    encoding[0] = 1.0 if bin_idx == 0 else 0.0
    encoding[1] = 1.0 if bin_idx == 1 else 0.0

    # Set sin/cos for the active bin only
    if bin_idx == 0:
        encoding[2] = math.sin(residual)
        encoding[3] = math.cos(residual)
    else:
        encoding[4] = math.sin(residual)
        encoding[5] = math.cos(residual)

    # Channels 6-7 are padding (remain 0)
    return encoding


