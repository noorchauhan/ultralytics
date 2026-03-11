# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any


def labels_to_box3d(
    labels: list[dict[str, Any]],
    calib: dict[str, Any] | None,
    image_hw: tuple[int, int],
    class_names: Any = None,
) -> list["Box3D"]:
    """Convert dataset label dicts to Box3D for visualization.

    Delegates to Box3D.from_label() for each label.

    Args:
        labels: List of label dicts from Stereo3DDetDataset (letterboxed space, normalized coords).
        calib: Calibration dict (ideally already transformed to the same letterboxed space as the images).
        image_hw: (H, W) of the image we are drawing onto (letterboxed training tensor).
        class_names: Optional list of class names for Box3D.class_label.
    """
    from ultralytics.data.stereo.box3d import Box3D

    if not labels or calib is None:
        return []
    return [b for lab in labels if (b := Box3D.from_label(lab, calib, class_names, image_hw)) is not None]
