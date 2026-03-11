# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Utility constants for stereo 3D detection."""

from __future__ import annotations

# Paper class names (Car, Pedestrian, Cyclist) — default for KITTI
PAPER_CLASS_NAMES: dict[int, str] = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}


def get_paper_class_names() -> dict[int, str]:
    """Get default KITTI paper class names mapping.

    Returns:
        dict[int, str]: Mapping from class ID to class name.
    """
    return PAPER_CLASS_NAMES.copy()
