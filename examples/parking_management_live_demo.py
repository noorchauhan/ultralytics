# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

from ultralytics import solutions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for live parking management demo."""
    parser = argparse.ArgumentParser(description="Ultralytics Parking Management live visualizer demo.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n_openvino_model",
        help="OpenVINO model path (directory or .xml).",
    )
    parser.add_argument("--json-file", type=str, required=True, help="Parking regions JSON path.")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g. 0) or video file/stream path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="intel:cpu",
        help="Initial device: intel:cpu/intel:gpu/intel:npu.",
    )
    parser.add_argument("--no-loop", action="store_true", help="Disable looping when source reaches end.")
    parser.add_argument("--no-hotkeys", action="store_true", help="Disable runtime hotkey device switching.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable on-screen performance overlay.")
    return parser.parse_args()


def parse_source(source_arg: str) -> str | int:
    """Convert numeric source strings to camera index integer."""
    return int(source_arg) if source_arg.isdigit() else source_arg


def main() -> None:
    """Run the parking management live visualizer demo."""
    args = parse_args()
    source = parse_source(args.source)
    manager = solutions.ParkingManagement(model=args.model, json_file=args.json_file, show=True, device=args.device)
    manager.visualize(
        source=source,
        loop=not args.no_loop,
        allow_device_hotkeys=not args.no_hotkeys,
        show_perf_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
