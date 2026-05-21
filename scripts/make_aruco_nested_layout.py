#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2

from benchmark_aruco_nested_layouts import (
    LayoutCase,
    ROOT,
    write_gazebo_model,
    write_marker_yaml,
    write_preview,
    write_world,
)


def build_layout(args):
    if args.corner_offset is None:
        corner_offset = args.board_size * 0.5 - args.edge_margin - args.corner_length * 0.5
    else:
        corner_offset = args.corner_offset

    edge_margin = args.board_size * 0.5 - corner_offset - args.corner_length * 0.5
    center_corner_gap = corner_offset - args.corner_length * 0.5 - args.center_length * 0.5
    side_corner_gap = 2.0 * corner_offset - args.corner_length

    if edge_margin < 0.0:
        raise SystemExit("corner marker is outside the board: reduce corner_offset or corner_length")
    if center_corner_gap < args.min_gap:
        raise SystemExit(
            "center/corner gap is too small: reduce center_length/corner_length or increase corner_offset"
        )
    if side_corner_gap < args.min_gap:
        raise SystemExit("corner markers overlap each other: reduce corner_length or increase board_size")

    name = args.case_name
    if name is None:
        name = "corner{:03d}_center{:03d}_edge{:03d}".format(
            int(round(args.corner_length * 1000.0)),
            int(round(args.center_length * 1000.0)),
            int(round(edge_margin * 1000.0)),
        )

    return LayoutCase(
        name=name,
        board_size=args.board_size,
        corner_length=args.corner_length,
        center_length=args.center_length,
        corner_offset=corner_offset,
        edge_margin=edge_margin,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a custom 1 m nested ArUco board YAML, Gazebo model, world, and preview."
    )
    parser.add_argument("--board-size", type=float, default=1.0, help="Landing platform side length in meters")
    parser.add_argument("--corner-length", type=float, default=0.24, help="Length of each corner marker in meters")
    parser.add_argument("--center-length", type=float, default=0.16, help="Length of center marker in meters")
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.10,
        help="Distance from marker outer edge to platform edge, used if --corner-offset is omitted",
    )
    parser.add_argument(
        "--corner-offset",
        type=float,
        default=None,
        help="Absolute x/y center offset for each corner marker, overrides --edge-margin",
    )
    parser.add_argument("--min-gap", type=float, default=0.03, help="Minimum white gap between markers")
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dictionary-id", type=int, default=int(cv2.aruco.DICT_6X6_1000))
    parser.add_argument("--preview-pixels-per-meter", type=int, default=1200)
    parser.add_argument("--output-root", default="/tmp/aruco_custom_layout")
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    layout = build_layout(args)
    model_name = args.model_name or f"aruco_nested_board_{layout.name}"

    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary_id)
    config_path = output_root / "config" / f"{layout.name}.yaml"
    preview_path = output_root / "preview" / f"{layout.name}.png"
    model_root = output_root / "gazebo_models"
    world_path = output_root / "worlds" / f"{layout.name}.world"

    write_marker_yaml(layout, config_path)
    write_preview(layout, dictionary, preview_path, args.preview_pixels_per_meter)
    write_gazebo_model(layout, dictionary, model_name, model_root)
    write_world(model_name, world_path)

    manifest = {
        "case": layout.name,
        "board_size_m": layout.board_size,
        "corner_length_m": layout.corner_length,
        "center_length_m": layout.center_length,
        "corner_offset_m": layout.corner_offset,
        "edge_margin_m": layout.edge_margin,
        "marker_config": str(config_path),
        "preview": str(preview_path),
        "gazebo_model_root": str(model_root),
        "gazebo_world": str(world_path),
        "model_name": model_name,
        "example_sitl_command": (
            "export GAZEBO_MODEL_PATH={model_root}:$GAZEBO_MODEL_PATH && "
            "python3 scripts/benchmark_aruco_landing.py --label {case} "
            "--controller aruco_search_and_detect.py "
            "--detector-type aruco_multi_marker_det_weighted.py "
            "--marker-config {config_path} --world {world_path} --runs 3 --timeout 150"
        ).format(
            model_root=model_root,
            case=layout.name,
            config_path=config_path,
            world_path=world_path,
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"marker config: {config_path}")
    print(f"gazebo model root: {model_root}")
    print(f"gazebo world: {world_path}")
    print(f"preview: {preview_path}")
    print(f"manifest: {manifest_path}")
    print("SITL model path:")
    print(f"  export GAZEBO_MODEL_PATH={model_root}:$GAZEBO_MODEL_PATH")


if __name__ == "__main__":
    main()
