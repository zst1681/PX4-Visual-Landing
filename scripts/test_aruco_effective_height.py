#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from benchmark_aruco_nested_layouts import (
    CENTER_MARKER_ID,
    ALL_MARKER_IDS,
    CORNER_MARKER_IDS,
    LayoutCase,
    ROOT,
    Scenario,
    create_board,
    create_detector_params,
    evaluate_detection,
    load_camera_params,
    make_board_texture,
    render_scenario,
    write_preview,
)


def build_layout_from_args(args):
    if args.marker_config:
        return build_layout_from_marker_config(Path(args.marker_config), args.board_size)

    if args.corner_offset is None:
        corner_offset = args.board_size * 0.5 - args.edge_margin - args.corner_length * 0.5
    else:
        corner_offset = args.corner_offset
    edge_margin = args.board_size * 0.5 - corner_offset - args.corner_length * 0.5

    return LayoutCase(
        name=args.case_name,
        board_size=args.board_size,
        corner_length=args.corner_length,
        center_length=args.center_length,
        corner_offset=corner_offset,
        edge_margin=edge_margin,
    )


def build_layout_from_marker_config(path, board_size):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    configs = data.get("marker_configs", [])
    if not configs:
        raise SystemExit(f"no marker_configs found in {path}")

    center_configs = [
        config
        for config in configs
        if config.get("role") == "center"
        or int(config.get("id", -1)) == CENTER_MARKER_ID
        or (
            abs(float(config.get("offset_x", 0.0))) < 1e-9
            and abs(float(config.get("offset_y", 0.0))) < 1e-9
        )
    ]
    corner_configs = [
        config
        for config in configs
        if int(config.get("id", -1)) in CORNER_MARKER_IDS or config.get("role") == "corner"
    ]
    if not center_configs or len(corner_configs) < 4:
        raise SystemExit("this height tester expects the five-marker symmetric nested-board YAML")

    center = center_configs[0]
    corner_lengths = {round(float(config["length"]), 6) for config in corner_configs}
    corner_offsets = {
        round(abs(float(config.get("offset_x", 0.0))), 6)
        for config in corner_configs
    }.union(
        {
            round(abs(float(config.get("offset_y", 0.0))), 6)
            for config in corner_configs
        }
    )
    corner_offsets.discard(0.0)
    if len(corner_lengths) != 1 or len(corner_offsets) != 1:
        raise SystemExit("non-uniform corner marker layouts should be tested with the full layout benchmark")

    corner_length = next(iter(corner_lengths))
    corner_offset = next(iter(corner_offsets))
    edge_margin = board_size * 0.5 - corner_offset - corner_length * 0.5
    return LayoutCase(
        name=path.stem,
        board_size=board_size,
        corner_length=corner_length,
        center_length=float(center["length"]),
        corner_offset=corner_offset,
        edge_margin=edge_margin,
    )


def make_height_values(height_min, height_max, height_step):
    values = []
    current = height_min
    while current <= height_max + 1e-9:
        values.append(round(current, 6))
        current += height_step
    return values


def make_scenario_for_height(height, args, rng):
    radius = args.lateral_max * math.sqrt(rng.random())
    theta = rng.uniform(0.0, math.tau)
    if args.yaw_mode == "random":
        yaw_rad = rng.uniform(-math.pi, math.pi)
    else:
        yaw_rad = math.radians(args.yaw_deg)

    tilt_rad = math.radians(args.tilt_deg)
    return Scenario(
        phase="height_sweep",
        camera_x=radius * math.cos(theta),
        camera_y=radius * math.sin(theta),
        height=height,
        yaw_rad=yaw_rad,
        tilt_x_rad=rng.uniform(-tilt_rad, tilt_rad),
        tilt_y_rad=rng.uniform(-tilt_rad, tilt_rad),
        contrast=rng.uniform(args.contrast_min, args.contrast_max),
        brightness=rng.uniform(-args.brightness_abs, args.brightness_abs),
        blur_sigma=rng.uniform(args.blur_min, args.blur_max),
        noise_sigma=rng.uniform(args.noise_min, args.noise_max),
        noise_seed=rng.randrange(0, 2**32),
    )


def evaluate_height(height, layout, texture, dictionary, detector_params, camera_matrix, dist_coeffs, board, args, rng):
    counts = {
        "any": 0,
        "center": 0,
        "corner_pair": 0,
        "board_pose": 0,
        "nested": 0,
        "all": 0,
    }
    per_id_hits = {marker_id: 0 for marker_id in sorted(ALL_MARKER_IDS)}
    detected_marker_total = 0
    reprojection_errors = []

    for _sample_index in range(args.samples_per_height):
        scenario = make_scenario_for_height(height, args, rng)
        frame = render_scenario(texture, layout, scenario, camera_matrix, (args.image_width, args.image_height))
        result = evaluate_detection(
            frame,
            layout,
            dictionary,
            detector_params,
            camera_matrix,
            dist_coeffs,
            board,
            args.max_board_reprojection_error,
        )
        detected_ids = result["detected_ids"]
        counts["any"] += 1 if detected_ids else 0
        counts["center"] += 1 if result["center_hit"] else 0
        counts["corner_pair"] += 1 if result["corner_count"] >= 2 else 0
        counts["board_pose"] += 1 if result["board_pose_hit"] else 0
        counts["nested"] += 1 if result["nested_hit"] else 0
        counts["all"] += 1 if result["all_hit"] else 0
        detected_marker_total += len(detected_ids)
        for marker_id in detected_ids:
            per_id_hits[marker_id] += 1
        if result["board_reprojection_error_px"] is not None:
            reprojection_errors.append(result["board_reprojection_error_px"])

    total = args.samples_per_height
    row = {
        "height_m": height,
        "samples": total,
        "nested_rate": counts["nested"] / total,
        "board_pose_rate": counts["board_pose"] / total,
        "center_rate": counts["center"] / total,
        "any_marker_rate": counts["any"] / total,
        "corner_pair_rate": counts["corner_pair"] / total,
        "all_markers_rate": counts["all"] / total,
        "mean_detected_markers": detected_marker_total / total,
        "board_reprojection_error_mean_px": (
            float(np.mean(reprojection_errors)) if reprojection_errors else None
        ),
        "board_reprojection_error_p95_px": (
            float(np.percentile(reprojection_errors, 95)) if reprojection_errors else None
        ),
    }
    for marker_id in sorted(ALL_MARKER_IDS):
        row[f"marker{marker_id}_rate"] = per_id_hits[marker_id] / total
    return row


def effective_ranges(rows, key, threshold):
    ranges = []
    start = None
    previous_height = None
    for row in rows:
        passed = row[key] >= threshold
        if passed and start is None:
            start = row["height_m"]
        if not passed and start is not None:
            ranges.append([start, previous_height])
            start = None
        previous_height = row["height_m"]
    if start is not None:
        ranges.append([start, previous_height])
    return ranges


def write_height_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "height_m",
        "samples",
        "nested_rate",
        "board_pose_rate",
        "center_rate",
        "any_marker_rate",
        "corner_pair_rate",
        "all_markers_rate",
        "mean_detected_markers",
        "marker31_rate",
        "marker32_rate",
        "marker33_rate",
        "marker34_rate",
        "marker35_rate",
        "board_reprojection_error_mean_px",
        "board_reprojection_error_p95_px",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Sweep camera height and estimate the effective detection height range for a nested ArUco board."
    )
    parser.add_argument("--marker-config", default=None, help="Generated marker YAML; overrides size args")
    parser.add_argument("--case-name", default="custom_height_test")
    parser.add_argument("--board-size", type=float, default=1.0)
    parser.add_argument("--corner-length", type=float, default=0.24)
    parser.add_argument("--center-length", type=float, default=0.16)
    parser.add_argument("--edge-margin", type=float, default=0.10)
    parser.add_argument("--corner-offset", type=float, default=None)
    parser.add_argument("--height-min", type=float, default=0.3)
    parser.add_argument("--height-max", type=float, default=5.0)
    parser.add_argument("--height-step", type=float, default=0.1)
    parser.add_argument("--samples-per-height", type=int, default=80)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--lateral-max", type=float, default=0.6)
    parser.add_argument("--tilt-deg", type=float, default=8.0)
    parser.add_argument("--yaw-mode", choices=["random", "fixed"], default="random")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--blur-min", type=float, default=0.0)
    parser.add_argument("--blur-max", type=float, default=1.0)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=5.0)
    parser.add_argument("--contrast-min", type=float, default=0.75)
    parser.add_argument("--contrast-max", type=float, default=1.15)
    parser.add_argument("--brightness-abs", type=float, default=15.0)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--texture-pixels-per-meter", type=int, default=1800)
    parser.add_argument(
        "--camera-param-path",
        default=str(ROOT / "config" / "camera_monocular_1280x720.yaml"),
    )
    parser.add_argument("--dictionary-id", type=int, default=int(cv2.aruco.DICT_6X6_1000))
    parser.add_argument("--max-board-reprojection-error", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-root", default="/tmp/aruco_height_test")
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layout = build_layout_from_args(args)
    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary_id)
    detector_params = create_detector_params()
    camera_matrix, dist_coeffs = load_camera_params(args.camera_param_path)
    texture = make_board_texture(layout, dictionary, args.texture_pixels_per_meter)
    board = create_board(layout, dictionary)
    rng = random.Random(args.seed)

    preview_path = output_root / f"{layout.name}_preview.png"
    write_preview(layout, dictionary, preview_path, 1200)

    rows = []
    for height in make_height_values(args.height_min, args.height_max, args.height_step):
        row = evaluate_height(
            height,
            layout,
            texture,
            dictionary,
            detector_params,
            camera_matrix,
            dist_coeffs,
            board,
            args,
            rng,
        )
        rows.append(row)
        print(
            "height={height_m:.2f}m nested={nested_rate:.3f} board_pose={board_pose_rate:.3f} center={center_rate:.3f}".format(
                **row
            ),
            flush=True,
        )

    csv_path = output_root / f"{layout.name}_height_results.csv"
    write_height_csv(rows, csv_path)
    summary = {
        "case": layout.name,
        "board_size_m": layout.board_size,
        "corner_length_m": layout.corner_length,
        "center_length_m": layout.center_length,
        "corner_offset_m": layout.corner_offset,
        "edge_margin_m": layout.edge_margin,
        "samples_per_height": args.samples_per_height,
        "success_threshold": args.success_threshold,
        "nested_effective_ranges_m": effective_ranges(rows, "nested_rate", args.success_threshold),
        "board_pose_effective_ranges_m": effective_ranges(rows, "board_pose_rate", args.success_threshold),
        "center_effective_ranges_m": effective_ranges(rows, "center_rate", args.success_threshold),
        "csv": str(csv_path),
        "preview": str(preview_path),
    }
    summary_path = output_root / f"{layout.name}_height_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"height results: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
