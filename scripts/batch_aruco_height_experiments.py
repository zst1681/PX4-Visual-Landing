#!/usr/bin/env python3

import argparse
import csv
import json
import random
import shutil
import statistics
from pathlib import Path

import cv2

from benchmark_aruco_nested_layouts import (
    ALL_MARKER_IDS,
    create_board,
    create_detector_params,
    load_camera_params,
    make_board_texture,
    write_preview,
)
from test_aruco_effective_height import (
    build_layout_from_marker_config,
    effective_ranges,
    evaluate_height,
    make_height_values,
    write_height_csv,
)


ROOT = Path(__file__).resolve().parents[1]
RATE_FIELDS = [
    "nested_rate",
    "board_pose_rate",
    "center_rate",
    "any_marker_rate",
    "corner_pair_rate",
    "all_markers_rate",
    "mean_detected_markers",
    "board_reprojection_error_mean_px",
    "board_reprojection_error_p95_px",
]
MARKER_FIELDS = [f"marker{marker_id}_rate" for marker_id in sorted(ALL_MARKER_IDS)]
NUMERIC_FIELDS = RATE_FIELDS + MARKER_FIELDS


def mm(value):
    return int(round(float(value) * 1000.0))


def format_dim_name(case_id, layout):
    return (
        f"case{case_id:02d}_"
        f"corner{mm(layout.corner_length):03d}mm_"
        f"center{mm(layout.center_length):03d}mm_"
        f"offset{mm(layout.corner_offset):03d}mm_"
        f"edge{mm(layout.edge_margin):03d}mm"
    )


def read_manifest(case_dir):
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def discover_case(case_id, source_template, board_size):
    case_dir = Path(source_template.format(case=case_id))
    config_path = case_dir / "config" / f"{case_id}.yaml"
    preview_path = case_dir / "preview" / f"{case_id}.png"
    manifest = read_manifest(case_dir)
    if manifest.get("marker_config"):
        config_path = Path(manifest["marker_config"])
    if manifest.get("preview"):
        preview_path = Path(manifest["preview"])
    if not config_path.exists():
        raise FileNotFoundError(f"missing marker config for case {case_id}: {config_path}")
    if not preview_path.exists():
        raise FileNotFoundError(f"missing preview image for case {case_id}: {preview_path}")

    layout = build_layout_from_marker_config(config_path, board_size)
    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "config_path": config_path,
        "preview_path": preview_path,
        "manifest": manifest,
        "layout": layout,
        "dim_name": format_dim_name(case_id, layout),
    }


def copy_preview(case, preview_root):
    preview_root.mkdir(parents=True, exist_ok=True)
    target_path = preview_root / f"{case['dim_name']}.png"
    shutil.copy2(case["preview_path"], target_path)
    return target_path


def add_metadata(row, case, repeat):
    layout = case["layout"]
    enriched = {
        "case_id": case["case_id"],
        "case_name": layout.name,
        "dimension_name": case["dim_name"],
        "repeat": repeat,
        "corner_length_m": layout.corner_length,
        "center_length_m": layout.center_length,
        "corner_offset_m": layout.corner_offset,
        "edge_margin_m": layout.edge_margin,
    }
    enriched.update(row)
    return enriched


def write_raw_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "case_id",
        "case_name",
        "dimension_name",
        "repeat",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        "height_m",
        "samples",
        *NUMERIC_FIELDS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def stdev(values):
    clean = [value for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else 0.0 if len(clean) == 1 else None


def aggregate_by_case_height(raw_rows):
    grouped = {}
    for row in raw_rows:
        key = (row["case_id"], row["height_m"])
        grouped.setdefault(key, []).append(row)

    averaged = []
    for key in sorted(grouped):
        rows = grouped[key]
        first = rows[0]
        out = {
            "case_id": first["case_id"],
            "case_name": first["case_name"],
            "dimension_name": first["dimension_name"],
            "corner_length_m": first["corner_length_m"],
            "center_length_m": first["center_length_m"],
            "corner_offset_m": first["corner_offset_m"],
            "edge_margin_m": first["edge_margin_m"],
            "height_m": first["height_m"],
            "repeat_count": len(rows),
            "samples_per_height": first["samples"],
        }
        for field in NUMERIC_FIELDS:
            values = [row.get(field) for row in rows]
            out[f"{field}_mean"] = mean(values)
            out[f"{field}_std"] = stdev(values)
        averaged.append(out)
    return averaged


def write_averaged_by_height(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "case_id",
        "case_name",
        "dimension_name",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        "height_m",
        "repeat_count",
        "samples_per_height",
    ]
    for field in NUMERIC_FIELDS:
        fieldnames.extend([f"{field}_mean", f"{field}_std"])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def range_bounds(ranges):
    if not ranges:
        return None, None
    return min(item[0] for item in ranges), max(item[1] for item in ranges)


def aggregate_by_case(averaged_rows, args):
    grouped = {}
    for row in averaged_rows:
        grouped.setdefault(row["case_id"], []).append(row)

    summary = []
    for case_id in sorted(grouped):
        rows = sorted(grouped[case_id], key=lambda row: row["height_m"])
        first = rows[0]
        nested_rows = [
            {"height_m": row["height_m"], "nested_rate": row["nested_rate_mean"]}
            for row in rows
        ]
        board_rows = [
            {"height_m": row["height_m"], "board_pose_rate": row["board_pose_rate_mean"]}
            for row in rows
        ]
        center_rows = [
            {"height_m": row["height_m"], "center_rate": row["center_rate_mean"]}
            for row in rows
        ]

        nested_ranges = effective_ranges(nested_rows, "nested_rate", args.success_threshold)
        board_ranges = effective_ranges(board_rows, "board_pose_rate", args.success_threshold)
        center_ranges = effective_ranges(center_rows, "center_rate", args.success_threshold)
        nested_min, nested_max = range_bounds(nested_ranges)
        board_min, board_max = range_bounds(board_ranges)
        center_min, center_max = range_bounds(center_ranges)
        best_nested = max(rows, key=lambda row: row["nested_rate_mean"])

        summary.append(
            {
                "case_id": first["case_id"],
                "case_name": first["case_name"],
                "dimension_name": first["dimension_name"],
                "corner_length_m": first["corner_length_m"],
                "center_length_m": first["center_length_m"],
                "corner_offset_m": first["corner_offset_m"],
                "edge_margin_m": first["edge_margin_m"],
                "repeat_count": args.repeats,
                "height_min_m": args.height_min,
                "height_max_m": args.height_max,
                "height_step_m": args.height_step,
                "samples_per_height": args.samples_per_height,
                "success_threshold": args.success_threshold,
                "mean_nested_rate_all_heights": mean([row["nested_rate_mean"] for row in rows]),
                "mean_board_pose_rate_all_heights": mean([row["board_pose_rate_mean"] for row in rows]),
                "mean_center_rate_all_heights": mean([row["center_rate_mean"] for row in rows]),
                "mean_any_marker_rate_all_heights": mean([row["any_marker_rate_mean"] for row in rows]),
                "mean_detected_markers_all_heights": mean(
                    [row["mean_detected_markers_mean"] for row in rows]
                ),
                "best_height_m_by_nested_rate": best_nested["height_m"],
                "best_nested_rate_mean": best_nested["nested_rate_mean"],
                "nested_effective_min_m": nested_min,
                "nested_effective_max_m": nested_max,
                "nested_effective_ranges_m": json.dumps(nested_ranges, ensure_ascii=False),
                "board_pose_effective_min_m": board_min,
                "board_pose_effective_max_m": board_max,
                "board_pose_effective_ranges_m": json.dumps(board_ranges, ensure_ascii=False),
                "center_effective_min_m": center_min,
                "center_effective_max_m": center_max,
                "center_effective_ranges_m": json.dumps(center_ranges, ensure_ascii=False),
            }
        )
    return summary


def write_case_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "case_id",
        "case_name",
        "dimension_name",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        "repeat_count",
        "height_min_m",
        "height_max_m",
        "height_step_m",
        "samples_per_height",
        "success_threshold",
        "mean_nested_rate_all_heights",
        "mean_board_pose_rate_all_heights",
        "mean_center_rate_all_heights",
        "mean_any_marker_rate_all_heights",
        "mean_detected_markers_all_heights",
        "best_height_m_by_nested_rate",
        "best_nested_rate_mean",
        "nested_effective_min_m",
        "nested_effective_max_m",
        "nested_effective_ranges_m",
        "board_pose_effective_min_m",
        "board_pose_effective_max_m",
        "board_pose_effective_ranges_m",
        "center_effective_min_m",
        "center_effective_max_m",
        "center_effective_ranges_m",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, args, dictionary, detector_params, camera_matrix, dist_coeffs, heights):
    layout = case["layout"]
    texture = make_board_texture(layout, dictionary, args.texture_pixels_per_meter)
    board = create_board(layout, dictionary)
    raw_case_rows = []
    case_raw_root = Path(args.output_root) / "raw_runs" / case["dim_name"]
    case_raw_root.mkdir(parents=True, exist_ok=True)

    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + case["case_id"] * 100000 + repeat * 1000
        rng = random.Random(repeat_seed)
        repeat_rows = []
        for height in heights:
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
            repeat_rows.append(row)
        repeat_path = case_raw_root / f"repeat_{repeat:02d}.csv"
        write_height_csv(repeat_rows, repeat_path)
        raw_case_rows.extend(add_metadata(row, case, repeat) for row in repeat_rows)
        print(
            "case {case_id:02d} repeat {repeat:02d}/{repeats}: raw={path}".format(
                case_id=case["case_id"],
                repeat=repeat,
                repeats=args.repeats,
                path=repeat_path,
            ),
            flush=True,
        )
    return raw_case_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Collect 15 ArUco previews, run repeated height tests, and aggregate results."
    )
    parser.add_argument("--case-start", type=int, default=1)
    parser.add_argument("--case-end", type=int, default=15)
    parser.add_argument("--source-template", default="/tmp/aruco_{case}")
    parser.add_argument("--output-root", default="/tmp/aruco_15case_height_report")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--board-size", type=float, default=1.0)
    parser.add_argument("--height-min", type=float, default=0.3)
    parser.add_argument("--height-max", type=float, default=5.0)
    parser.add_argument("--height-step", type=float, default=0.1)
    parser.add_argument("--samples-per-height", type=int, default=20)
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
    parser.add_argument("--preview-only", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    preview_root = output_root / "previews_by_size"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        discover_case(case_id, args.source_template, args.board_size)
        for case_id in range(args.case_start, args.case_end + 1)
    ]
    preview_records = []
    for case in cases:
        target_preview = copy_preview(case, preview_root)
        preview_records.append(
            {
                "case_id": case["case_id"],
                "dimension_name": case["dim_name"],
                "source_preview": str(case["preview_path"]),
                "organized_preview": str(target_preview),
                "corner_length_m": case["layout"].corner_length,
                "center_length_m": case["layout"].center_length,
                "corner_offset_m": case["layout"].corner_offset,
                "edge_margin_m": case["layout"].edge_margin,
            }
        )

    preview_index = output_root / "preview_index.csv"
    with preview_index.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(preview_records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(preview_records)
    print(f"organized previews: {preview_root}", flush=True)
    print(f"preview index: {preview_index}", flush=True)

    if args.preview_only:
        return

    heights = make_height_values(args.height_min, args.height_max, args.height_step)
    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary_id)
    detector_params = create_detector_params()
    camera_matrix, dist_coeffs = load_camera_params(args.camera_param_path)

    all_raw_rows = []
    for case in cases:
        all_raw_rows.extend(
            run_case(case, args, dictionary, detector_params, camera_matrix, dist_coeffs, heights)
        )

    raw_csv = output_root / "all_raw_height_results.csv"
    averaged_height_csv = output_root / "averaged_by_case_height.csv"
    summary_csv = output_root / "summary_by_case.csv"
    write_raw_csv(all_raw_rows, raw_csv)
    averaged_rows = aggregate_by_case_height(all_raw_rows)
    write_averaged_by_height(averaged_rows, averaged_height_csv)
    summary_rows = aggregate_by_case(averaged_rows, args)
    write_case_summary(summary_rows, summary_csv)

    manifest = {
        "case_count": len(cases),
        "repeats": args.repeats,
        "height_min_m": args.height_min,
        "height_max_m": args.height_max,
        "height_step_m": args.height_step,
        "samples_per_height": args.samples_per_height,
        "success_threshold": args.success_threshold,
        "previews": str(preview_root),
        "preview_index": str(preview_index),
        "raw_combined_csv": str(raw_csv),
        "raw_repeat_root": str(output_root / "raw_runs"),
        "averaged_by_case_height_csv": str(averaged_height_csv),
        "summary_by_case_csv": str(summary_csv),
    }
    manifest_path = output_root / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"raw combined: {raw_csv}", flush=True)
    print(f"averaged by case/height: {averaged_height_csv}", flush=True)
    print(f"summary by case: {summary_csv}", flush=True)
    print(f"batch manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
