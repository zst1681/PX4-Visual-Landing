#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

from batch_aruco_height_experiments import discover_case
from benchmark_aruco_nested_layouts import (
    ALL_MARKER_IDS,
    create_board,
    create_detector_params,
    evaluate_detection,
    load_camera_params,
    make_board_texture,
    render_scenario,
)
from test_aruco_effective_height import make_scenario_for_height


ROOT = Path(__file__).resolve().parents[1]
COUNT_FIELDS = [
    "zero_marker_rate",
    "any_marker_rate",
    "one_marker_rate",
    "two_marker_rate",
    "three_marker_rate",
    "four_marker_rate",
    "all_five_marker_rate",
    "center_rate",
    "board_pose_rate",
    "nested_rate",
    "mean_detected_markers",
]


def mean(values):
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def stdev(values):
    clean = [value for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else 0.0 if len(clean) == 1 else None


def height_values(start, max_height, step):
    values = []
    current = start
    while current <= max_height + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def rate_range(rows, field, threshold):
    ranges = []
    start = None
    previous_height = None
    for row in rows:
        passed = row[field] >= threshold
        if passed and start is None:
            start = row["height_m"]
        if not passed and start is not None:
            ranges.append([start, previous_height])
            start = None
        previous_height = row["height_m"]
    if start is not None:
        ranges.append([start, previous_height])
    return ranges


def max_height_for(rows, field, threshold):
    passed = [row["height_m"] for row in rows if row[field] >= threshold]
    return max(passed) if passed else None


def evaluate_one_height(height, layout, texture, dictionary, detector_params, camera_matrix, dist_coeffs, board, args, rng):
    counts = {count: 0 for count in range(6)}
    center_hits = 0
    board_pose_hits = 0
    nested_hits = 0
    detected_marker_total = 0

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
        detected_count = min(len(result["detected_ids"]), 5)
        counts[detected_count] += 1
        detected_marker_total += detected_count
        center_hits += 1 if result["center_hit"] else 0
        board_pose_hits += 1 if result["board_pose_hit"] else 0
        nested_hits += 1 if result["nested_hit"] else 0

    total = args.samples_per_height
    return {
        "height_m": height,
        "samples": total,
        "zero_marker_rate": counts[0] / total,
        "any_marker_rate": 1.0 - counts[0] / total,
        "one_marker_rate": counts[1] / total,
        "two_marker_rate": counts[2] / total,
        "three_marker_rate": counts[3] / total,
        "four_marker_rate": counts[4] / total,
        "all_five_marker_rate": counts[5] / total,
        "center_rate": center_hits / total,
        "board_pose_rate": board_pose_hits / total,
        "nested_rate": nested_hits / total,
        "mean_detected_markers": detected_marker_total / total,
    }


def write_csv(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, args, dictionary, detector_params, camera_matrix, dist_coeffs, heights):
    layout = case["layout"]
    texture = make_board_texture(layout, dictionary, args.texture_pixels_per_meter)
    board = create_board(layout, dictionary)
    raw_rows = []
    averaged_rows = []
    empty_streak = 0
    case_root = Path(args.output_root) / "raw_runs" / case["dim_name"]
    case_root.mkdir(parents=True, exist_ok=True)

    for height in heights:
        repeat_rows = []
        for repeat in range(1, args.repeats + 1):
            rng = random.Random(args.seed + case["case_id"] * 100000 + repeat * 1000 + int(round(height * 100)))
            row = evaluate_one_height(
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
            row.update(
                {
                    "case_id": case["case_id"],
                    "dimension_name": case["dim_name"],
                    "repeat": repeat,
                    "corner_length_m": layout.corner_length,
                    "center_length_m": layout.center_length,
                    "corner_offset_m": layout.corner_offset,
                    "edge_margin_m": layout.edge_margin,
                }
            )
            repeat_rows.append(row)
            raw_rows.append(row)

        averaged = {
            "case_id": case["case_id"],
            "dimension_name": case["dim_name"],
            "height_m": height,
            "repeat_count": args.repeats,
            "samples_per_height": args.samples_per_height,
            "corner_length_m": layout.corner_length,
            "center_length_m": layout.center_length,
            "corner_offset_m": layout.corner_offset,
            "edge_margin_m": layout.edge_margin,
        }
        for field in COUNT_FIELDS:
            values = [row[field] for row in repeat_rows]
            averaged[f"{field}_mean"] = mean(values)
            averaged[f"{field}_std"] = stdev(values)
        averaged_rows.append(averaged)

        print(
            "case {case_id:02d} height={height:.2f} any={any:.3f} all5={all5:.3f} three={three:.3f}".format(
                case_id=case["case_id"],
                height=height,
                any=averaged["any_marker_rate_mean"],
                all5=averaged["all_five_marker_rate_mean"],
                three=averaged["three_marker_rate_mean"],
            ),
            flush=True,
        )

        if averaged["any_marker_rate_mean"] <= 0.0:
            empty_streak += 1
        else:
            empty_streak = 0
        if empty_streak >= args.stop_empty_heights:
            break

    raw_fieldnames = [
        "case_id",
        "dimension_name",
        "repeat",
        "height_m",
        "samples",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        *COUNT_FIELDS,
    ]
    avg_fieldnames = [
        "case_id",
        "dimension_name",
        "height_m",
        "repeat_count",
        "samples_per_height",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
    ]
    for field in COUNT_FIELDS:
        avg_fieldnames.extend([f"{field}_mean", f"{field}_std"])

    write_csv(raw_rows, case_root / "raw_by_repeat.csv", raw_fieldnames)
    write_csv(averaged_rows, case_root / "averaged_by_height.csv", avg_fieldnames)
    return raw_rows, averaged_rows


def summarize_case(case, averaged_rows, args):
    layout = case["layout"]
    rows = sorted(averaged_rows, key=lambda row: row["height_m"])
    low_mid_rows = [
        row
        for row in rows
        if args.center_low_height <= row["height_m"] <= args.center_mid_height
    ]
    best_any = max(rows, key=lambda row: (row["any_marker_rate_mean"], row["height_m"]))
    best_center_low_mid = max(
        low_mid_rows or rows,
        key=lambda row: (row["center_rate_mean"], -abs(row["height_m"] - args.center_mid_height)),
    )

    all5_rows = [
        {"height_m": row["height_m"], "all_five_marker_rate": row["all_five_marker_rate_mean"]}
        for row in rows
    ]
    three_rows = [
        {"height_m": row["height_m"], "three_marker_rate": row["three_marker_rate_mean"]}
        for row in rows
    ]
    any_rows = [
        {"height_m": row["height_m"], "any_marker_rate": row["any_marker_rate_mean"]}
        for row in rows
    ]

    return {
        "case_id": case["case_id"],
        "dimension_name": case["dim_name"],
        "corner_length_m": layout.corner_length,
        "center_length_m": layout.center_length,
        "corner_offset_m": layout.corner_offset,
        "edge_margin_m": layout.edge_margin,
        "max_any_marker_height_m": max_height_for(rows, "any_marker_rate_mean", args.any_threshold),
        "stable_any_marker_max_height_m": max_height_for(rows, "any_marker_rate_mean", args.stable_threshold),
        "max_all_five_height_m": max_height_for(rows, "all_five_marker_rate_mean", args.any_threshold),
        "stable_all_five_max_height_m": max_height_for(rows, "all_five_marker_rate_mean", args.stable_threshold),
        "max_three_marker_height_m": max_height_for(rows, "three_marker_rate_mean", args.any_threshold),
        "stable_three_marker_max_height_m": max_height_for(rows, "three_marker_rate_mean", args.stable_threshold),
        "any_marker_effective_ranges_m": json.dumps(
            rate_range(any_rows, "any_marker_rate", args.stable_threshold),
            ensure_ascii=False,
        ),
        "all_five_effective_ranges_m": json.dumps(
            rate_range(all5_rows, "all_five_marker_rate", args.stable_threshold),
            ensure_ascii=False,
        ),
        "three_marker_effective_ranges_m": json.dumps(
            rate_range(three_rows, "three_marker_rate", args.stable_threshold),
            ensure_ascii=False,
        ),
        "mean_any_marker_rate_all_tested_heights": mean([row["any_marker_rate_mean"] for row in rows]),
        "mean_all_five_rate_all_tested_heights": mean([row["all_five_marker_rate_mean"] for row in rows]),
        "mean_three_marker_rate_all_tested_heights": mean([row["three_marker_rate_mean"] for row in rows]),
        "mean_center_rate_low_mid_heights": mean([row["center_rate_mean"] for row in low_mid_rows]),
        "max_center_rate_low_mid_heights": max(row["center_rate_mean"] for row in low_mid_rows) if low_mid_rows else None,
        "best_center_low_mid_height_m": best_center_low_mid["height_m"],
        "best_center_low_mid_rate": best_center_low_mid["center_rate_mean"],
        "best_any_marker_height_m": best_any["height_m"],
        "best_any_marker_rate": best_any["any_marker_rate_mean"],
        "last_tested_height_m": rows[-1]["height_m"],
        "stopped_after_empty_heights": rows[-1]["any_marker_rate_mean"] <= 0.0,
    }


def write_chinese_summary(summary_rows, output_path):
    columns = [
        ("case_id", "组别"),
        ("dimension_name", "尺寸命名"),
        ("corner_length_m", "四角码边长_m"),
        ("center_length_m", "中心码边长_m"),
        ("corner_offset_m", "四角码中心偏移_m"),
        ("edge_margin_m", "边缘留白_m"),
        ("max_any_marker_height_m", "任意码可识别最大高度_m"),
        ("stable_any_marker_max_height_m", "任意码稳定识别最大高度_m"),
        ("max_all_five_height_m", "五码全识别最大高度_m"),
        ("stable_all_five_max_height_m", "五码全识别稳定最大高度_m"),
        ("max_three_marker_height_m", "恰好3码识别最大高度_m"),
        ("stable_three_marker_max_height_m", "恰好3码稳定最大高度_m"),
        ("mean_center_rate_low_mid_heights", "中低高度中心码平均识别率"),
        ("max_center_rate_low_mid_heights", "中低高度中心码最高识别率"),
        ("best_center_low_mid_height_m", "中心码最佳中低高度_m"),
        ("all_five_effective_ranges_m", "五码稳定识别高度区间_m"),
        ("three_marker_effective_ranges_m", "恰好3码稳定识别高度区间_m"),
        ("any_marker_effective_ranges_m", "任意码稳定识别高度区间_m"),
        ("last_tested_height_m", "最后测试高度_m"),
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name_cn for _name, name_cn in columns])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name_cn: row.get(name) for name, name_cn in columns})


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Sweep upward until no ArUco marker is detected.")
    parser.add_argument("--case-start", type=int, default=1)
    parser.add_argument("--case-end", type=int, default=15)
    parser.add_argument("--source-template", default="/tmp/aruco_{case}")
    parser.add_argument("--output-root", default="/tmp/aruco_15case_max_height_report")
    parser.add_argument("--board-size", type=float, default=1.0)
    parser.add_argument("--height-start", type=float, default=0.6)
    parser.add_argument("--height-max", type=float, default=20.0)
    parser.add_argument("--height-step", type=float, default=0.5)
    parser.add_argument("--stop-empty-heights", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples-per-height", type=int, default=5)
    parser.add_argument("--any-threshold", type=float, default=0.001)
    parser.add_argument("--stable-threshold", type=float, default=0.8)
    parser.add_argument("--center-low-height", type=float, default=0.6)
    parser.add_argument("--center-mid-height", type=float, default=3.0)
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
    parser.add_argument("--seed", type=int, default=67)
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        discover_case(case_id, args.source_template, args.board_size)
        for case_id in range(args.case_start, args.case_end + 1)
    ]
    heights = height_values(args.height_start, args.height_max, args.height_step)
    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary_id)
    detector_params = create_detector_params()
    camera_matrix, dist_coeffs = load_camera_params(args.camera_param_path)

    all_raw_rows = []
    all_averaged_rows = []
    summary_rows = []
    for case in cases:
        raw_rows, averaged_rows = run_case(
            case,
            args,
            dictionary,
            detector_params,
            camera_matrix,
            dist_coeffs,
            heights,
        )
        all_raw_rows.extend(raw_rows)
        all_averaged_rows.extend(averaged_rows)
        summary_rows.append(summarize_case(case, averaged_rows, args))

    raw_fieldnames = [
        "case_id",
        "dimension_name",
        "repeat",
        "height_m",
        "samples",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        *COUNT_FIELDS,
    ]
    avg_fieldnames = [
        "case_id",
        "dimension_name",
        "height_m",
        "repeat_count",
        "samples_per_height",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
    ]
    for field in COUNT_FIELDS:
        avg_fieldnames.extend([f"{field}_mean", f"{field}_std"])

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(all_raw_rows, output_root / "all_raw_by_repeat.csv", raw_fieldnames)
    write_csv(all_averaged_rows, output_root / "averaged_by_case_height.csv", avg_fieldnames)
    write_csv(summary_rows, output_root / "summary_by_case.csv", summary_fieldnames)
    write_chinese_summary(summary_rows, output_root / "summary_by_case_cn.csv")

    best_distance = max(
        summary_rows,
        key=lambda row: (
            row["max_any_marker_height_m"] if row["max_any_marker_height_m"] is not None else -1.0,
            row["stable_any_marker_max_height_m"] if row["stable_any_marker_max_height_m"] is not None else -1.0,
            row["mean_any_marker_rate_all_tested_heights"],
        ),
    )
    best_center = max(
        summary_rows,
        key=lambda row: (
            row["mean_center_rate_low_mid_heights"] if row["mean_center_rate_low_mid_heights"] is not None else -1.0,
            row["max_center_rate_low_mid_heights"] if row["max_center_rate_low_mid_heights"] is not None else -1.0,
            row["center_length_m"],
        ),
    )
    judgement = {
        "best_recognition_distance_case": best_distance,
        "best_low_mid_center_case": best_center,
    }
    (output_root / "judgement.json").write_text(
        json.dumps(judgement, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "case_count": len(cases),
        "height_start_m": args.height_start,
        "height_max_m": args.height_max,
        "height_step_m": args.height_step,
        "stop_empty_heights": args.stop_empty_heights,
        "repeats": args.repeats,
        "samples_per_height": args.samples_per_height,
        "stable_threshold": args.stable_threshold,
        "center_low_height_m": args.center_low_height,
        "center_mid_height_m": args.center_mid_height,
        "raw_csv": str(output_root / "all_raw_by_repeat.csv"),
        "averaged_csv": str(output_root / "averaged_by_case_height.csv"),
        "summary_csv": str(output_root / "summary_by_case.csv"),
        "summary_cn_csv": str(output_root / "summary_by_case_cn.csv"),
        "judgement_json": str(output_root / "judgement.json"),
    }
    (output_root / "max_height_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary cn: {output_root / 'summary_by_case_cn.csv'}", flush=True)
    print(f"best distance case: {best_distance['case_id']} {best_distance['dimension_name']}", flush=True)
    print(f"best center case: {best_center['case_id']} {best_center['dimension_name']}", flush=True)


if __name__ == "__main__":
    main()
