#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

from benchmark_aruco_landing import (
    TMP_ROOT,
    benchmark_controller,
    cleanup_processes,
    write_landing_results_table,
)


DEFAULT_ALIGN_PHASES = ("LAND_ALIGN", "DESCEND", "AUTO_LAND", "SUCCESS")
ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_timeseries_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            try:
                rows.append(
                    {
                        "rel_time_s": float(row["rel_time_s"]),
                        "phase": row["phase"],
                        "horizontal_error_m": float(row["horizontal_error_m"]),
                        "roll_deg": float(row["roll_deg"]),
                        "pitch_deg": float(row["pitch_deg"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return rows


def find_alignment_time(rows, align_phases):
    for row in rows:
        if row["phase"] in align_phases:
            return row["rel_time_s"]
    return rows[0]["rel_time_s"] if rows else None


def build_signal_series(rows, align_phases):
    align_time = find_alignment_time(rows, align_phases)
    if align_time is None:
        return None

    times = []
    error_values = []
    roll_abs_values = []
    pitch_abs_values = []
    for row in rows:
        rel_time = row["rel_time_s"] - align_time
        if rel_time < 0.0:
            continue
        times.append(rel_time)
        error_values.append(row["horizontal_error_m"])
        roll_abs_values.append(abs(row["roll_deg"]))
        pitch_abs_values.append(abs(row["pitch_deg"]))

    if len(times) < 2:
        return None

    return {
        "times": times,
        "horizontal_error_m": error_values,
        "roll_abs_deg": roll_abs_values,
        "pitch_abs_deg": pitch_abs_values,
    }


def interp_linear(times, values, target_time):
    if target_time < times[0] or target_time > times[-1]:
        return None
    if target_time == times[0]:
        return values[0]
    if target_time == times[-1]:
        return values[-1]

    left = 0
    right = len(times) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if times[mid] <= target_time:
            left = mid
        else:
            right = mid

    t0 = times[left]
    t1 = times[right]
    v0 = values[left]
    v1 = values[right]
    if t1 <= t0:
        return v0
    ratio = (target_time - t0) / (t1 - t0)
    return v0 + ratio * (v1 - v0)


def percentile(values, ratio):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * ratio
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_series(series_list, field_name, dt_s, max_time_s, min_samples):
    if not series_list:
        return []

    grids = []
    step_count = int(round(max_time_s / dt_s))
    for step in range(step_count + 1):
        stamp = round(step * dt_s, 6)
        samples = []
        for series in series_list:
            sample = interp_linear(series["times"], series[field_name], stamp)
            if sample is not None:
                samples.append(sample)
        if len(samples) < min_samples:
            continue
        grids.append(
            {
                "time_s": stamp,
                "mean": statistics.mean(samples),
                "p25": percentile(samples, 0.25),
                "p75": percentile(samples, 0.75),
                "count": len(samples),
            }
        )
    return grids


def plot_comparison(output_root, group_specs, dt_s, max_time_s, min_samples, align_phases):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    configure_chinese_font(matplotlib, font_manager)

    plots_dir = output_root / "pid_group_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    aligned_groups = []
    for group in group_specs:
        series_list = []
        group_dir = output_root / group["label"]
        for timeseries_path in sorted(group_dir.glob("run_*/timeseries.csv")):
            rows = load_timeseries_rows(timeseries_path)
            series = build_signal_series(rows, align_phases)
            if series is not None:
                series_list.append(series)
        aligned_groups.append(
            {
                "label": display_group_label(group["label"]),
                "param_text": parameter_text(group),
                "series": series_list,
            }
        )

    parameter_note = build_parameter_note(aligned_groups)

    plt.figure(figsize=(11.5, 5.8))
    for group in aligned_groups:
        aggregated = aggregate_series(group["series"], "horizontal_error_m", dt_s, max_time_s, min_samples)
        if not aggregated:
            continue
        time_values = [row["time_s"] for row in aggregated]
        mean_values = [row["mean"] for row in aggregated]
        lower_values = [row["p25"] for row in aggregated]
        upper_values = [row["p75"] for row in aggregated]
        plt.plot(time_values, mean_values, linewidth=2.0, label=group["label"])
        plt.fill_between(time_values, lower_values, upper_values, alpha=0.18)
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=0.9)
    plt.xlabel("对准阶段开始后的时间 (s)")
    plt.ylabel("水平误差 (m)")
    plt.title("不同 PID 参数组下的误差收敛对比图")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    if parameter_note:
        plt.gcf().text(
            0.70,
            0.50,
            parameter_note,
            ha="left",
            va="center",
            fontsize=8.4,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#bbbbbb"},
        )
    plt.tight_layout(rect=(0.0, 0.0, 0.98, 1.0))
    plt.savefig(plots_dir / "error_convergence_comparison.png", dpi=180)
    safe_savefig(plt.gcf(), plots_dir / "error_convergence_comparison.pdf")
    plt.savefig(plots_dir / "误差收敛对比图.png", dpi=180)
    safe_savefig(plt.gcf(), plots_dir / "误差收敛对比图.pdf")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
    for axis, field_name, axis_title in (
        (axes[0], "roll_abs_deg", "横滚响应"),
        (axes[1], "pitch_abs_deg", "俯仰响应"),
    ):
        for group in aligned_groups:
            aggregated = aggregate_series(group["series"], field_name, dt_s, max_time_s, min_samples)
            if not aggregated:
                continue
            time_values = [row["time_s"] for row in aggregated]
            mean_values = [row["mean"] for row in aggregated]
            lower_values = [row["p25"] for row in aggregated]
            upper_values = [row["p75"] for row in aggregated]
            axis.plot(time_values, mean_values, linewidth=2.0, label=group["label"])
            axis.fill_between(time_values, lower_values, upper_values, alpha=0.18)
        axis.set_ylabel("姿态角绝对值 (deg)")
        axis.set_title(axis_title)
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("对准阶段开始后的时间 (s)")
    axes[0].legend(loc="upper right")
    if parameter_note:
        fig.text(
            0.70,
            0.52,
            parameter_note,
            ha="left",
            va="center",
            fontsize=8.4,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#bbbbbb"},
        )
    fig.suptitle("不同 PID 参数组下的姿态响应对比图", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.97))
    fig.savefig(plots_dir / "attitude_response_comparison.png", dpi=180)
    safe_savefig(fig, plots_dir / "attitude_response_comparison.pdf")
    fig.savefig(plots_dir / "姿态响应对比图.png", dpi=180)
    safe_savefig(fig, plots_dir / "姿态响应对比图.pdf")
    plt.close(fig)

    print(f"saved comparison plots to {plots_dir}", flush=True)
    return plots_dir


def build_parameter_note(aligned_groups):
    lines = []
    for group in aligned_groups:
        if not group["param_text"]:
            continue
        lines.append(f"{group['label']}")
        lines.append(group["param_text"])
    return "\n".join(lines)


def safe_savefig(fig, path):
    try:
        fig.savefig(path)
    except Exception as exc:
        print(f"warning: failed to save {path}: {exc}", flush=True)


def parameter_text(group):
    launch_args = group.get("launch_args", {})
    controller = str(group.get("controller") or "")
    if "no_pid" in controller:
        return "无 PID 外环\n直接目标位置跟随\n固定下降速率"
    if "baseline" in controller:
        return "基线控制器\n简单均值估计\n简化下降策略"
    if not launch_args:
        return ""
    track = (
        launch_args.get("track_pid_kp"),
        launch_args.get("track_pid_ki"),
        launch_args.get("track_pid_kd"),
    )
    land = (
        launch_args.get("land_pid_kp"),
        launch_args.get("land_pid_ki"),
        launch_args.get("land_pid_kd"),
    )
    land_z = (
        launch_args.get("land_z_pid_kp"),
        launch_args.get("land_z_pid_ki"),
        launch_args.get("land_z_pid_kd"),
    )
    parts = []
    if any(value is not None for value in track):
        parts.append("跟踪环  = ({:.2f}, {:.2f}, {:.2f})".format(*float_triplet(track)))
    if any(value is not None for value in land):
        parts.append("对准环  = ({:.2f}, {:.2f}, {:.2f})".format(*float_triplet(land)))
    if any(value is not None for value in land_z):
        parts.append("高度环  = ({:.2f}, {:.2f}, {:.2f})".format(*float_triplet(land_z)))
    return "\n".join(parts)


def float_triplet(values):
    out = []
    for value in values:
        if value is None:
            out.append(0.0)
        else:
            out.append(float(value))
    return out


def display_group_label(label):
    return {
        "no_pid": "无PID组",
        "baseline_controller": "基线控制器组",
        "pid_baseline": "基准PID组",
        "conservative": "保守组",
        "baseline": "基准组",
        "aggressive": "激进组",
    }.get(label, label)


def configure_chinese_font(matplotlib, font_manager):
    font_file_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for font_path in font_file_candidates:
        if os.path.exists(font_path):
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.family"] = [font_name]
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = None
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break
    if selected_font is None:
        return
    matplotlib.rcParams["font.family"] = [selected_font]
    matplotlib.rcParams["font.sans-serif"] = [selected_font]
    matplotlib.rcParams["axes.unicode_minus"] = False


def write_group_metrics_csv(output_root, results, group_specs):
    output_path = output_root / "pid_group_metrics.csv"
    fieldnames = [
        "group",
        "controller_type",
        "detector_type",
        "runs",
        "success_rate_mean",
        "final_error_mean_m",
        "landing_time_mean_s",
        "roll_p95_mean_deg",
        "pitch_p95_mean_deg",
        "attitude_within_2deg_ratio_mean",
        "response_latency_p95_mean_s",
        "launch_args_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in group_specs:
            group_result = results.get(group["label"], {})
            summary = group_result.get("summary", {})
            writer.writerow(
                {
                    "group": group["label"],
                    "controller_type": group_result.get("controller_type"),
                    "detector_type": group_result.get("detector_type"),
                    "runs": len(group_result.get("runs", [])),
                    "success_rate_mean": summary.get("success_rate"),
                    "final_error_mean_m": nested_mean(summary.get("final_error_m")),
                    "landing_time_mean_s": nested_mean(summary.get("landing_time_s")),
                    "roll_p95_mean_deg": nested_mean(summary.get("roll_abs_p95_deg")),
                    "pitch_p95_mean_deg": nested_mean(summary.get("pitch_abs_p95_deg")),
                    "attitude_within_2deg_ratio_mean": nested_mean(summary.get("attitude_within_2deg_ratio")),
                    "response_latency_p95_mean_s": nested_mean(summary.get("response_latency_p95_s")),
                    "launch_args_json": json.dumps(group_result.get("launch_args", {}), ensure_ascii=False, sort_keys=True),
                }
            )
    return output_path


def nested_mean(value):
    if isinstance(value, dict):
        return value.get("mean")
    return value


def normalize_group_specs(raw_groups):
    group_specs = []
    for item in raw_groups:
        label = str(item.get("label", "")).strip()
        if not label:
            raise SystemExit("each PID group must define a non-empty label")
        launch_args = item.get("launch_args", {})
        if not isinstance(launch_args, dict):
            raise SystemExit(f"launch_args for group {label} must be a JSON object")
        group_specs.append(
            {
                "label": label,
                "controller": item.get("controller"),
                "detector_type": item.get("detector_type"),
                "marker_config": item.get("marker_config"),
                "world": item.get("world"),
                "target_x": item.get("target_x"),
                "target_y": item.get("target_y"),
                "launch_args": launch_args,
            }
        )
    if not group_specs:
        raise SystemExit("config must contain at least one PID group")
    return group_specs


def write_parameter_table(output_root, group_specs):
    output_path = output_root / "pid_group_parameters.csv"
    fieldnames = [
        "group",
        "track_pid_kp",
        "track_pid_ki",
        "track_pid_kd",
        "land_pid_kp",
        "land_pid_ki",
        "land_pid_kd",
        "land_z_pid_kp",
        "land_z_pid_ki",
        "land_z_pid_kd",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in group_specs:
            row = {"group": group["label"]}
            row.update(group["launch_args"])
            writer.writerow(row)
    return output_path


def run_groups(config, output_root):
    env = dict(os.environ)
    world = config.get("world", "$(find mavlink_sitl_gazebo)/worlds/aruco_search_demo.world")
    marker_config = config.get("marker_config", "$(find px4)/config/aruco_nested_board.yaml")
    detector_type = config.get("detector_type", "aruco_multi_marker_det.py")
    controller = config.get("controller", "aruco_search_and_detect.py")
    runs = int(config.get("runs", 3))
    timeout_s = float(config.get("timeout", 150.0))
    target_xy = (
        float(config.get("target_x", 1.5)),
        float(config.get("target_y", 1.5)),
    )

    results = {}
    group_specs = normalize_group_specs(config.get("groups", []))
    for group in group_specs:
        label = group["label"]
        results[label] = benchmark_controller(
            label=label,
            controller_type=group["controller"] or controller,
            runs=runs,
            timeout_s=timeout_s,
            env=env,
            output_root=output_root,
            world=group["world"] or world,
            marker_config=group["marker_config"] or marker_config,
            detector_type=group["detector_type"] or detector_type,
            target_xy=(
                float(group["target_x"]) if group["target_x"] is not None else target_xy[0],
                float(group["target_y"]) if group["target_y"] is not None else target_xy[1],
            ),
            launch_args=group["launch_args"],
        )
    return group_specs, results


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple PID parameter groups and generate comparison plots from benchmark timeseries."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "pid_groups_default.json"),
        help="JSON file listing PID groups and shared benchmark settings",
    )
    parser.add_argument("--output-root", default=None, help="Override output directory from config")
    parser.add_argument("--plots-only", action="store_true", help="Skip benchmarking and only render plots")
    parser.add_argument("--dt", type=float, default=0.05, help="Interpolation step for comparison plots")
    parser.add_argument("--max-time", type=float, default=12.0, help="Plot window after LAND_ALIGN starts")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum number of runs contributing to each plotted point",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_json(config_path)
    output_root = resolve_project_path(args.output_root or config.get("output_root", str(TMP_ROOT / "pid_groups")))
    output_root.mkdir(parents=True, exist_ok=True)

    group_specs = normalize_group_specs(config.get("groups", []))
    results = {}
    if not args.plots_only:
        group_specs, results = run_groups(config, output_root)
        dump_json(output_root / "pid_group_summary.json", results)
        dump_json(output_root / "pid_groups_manifest.json", {"config_path": str(config_path), "groups": group_specs})
        write_landing_results_table(results, output_root)
        write_group_metrics_csv(output_root, results, group_specs)
    write_parameter_table(output_root, group_specs)

    plots_dir = plot_comparison(
        output_root=output_root,
        group_specs=group_specs,
        dt_s=args.dt,
        max_time_s=args.max_time,
        min_samples=args.min_samples,
        align_phases=DEFAULT_ALIGN_PHASES,
    )
    if not args.plots_only:
        print(f"summary saved to {output_root / 'pid_group_summary.json'}", flush=True)
        print(f"metrics saved to {output_root / 'pid_group_metrics.csv'}", flush=True)
    print(f"plots saved to {plots_dir}", flush=True)
    cleanup_processes()


if __name__ == "__main__":
    main()
