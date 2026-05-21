#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "generated" / "pid_group_plots_with_no_pid_and_baseline_controller_20260513"
DEFAULT_NO_PID_SOURCE_ROOT = ROOT / "generated" / "no_pid_fixed_response_20260514"
DEFAULT_OUTPUT_ROOT = ROOT / "generated" / "outer_pid_position_input_response_20260514"


def load_pose(path):
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        start_time = None
        for row in reader:
            try:
                stamp = float(row["field.header.stamp"])
                x_val = float(row["field.pose.position.x"])
                y_val = float(row["field.pose.position.y"])
            except (KeyError, TypeError, ValueError):
                continue
            if start_time is None:
                start_time = stamp
            rows.append(
                {
                    "rel_time_s": (stamp - start_time) / 1e9,
                    "x_m": x_val,
                    "y_m": y_val,
                }
            )
        return rows


def constant_input_like(rows, x_value, y_value):
    return [
        {
            "rel_time_s": row["rel_time_s"],
            "x_m": x_value,
            "y_m": y_value,
        }
        for row in rows
    ]


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


def load_metrics_by_group(path):
    metrics = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics[row["group"]] = row
    return metrics


def metric_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def safe_savefig(fig, path):
    try:
        fig.savefig(path, dpi=180)
    except Exception as exc:
        print(f"warning: failed to save {path}: {exc}", flush=True)


def plot_case(axis_x, axis_y, title, input_rows, output_rows, target_x, target_y, input_note):
    axis_x.set_title(f"{title}：X 通道")
    axis_y.set_title(f"{title}：Y 通道")

    if input_rows:
        axis_x.plot(
            [row["rel_time_s"] for row in input_rows],
            [row["x_m"] for row in input_rows],
            color="#111111",
            linestyle="--",
            linewidth=1.6,
            label=f"输入 x setpoint（{input_note}）",
        )
        axis_y.plot(
            [row["rel_time_s"] for row in input_rows],
            [row["y_m"] for row in input_rows],
            color="#111111",
            linestyle="--",
            linewidth=1.6,
            label=f"输入 y setpoint（{input_note}）",
        )
    else:
        axis_x.axhline(
            target_x,
            color="#111111",
            linestyle="--",
            linewidth=1.6,
            label="输入 x 目标",
        )
        axis_y.axhline(
            target_y,
            color="#111111",
            linestyle="--",
            linewidth=1.6,
            label="输入 y 目标",
        )

    axis_x.plot(
        [row["rel_time_s"] for row in output_rows],
        [row["x_m"] for row in output_rows],
        color="#2563eb",
        linewidth=2.0,
        label="输出 x 实际位置",
    )
    axis_y.plot(
        [row["rel_time_s"] for row in output_rows],
        [row["y_m"] for row in output_rows],
        color="#dc2626",
        linewidth=2.0,
        label="输出 y 实际位置",
    )

    axis_x.set_ylabel("位置 (m)")
    axis_y.set_ylabel("位置 (m)")
    for axis in (axis_x, axis_y):
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8.4)


def save_case_figure(output_root, filename, title, input_rows, output_rows, target_x, target_y, input_note, note):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True)
    plot_case(
        axes[0],
        axes[1],
        title,
        input_rows,
        output_rows,
        target_x,
        target_y,
        input_note,
    )
    axes[1].set_xlabel("任务开始后的时间 (s)")
    if note:
        fig.text(
            0.70,
            0.52,
            note,
            ha="left",
            va="center",
            fontsize=9.0,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#bbbbbb"},
        )
    fig.suptitle(title + "输入-输出响应曲线", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
    safe_savefig(fig, output_root / filename)
    plt.close(fig)


def summary_text(no_pid_metrics_by_group, pid_metrics_by_group, target_x, target_y):
    no_pid = no_pid_metrics_by_group["no_pid"]
    pid = pid_metrics_by_group["pid_baseline"]
    lines = [
        "位置输入",
        "目标位置 = ({:.1f}, {:.1f}) m".format(target_x, target_y),
        "",
        "无 PID 外环（正常启动）",
        "成功率 = {:.0f}%".format(100.0 * metric_float(no_pid, "success_rate_mean")),
        "最终误差 = {:.3f} m".format(metric_float(no_pid, "final_error_mean_m")),
        "着陆时间 = {:.3f} s".format(metric_float(no_pid, "landing_time_mean_s")),
        "",
        "有 PID 外环（基准 PID 组）",
        "成功率 = {:.0f}%".format(100.0 * metric_float(pid, "success_rate_mean")),
        "最终误差 = {:.3f} m".format(metric_float(pid, "final_error_mean_m")),
        "着陆时间 = {:.3f} s".format(metric_float(pid, "landing_time_mean_s")),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot outer-loop no-PID vs PID position input-response curves.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--no-pid-source-root", default=str(DEFAULT_NO_PID_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-x", type=float, default=1.5)
    parser.add_argument("--target-y", type=float, default=1.5)
    parser.add_argument("--max-time", type=float, default=75.0)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    no_pid_source_root = Path(args.no_pid_source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    no_pid_rows = load_pose(no_pid_source_root / "no_pid" / "run_01" / "local_pose.csv")
    pid_rows = load_pose(source_root / "pid_baseline" / "run_01" / "local_pose.csv")
    no_pid_setpoint_rows = load_pose(no_pid_source_root / "no_pid" / "run_01" / "setpoint_pose.csv")
    pid_setpoint_rows = load_pose(source_root / "pid_baseline" / "run_01" / "setpoint_pose.csv")
    metrics_by_group = load_metrics_by_group(source_root / "pid_group_metrics.csv")
    no_pid_metrics_by_group = load_metrics_by_group(no_pid_source_root / "pid_group_metrics.csv")

    no_pid_rows = [row for row in no_pid_rows if row["rel_time_s"] <= args.max_time]
    pid_rows = [row for row in pid_rows if row["rel_time_s"] <= args.max_time]
    no_pid_setpoint_rows = [row for row in no_pid_setpoint_rows if row["rel_time_s"] <= args.max_time]
    pid_setpoint_rows = [row for row in pid_setpoint_rows if row["rel_time_s"] <= args.max_time]

    if len(no_pid_setpoint_rows) < 2:
        no_pid_setpoint_rows = constant_input_like(no_pid_rows, args.target_x, args.target_y)
        no_pid_input_note = "目标输入，原 no_pid 节点未持续发布 setpoint"
    else:
        no_pid_input_note = "实际 setpoint"
    pid_input_note = "实际 setpoint"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    configure_chinese_font(matplotlib, font_manager)
    note = summary_text(no_pid_metrics_by_group, metrics_by_group, args.target_x, args.target_y)
    no_pid_note = "\n".join(
        [
            "无 PID 外环，正常启动",
            "成功率 = {:.0f}%".format(100.0 * metric_float(no_pid_metrics_by_group["no_pid"], "success_rate_mean")),
            "最终误差 = {:.3f} m".format(metric_float(no_pid_metrics_by_group["no_pid"], "final_error_mean_m")),
            "着陆时间 = {:.3f} s".format(metric_float(no_pid_metrics_by_group["no_pid"], "landing_time_mean_s")),
        ]
    )
    pid_note = "\n".join(
        [
            "有 PID 外环，基准 PID 组",
            "成功率 = {:.0f}%".format(100.0 * metric_float(metrics_by_group["pid_baseline"], "success_rate_mean")),
            "最终误差 = {:.3f} m".format(metric_float(metrics_by_group["pid_baseline"], "final_error_mean_m")),
            "着陆时间 = {:.3f} s".format(metric_float(metrics_by_group["pid_baseline"], "landing_time_mean_s")),
        ]
    )

    save_case_figure(
        output_root,
        "无PID工况_位置输入输出响应曲线.png",
        "无 PID 外环工况",
        no_pid_setpoint_rows,
        no_pid_rows,
        args.target_x,
        args.target_y,
        no_pid_input_note,
        no_pid_note,
    )
    save_case_figure(
        output_root,
        "有PID工况_位置输入输出响应曲线.png",
        "有 PID 外环工况",
        pid_setpoint_rows,
        pid_rows,
        args.target_x,
        args.target_y,
        pid_input_note,
        pid_note,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), sharex="row")
    plot_case(
        axes[0][0],
        axes[0][1],
        "无 PID 外环工况",
        no_pid_setpoint_rows,
        no_pid_rows,
        args.target_x,
        args.target_y,
        no_pid_input_note,
    )
    plot_case(
        axes[1][0],
        axes[1][1],
        "有 PID 外环工况",
        pid_setpoint_rows,
        pid_rows,
        args.target_x,
        args.target_y,
        pid_input_note,
    )
    axes[1][0].set_xlabel("任务开始后的时间 (s)")
    axes[1][1].set_xlabel("任务开始后的时间 (s)")
    fig.suptitle("两种工况下的位置输入-输出响应曲线", y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    safe_savefig(fig, output_root / "两种工况_位置输入输出响应曲线.png")
    safe_savefig(fig, output_root / "外环PID有无_位置输入响应曲线.png")
    plt.close(fig)

    with (output_root / "外环PID有无_位置输入响应说明.txt").open("w", encoding="utf-8") as handle:
        handle.write(note + "\n")

    print(f"saved plots to {output_root}", flush=True)


if __name__ == "__main__":
    main()
