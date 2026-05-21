#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "generated" / "pid_group_plots_with_no_pid_and_baseline_controller_20260513"
DEFAULT_OUTPUT_ROOT = ROOT / "generated" / "outer_pid_on_off_20260514"


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_timeseries(path):
    rows = []
    for row in load_rows(path):
        try:
            rows.append(
                {
                    "rel_time_s": float(row["rel_time_s"]),
                    "horizontal_error_m": float(row["horizontal_error_m"]),
                    "roll_deg": abs(float(row["roll_deg"])),
                    "pitch_deg": abs(float(row["pitch_deg"])),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def load_metrics_by_group(path):
    metrics = {}
    for row in load_rows(path):
        metrics[row["group"]] = row
    return metrics


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


def metric_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def make_summary_text(metrics_by_group):
    no_pid = metrics_by_group["no_pid"]
    pid = metrics_by_group["pid_baseline"]
    lines = [
        "无 PID 外环",
        "成功率 = {:.0f}%".format(100.0 * metric_float(no_pid, "success_rate_mean")),
        "最终误差 = {:.3f} m".format(metric_float(no_pid, "final_error_mean_m")),
        "",
        "有 PID 外环（基准 PID 组）",
        "成功率 = {:.0f}%".format(100.0 * metric_float(pid, "success_rate_mean")),
        "最终误差 = {:.3f} m".format(metric_float(pid, "final_error_mean_m")),
        "着陆时间 = {:.3f} s".format(metric_float(pid, "landing_time_mean_s")),
        "track = (0.72, 0.00, 0.18)",
        "land  = (0.58, 0.02, 0.16)",
        "z环    = (1.05, 0.06, 0.18)",
    ]
    return "\n".join(lines)


def safe_savefig(fig, path):
    try:
        fig.savefig(path, dpi=180)
    except Exception as exc:
        print(f"warning: failed to save {path}: {exc}", flush=True)


def write_summary_csv(output_path, metrics_by_group):
    fieldnames = [
        "group",
        "label",
        "success_rate",
        "final_error_m",
        "landing_time_s",
        "roll_p95_deg",
        "pitch_p95_deg",
        "response_latency_p95_s",
    ]
    rows = [
        {
            "group": "no_pid",
            "label": "无PID外环",
            "success_rate": metric_float(metrics_by_group["no_pid"], "success_rate_mean"),
            "final_error_m": metric_float(metrics_by_group["no_pid"], "final_error_mean_m"),
            "landing_time_s": metric_float(metrics_by_group["no_pid"], "landing_time_mean_s"),
            "roll_p95_deg": metric_float(metrics_by_group["no_pid"], "roll_p95_mean_deg"),
            "pitch_p95_deg": metric_float(metrics_by_group["no_pid"], "pitch_p95_mean_deg"),
            "response_latency_p95_s": metric_float(metrics_by_group["no_pid"], "response_latency_p95_mean_s"),
        },
        {
            "group": "pid_baseline",
            "label": "有PID外环（基准PID组）",
            "success_rate": metric_float(metrics_by_group["pid_baseline"], "success_rate_mean"),
            "final_error_m": metric_float(metrics_by_group["pid_baseline"], "final_error_mean_m"),
            "landing_time_s": metric_float(metrics_by_group["pid_baseline"], "landing_time_mean_s"),
            "roll_p95_deg": metric_float(metrics_by_group["pid_baseline"], "roll_p95_mean_deg"),
            "pitch_p95_deg": metric_float(metrics_by_group["pid_baseline"], "pitch_p95_mean_deg"),
            "response_latency_p95_s": metric_float(metrics_by_group["pid_baseline"], "response_latency_p95_mean_s"),
        },
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot no-PID vs PID outer-loop response curves from saved benchmark data.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-time", type=float, default=50.0)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    no_pid_timeseries = load_timeseries(source_root / "no_pid" / "run_01" / "timeseries.csv")
    pid_timeseries = load_timeseries(source_root / "pid_baseline" / "run_01" / "timeseries.csv")
    metrics_by_group = load_metrics_by_group(source_root / "pid_group_metrics.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    configure_chinese_font(matplotlib, font_manager)

    summary_text = make_summary_text(metrics_by_group)

    no_pid_rows = [row for row in no_pid_timeseries if row["rel_time_s"] <= args.max_time]
    pid_rows = [row for row in pid_timeseries if row["rel_time_s"] <= args.max_time]

    fig = plt.figure(figsize=(11.2, 5.8))
    plt.plot(
        [row["rel_time_s"] for row in no_pid_rows],
        [row["horizontal_error_m"] for row in no_pid_rows],
        linewidth=2.2,
        label="无PID外环",
    )
    plt.plot(
        [row["rel_time_s"] for row in pid_rows],
        [row["horizontal_error_m"] for row in pid_rows],
        linewidth=2.2,
        label="有PID外环（基准PID组）",
    )
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=0.9)
    plt.xlabel("任务开始后的时间 (s)")
    plt.ylabel("水平误差 (m)")
    plt.title("外环PID有无情况下的水平误差响应曲线")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    fig.text(
        0.72,
        0.52,
        summary_text,
        ha="left",
        va="center",
        fontsize=9.0,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#bbbbbb"},
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.98, 1.0))
    safe_savefig(fig, output_root / "外环PID有无_水平误差响应曲线.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True)
    axes[0].plot(
        [row["rel_time_s"] for row in no_pid_rows],
        [row["roll_deg"] for row in no_pid_rows],
        linewidth=2.0,
        label="无PID外环",
    )
    axes[0].plot(
        [row["rel_time_s"] for row in pid_rows],
        [row["roll_deg"] for row in pid_rows],
        linewidth=2.0,
        label="有PID外环（基准PID组）",
    )
    axes[0].set_ylabel("横滚角绝对值 (deg)")
    axes[0].set_title("横滚响应")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        [row["rel_time_s"] for row in no_pid_rows],
        [row["pitch_deg"] for row in no_pid_rows],
        linewidth=2.0,
        label="无PID外环",
    )
    axes[1].plot(
        [row["rel_time_s"] for row in pid_rows],
        [row["pitch_deg"] for row in pid_rows],
        linewidth=2.0,
        label="有PID外环（基准PID组）",
    )
    axes[1].set_ylabel("俯仰角绝对值 (deg)")
    axes[1].set_xlabel("任务开始后的时间 (s)")
    axes[1].set_title("俯仰响应")
    axes[1].grid(True, alpha=0.3)

    fig.text(
        0.72,
        0.52,
        summary_text,
        ha="left",
        va="center",
        fontsize=9.0,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#bbbbbb"},
    )
    fig.suptitle("外环PID有无情况下的姿态响应曲线", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
    safe_savefig(fig, output_root / "外环PID有无_姿态响应曲线.png")
    plt.close(fig)

    write_summary_csv(output_root / "外环PID有无_结果汇总.csv", metrics_by_group)
    print(f"saved plots to {output_root}", flush=True)


if __name__ == "__main__":
    main()
