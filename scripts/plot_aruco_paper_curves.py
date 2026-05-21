#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def main():
    parser = argparse.ArgumentParser(description="Plot XY trajectory, Z-T, and error convergence from benchmark timeseries.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing timeseries.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    timeseries_path = run_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise SystemExit(f"missing file: {timeseries_path}")

    rows = load_rows(timeseries_path)
    if not rows:
        raise SystemExit(f"empty file: {timeseries_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    plots_dir = run_dir / "paper_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    time_values = [parse_float(row, "rel_time_s") for row in rows]
    x_values = [parse_float(row, "x_m") for row in rows]
    y_values = [parse_float(row, "y_m") for row in rows]
    altitude_values = [parse_float(row, "altitude_m") for row in rows]
    error_values = [parse_float(row, "horizontal_error_m") for row in rows]

    target_x = None
    target_y = None
    for row in rows:
        x_m = parse_float(row, "x_m")
        y_m = parse_float(row, "y_m")
        x_err = parse_float(row, "x_error_m")
        y_err = parse_float(row, "y_error_m")
        if None not in (x_m, y_m, x_err, y_err):
            target_x = x_m - x_err
            target_y = y_m - y_err
            break

    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_values, linewidth=1.8, label="UAV trajectory")
    if target_x is not None and target_y is not None:
        plt.scatter([target_x], [target_y], marker="x", s=80, linewidths=2.0, label="Target center")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Landing XY Trajectory")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "landing_xy_trajectory.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(time_values, altitude_values, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude-Time Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "height_time_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(time_values, error_values, linewidth=1.8)
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal error (m)")
    plt.title("Error Convergence Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_convergence_curve.png", dpi=150)
    plt.close()

    print(f"saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
