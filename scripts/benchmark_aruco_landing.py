#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
from pathlib import Path


TBC_ROT = (
    (0.0, -1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
)

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/aruco_benchmark")


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def quat_to_matrix(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def quat_to_euler_deg(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_to_roll_pitch_deg(qx, qy, qz, qw):
    roll, pitch, _yaw = quat_to_euler_deg(qx, qy, qz, qw)
    return roll, pitch


def mat_vec_mul(matrix, vector):
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def run_command(cmd, env=None, stdout=None):
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
    )


def cleanup_processes():
    patterns = [
        "roslaunch px4 aruco_search_and_land_demo.launch",
        "roslaunch px4 aruco_search_and_land_benchmark.launch",
        "aruco_search_and_detect_no_pid.py",
        "aruco_search_and_detect_baseline.py",
        "aruco_search_and_detect.py",
        "aruco_multi_marker_det.py",
        "aruco_multi_marker_det_weighted.py",
        "px4_sitl_default/bin/px4",
        "gzserver",
        "gzclient",
        "mavros_node",
        "rosmaster",
    ]
    protected_pids = {os.getpid(), os.getppid()}
    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        for raw_pid in result.stdout.splitlines():
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid in protected_pids:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
    sock_path = Path("/tmp/px4-sock-0")
    if sock_path.exists():
        sock_path.unlink()
    time.sleep(3.0)


def wait_for_file(path, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.5)
    return False


def wait_for_ros_topics(env, topics, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["rostopic", "list"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            listed_topics = set(result.stdout.splitlines())
            if all(topic in listed_topics for topic in topics):
                return True
        time.sleep(1.0)
    return False


def terminate_process(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def format_launch_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_launch_args(items):
    launch_args = {}
    for item in items:
        if ":=" in item:
            key, value = item.split(":=", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            raise ValueError(f"launch arg must look like name=value or name:=value: {item}")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"launch arg key must not be empty: {item}")
        launch_args[key] = value
    return launch_args


def parse_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = [
            line
            for line in handle
            if line.startswith("%time") or (line and line[0].isdigit())
        ]
        if not lines:
            return []
        reader = csv.DictReader(lines)
        return list(reader)


def pose_samples(rows):
    samples = []
    for row in rows:
        stamp = parse_timestamp(row.get("field.header.stamp") or row.get("%time"))
        try:
            if stamp is None:
                continue
            samples.append(
                {
                    "time": stamp,
                    "x": float(row["field.pose.position.x"]),
                    "y": float(row["field.pose.position.y"]),
                    "z": float(row["field.pose.position.z"]),
                    "qx": float(row["field.pose.orientation.x"]),
                    "qy": float(row["field.pose.orientation.y"]),
                    "qz": float(row["field.pose.orientation.z"]),
                    "qw": float(row["field.pose.orientation.w"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return samples


def nearest_pose(samples, stamp):
    if not samples:
        return None
    best = min(samples, key=lambda sample: abs(sample["time"] - stamp))
    if abs(best["time"] - stamp) > 0.08:
        return None
    return best


def parse_timestamp(value):
    if value in (None, ""):
        return None
    stamp = float(value)
    if abs(stamp) > 1e6:
        stamp /= 1e9
    return stamp


def compute_marker_world_xy(local_pose, aruco_pose):
    rotation_wb = quat_to_matrix(local_pose["qx"], local_pose["qy"], local_pose["qz"], local_pose["qw"])
    t_ca = (aruco_pose["x"], aruco_pose["y"], aruco_pose["z"])
    t_ba = mat_vec_mul(TBC_ROT, t_ca)
    t_wa_offset = mat_vec_mul(rotation_wb, t_ba)
    marker_x = local_pose["x"] + t_wa_offset[0]
    marker_y = local_pose["y"] + t_wa_offset[1]
    return marker_x, marker_y


def compute_marker_error(local_pose, aruco_pose):
    marker_x, marker_y = compute_marker_world_xy(local_pose, aruco_pose)
    return math.hypot(local_pose["x"] - marker_x, local_pose["y"] - marker_y)


def parse_log_times(log_path):
    phases = {}
    if not log_path.exists():
        return phases

    markers = [
        ("switching to TAKEOFF", "takeoff_start"),
        ("takeoff complete, switching to SEARCH", "search_start"),
        ("switching to TRACK", "track_start"),
        ("switching to LAND_ALIGN", "land_align_start"),
        ("landing target locked", "descent_start"),
        ("landing alignment complete, switching to AUTO_LAND", "auto_land_start"),
        ("center locked and descent complete, switching to AUTO_LAND", "auto_land_start"),
        ("landing complete, switching to SUCCESS", "success"),
    ]

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for text, key in markers:
                if text in line and key not in phases:
                    parsed_time = extract_log_sim_time(line)
                    if parsed_time is not None:
                        phases[key] = parsed_time
    return phases


def extract_log_sim_time(line):
    try:
        bracket_content = line.split("[")[2].split("]")[0]
    except IndexError:
        return None
    parts = [part.strip() for part in bracket_content.split(",")]
    for candidate in reversed(parts):
        try:
            return float(candidate)
        except ValueError:
            continue
    return None


def summarize_run(local_csv, aruco_csv, setpoint_csv, result_json, log_path, target_xy, run_dir=None):
    local_rows = pose_samples(parse_csv(local_csv))
    setpoint_rows = pose_samples(parse_csv(setpoint_csv))
    aruco_rows_raw = parse_csv(aruco_csv)
    aruco_rows = []
    for row in aruco_rows_raw:
        stamp = parse_timestamp(row.get("field.header.stamp") or row.get("%time"))
        try:
            if stamp is None:
                continue
            aruco_rows.append(
                {
                    "time": stamp,
                    "x": float(row["field.pose.position.x"]),
                    "y": float(row["field.pose.position.y"]),
                    "z": float(row["field.pose.position.z"]),
                }
            )
        except (KeyError, ValueError):
            continue

    result = {}
    if result_json.exists():
        result = json.loads(result_json.read_text(encoding="utf-8"))

    phases = parse_log_times(log_path)

    home_z = local_rows[0]["z"] if local_rows else 0.0
    final_pose = None
    if result:
        vehicle_position = result.get("vehicle_position", {})
        try:
            final_pose = {
                "x": float(vehicle_position["x"]),
                "y": float(vehicle_position["y"]),
                "z": float(vehicle_position["z"]),
            }
        except (KeyError, TypeError, ValueError):
            final_pose = None
    if final_pose is None and local_rows:
        final_pose = local_rows[-1]

    final_x_error = None
    final_y_error = None
    final_target_error = None
    if final_pose is not None:
        final_x_error = final_pose["x"] - target_xy[0]
        final_y_error = final_pose["y"] - target_xy[1]
        final_target_error = math.hypot(final_x_error, final_y_error)

    initial_landing_height = None
    initial_lateral_error = None
    if local_rows:
        initial_sample = local_rows[0]
        if "land_align_start" in phases:
            initial_sample = nearest_pose(local_rows, phases["land_align_start"]) or initial_sample
        initial_landing_height = initial_sample["z"] - home_z
        initial_lateral_error = math.hypot(initial_sample["x"] - target_xy[0], initial_sample["y"] - target_xy[1])

    final_descent_window = [sample for sample in local_rows if 0.45 <= (sample["z"] - home_z) <= 0.9]
    xy_speeds = []
    for prev, cur in zip(final_descent_window, final_descent_window[1:]):
        dt = cur["time"] - prev["time"]
        if dt <= 1e-3:
            continue
        xy_speeds.append(math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"]) / dt)

    final_descent_errors = []
    final_descent_localization_errors = []
    for detection in aruco_rows:
        pose = nearest_pose(local_rows, detection["time"])
        if pose is None:
            continue
        agl = pose["z"] - home_z
        if 0.45 <= agl <= 0.9:
            final_descent_errors.append(compute_marker_error(pose, detection))
            marker_x, marker_y = compute_marker_world_xy(pose, detection)
            final_descent_localization_errors.append(
                math.hypot(marker_x - target_xy[0], marker_y - target_xy[1])
            )

    final_marker_estimate_error = None
    if result:
        final_marker_estimate_error = math.hypot(
            result["vehicle_position"]["x"] - result["marker_world_estimate"]["x"],
            result["vehicle_position"]["y"] - result["marker_world_estimate"]["y"],
        )
    elif aruco_rows:
        last_detection = aruco_rows[-1]
        last_pose = nearest_pose(local_rows, last_detection["time"])
        if last_pose is not None:
            final_marker_estimate_error = compute_marker_error(last_pose, last_detection)

    mission_time = None
    if "takeoff_start" in phases and "success" in phases:
        mission_time = phases["success"] - phases["takeoff_start"]

    run_duration = None
    if local_rows:
        run_duration = local_rows[-1]["time"] - local_rows[0]["time"]

    search_time = None
    if "search_start" in phases and "track_start" in phases:
        search_time = phases["track_start"] - phases["search_start"]

    align_time = None
    if "land_align_start" in phases and "success" in phases:
        align_time = phases["success"] - phases["land_align_start"]

    landing_time = align_time

    land_align_window = []
    land_align_errors = []
    localization_errors = []
    if "land_align_start" in phases:
        land_align_end = phases.get("success")
        land_align_window = [
            sample
            for sample in local_rows
            if sample["time"] >= phases["land_align_start"]
            and (land_align_end is None or sample["time"] <= land_align_end)
        ]
        for detection in aruco_rows:
            if detection["time"] < phases["land_align_start"]:
                continue
            if land_align_end is not None and detection["time"] > land_align_end:
                continue
            pose = nearest_pose(local_rows, detection["time"])
            if pose is None:
                continue
            land_align_errors.append(compute_marker_error(pose, detection))
            marker_x, marker_y = compute_marker_world_xy(pose, detection)
            localization_errors.append(
                math.hypot(marker_x - target_xy[0], marker_y - target_xy[1])
            )

    land_align_xy_speeds = []
    for prev, cur in zip(land_align_window, land_align_window[1:]):
        dt = cur["time"] - prev["time"]
        if dt <= 1e-3:
            continue
        land_align_xy_speeds.append(math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"]) / dt)

    attitude_roll_abs = []
    attitude_pitch_abs = []
    for sample in land_align_window:
        roll_deg, pitch_deg = quat_to_roll_pitch_deg(
            sample["qx"],
            sample["qy"],
            sample["qz"],
            sample["qw"],
        )
        attitude_roll_abs.append(abs(roll_deg))
        attitude_pitch_abs.append(abs(pitch_deg))

    response_latencies = compute_response_latencies(aruco_rows, setpoint_rows, phases.get("land_align_start"))

    metrics = {
        "success": bool(result),
        "mission_time_s": mission_time,
        "run_duration_s": run_duration,
        "search_time_s": search_time,
        "align_to_land_time_s": align_time,
        "initial_landing_height_m": initial_landing_height,
        "initial_lateral_error_m": initial_lateral_error,
        "landing_time_s": landing_time,
        "final_x_error_m": final_x_error,
        "final_y_error_m": final_y_error,
        "final_error_m": final_target_error,
        "final_marker_estimate_error_m": final_marker_estimate_error,
        "localization_error_mean_m": statistics.mean(localization_errors) if localization_errors else None,
        "localization_error_p95_m": percentile(localization_errors, 95) if localization_errors else None,
        "localization_error_max_m": max(localization_errors) if localization_errors else None,
        "localization_within_5cm_ratio": ratio_within(localization_errors, 0.05),
        "land_align_error_mean_m": statistics.mean(land_align_errors) if land_align_errors else None,
        "land_align_error_p95_m": percentile(land_align_errors, 95) if land_align_errors else None,
        "land_align_xy_speed_rms_mps": rms(land_align_xy_speeds) if land_align_xy_speeds else None,
        "land_align_xy_speed_p95_mps": percentile(land_align_xy_speeds, 95) if land_align_xy_speeds else None,
        "final_descent_error_mean_m": statistics.mean(final_descent_errors) if final_descent_errors else None,
        "final_descent_error_p95_m": percentile(final_descent_errors, 95) if final_descent_errors else None,
        "final_descent_localization_error_mean_m": (
            statistics.mean(final_descent_localization_errors) if final_descent_localization_errors else None
        ),
        "final_descent_localization_error_p95_m": (
            percentile(final_descent_localization_errors, 95) if final_descent_localization_errors else None
        ),
        "final_descent_xy_speed_rms_mps": rms(xy_speeds) if xy_speeds else None,
        "final_descent_xy_speed_p95_mps": percentile(xy_speeds, 95) if xy_speeds else None,
        "roll_abs_max_deg": max(attitude_roll_abs) if attitude_roll_abs else None,
        "roll_abs_p95_deg": percentile(attitude_roll_abs, 95) if attitude_roll_abs else None,
        "pitch_abs_max_deg": max(attitude_pitch_abs) if attitude_pitch_abs else None,
        "pitch_abs_p95_deg": percentile(attitude_pitch_abs, 95) if attitude_pitch_abs else None,
        "attitude_within_2deg_ratio": attitude_within_ratio(attitude_roll_abs, attitude_pitch_abs, 2.0),
        "response_latency_mean_s": statistics.mean(response_latencies) if response_latencies else None,
        "response_latency_p95_s": percentile(response_latencies, 95) if response_latencies else None,
        "response_latency_max_s": max(response_latencies) if response_latencies else None,
        "aruco_samples_land_align": len(land_align_errors),
        "pose_samples_land_align": len(land_align_window),
        "aruco_samples_final_descent": len(final_descent_errors),
        "pose_samples_final_descent": len(final_descent_window),
    }

    if run_dir is not None:
        write_run_timeseries_and_plots(run_dir, local_rows, phases, target_xy, home_z)

    return metrics


def write_run_timeseries_and_plots(run_dir, local_rows, phases, target_xy, home_z):
    timeseries_path = run_dir / "timeseries.csv"
    rows = build_timeseries_rows(local_rows, phases, target_xy, home_z)
    with timeseries_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "time_s",
            "rel_time_s",
            "phase",
            "x_m",
            "y_m",
            "z_m",
            "x_error_m",
            "y_error_m",
            "horizontal_error_m",
            "altitude_m",
            "vz_mps",
            "descent_speed_mps",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_run_plots(run_dir, rows)


def build_timeseries_rows(local_rows, phases, target_xy, home_z):
    if not local_rows:
        return []

    phase_sequence = [
        ("TAKEOFF", phases.get("takeoff_start")),
        ("SEARCH", phases.get("search_start")),
        ("TRACK", phases.get("track_start")),
        ("LAND_ALIGN", phases.get("land_align_start")),
        ("DESCEND", phases.get("descent_start")),
        ("AUTO_LAND", phases.get("auto_land_start")),
        ("SUCCESS", phases.get("success")),
    ]
    phase_sequence = [(name, stamp) for name, stamp in phase_sequence if stamp is not None]
    phase_sequence.sort(key=lambda item: item[1])

    rows = []
    start_time = local_rows[0]["time"]
    previous = None
    for sample in local_rows:
        roll_deg, pitch_deg, yaw_deg = quat_to_euler_deg(
            sample["qx"],
            sample["qy"],
            sample["qz"],
            sample["qw"],
        )
        vz_mps = 0.0
        if previous is not None:
            dt = sample["time"] - previous["time"]
            if dt > 1e-3:
                vz_mps = (sample["z"] - previous["z"]) / dt

        x_error = sample["x"] - target_xy[0]
        y_error = sample["y"] - target_xy[1]
        rows.append(
            {
                "time_s": sample["time"],
                "rel_time_s": sample["time"] - start_time,
                "phase": phase_at(sample["time"], phase_sequence),
                "x_m": sample["x"],
                "y_m": sample["y"],
                "z_m": sample["z"],
                "x_error_m": x_error,
                "y_error_m": y_error,
                "horizontal_error_m": math.hypot(x_error, y_error),
                "altitude_m": sample["z"] - home_z,
                "vz_mps": vz_mps,
                "descent_speed_mps": max(0.0, -vz_mps),
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
            }
        )
        previous = sample
    return rows


def phase_at(stamp, phase_sequence):
    current = "INIT"
    for phase_name, phase_start in phase_sequence:
        if stamp >= phase_start:
            current = phase_name
        else:
            break
    return current


def write_run_plots(run_dir, rows):
    if not rows:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    time_values = [row["rel_time_s"] for row in rows]

    plt.figure(figsize=(9, 4.8))
    plt.plot(time_values, [row["x_error_m"] for row in rows], label="x error")
    plt.plot(time_values, [row["y_error_m"] for row in rows], label="y error")
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-0.05, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (m)")
    plt.title("Lateral Position Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "xy_error_curve.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time_values, [row["altitude_m"] for row in rows], label="altitude")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Altitude")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_values, [row["vz_mps"] for row in rows], label="vertical velocity")
    axes[1].plot(time_values, [row["descent_speed_mps"] for row in rows], label="descent speed")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Vertical Velocity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "height_vertical_speed_curve.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(9, 4.8))
    plt.plot(time_values, [row["roll_deg"] for row in rows], label="roll")
    plt.plot(time_values, [row["pitch_deg"] for row in rows], label="pitch")
    plt.plot(time_values, [row["yaw_deg"] for row in rows], label="yaw")
    plt.axhline(2.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-2.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Attitude Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "attitude_curve.png", dpi=150)
    plt.close()


def compute_response_latencies(aruco_rows, setpoint_rows, start_time):
    if not aruco_rows or not setpoint_rows:
        return []

    latencies = []
    setpoint_index = 0
    for detection in aruco_rows:
        detection_time = detection["time"]
        if start_time is not None and detection_time < start_time:
            continue
        while setpoint_index < len(setpoint_rows) and setpoint_rows[setpoint_index]["time"] < detection_time:
            setpoint_index += 1
        if setpoint_index >= len(setpoint_rows):
            break
        latency = setpoint_rows[setpoint_index]["time"] - detection_time
        if 0.0 <= latency <= 5.0:
            latencies.append(latency)
    return latencies


def ratio_within(values, threshold):
    if not values:
        return None
    return sum(1 for value in values if value <= threshold) / len(values)


def attitude_within_ratio(roll_abs_values, pitch_abs_values, threshold):
    if not roll_abs_values or not pitch_abs_values:
        return None
    paired = zip(roll_abs_values, pitch_abs_values)
    total = min(len(roll_abs_values), len(pitch_abs_values))
    if total <= 0:
        return None
    return sum(1 for roll_abs, pitch_abs in paired if roll_abs <= threshold and pitch_abs <= threshold) / total


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rms(values):
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def aggregate(runs):
    successes = sum(1 for run in runs if run["success"])
    summary = {
        "runs": len(runs),
        "successes": successes,
        "success_rate": (successes / len(runs)) if runs else None,
    }
    keys = [
        "mission_time_s",
        "run_duration_s",
        "search_time_s",
        "align_to_land_time_s",
        "landing_time_s",
        "initial_landing_height_m",
        "initial_lateral_error_m",
        "final_x_error_m",
        "final_y_error_m",
        "final_error_m",
        "final_marker_estimate_error_m",
        "localization_error_mean_m",
        "localization_error_p95_m",
        "localization_error_max_m",
        "localization_within_5cm_ratio",
        "land_align_error_mean_m",
        "land_align_error_p95_m",
        "land_align_xy_speed_rms_mps",
        "land_align_xy_speed_p95_mps",
        "final_descent_error_mean_m",
        "final_descent_error_p95_m",
        "final_descent_localization_error_mean_m",
        "final_descent_localization_error_p95_m",
        "final_descent_xy_speed_rms_mps",
        "final_descent_xy_speed_p95_mps",
        "roll_abs_max_deg",
        "roll_abs_p95_deg",
        "pitch_abs_max_deg",
        "pitch_abs_p95_deg",
        "attitude_within_2deg_ratio",
        "response_latency_mean_s",
        "response_latency_p95_s",
        "response_latency_max_s",
    ]
    for key in keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        if not values:
            summary[key] = None
            continue
        summary[key] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    return summary


def write_landing_results_table(results, output_root):
    output_path = output_root / "landing_results.csv"
    output_path_cn = output_root / "landing_results_cn.csv"
    fieldnames = [
        "case",
        "run",
        "initial_height_m",
        "initial_lateral_error_m",
        "landing_time_s",
        "final_x_error_m",
        "final_y_error_m",
        "final_landing_error_m",
        "success",
        "localization_p95_m",
        "roll_abs_max_deg",
        "pitch_abs_max_deg",
        "response_latency_p95_s",
    ]
    cn_fieldnames = [
        "试验编号",
        "实验组",
        "初始高度_m",
        "初始横向偏差_m",
        "着陆时间_s",
        "最终x误差_m",
        "最终y误差_m",
        "综合着陆误差_m",
        "是否成功",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        english_rows = []
        chinese_rows = []
        for case_name, case_result in results.items():
            for run in case_result.get("runs", []):
                english_row = {
                    "case": case_name,
                    "run": run.get("run"),
                    "initial_height_m": run.get("initial_landing_height_m"),
                    "initial_lateral_error_m": run.get("initial_lateral_error_m"),
                    "landing_time_s": run.get("landing_time_s"),
                    "final_x_error_m": run.get("final_x_error_m"),
                    "final_y_error_m": run.get("final_y_error_m"),
                    "final_landing_error_m": run.get("final_error_m"),
                    "success": run.get("success"),
                    "localization_p95_m": run.get("final_descent_localization_error_p95_m"),
                    "roll_abs_max_deg": run.get("roll_abs_max_deg"),
                    "pitch_abs_max_deg": run.get("pitch_abs_max_deg"),
                    "response_latency_p95_s": run.get("response_latency_p95_s"),
                }
                english_rows.append(english_row)
                chinese_rows.append(
                    {
                        "试验编号": run.get("run"),
                        "实验组": case_name,
                        "初始高度_m": run.get("initial_landing_height_m"),
                        "初始横向偏差_m": run.get("initial_lateral_error_m"),
                        "着陆时间_s": run.get("landing_time_s"),
                        "最终x误差_m": run.get("final_x_error_m"),
                        "最终y误差_m": run.get("final_y_error_m"),
                        "综合着陆误差_m": run.get("final_error_m"),
                        "是否成功": "是" if run.get("success") else "否",
                    }
                )
        writer.writerows(english_rows)
    with output_path_cn.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cn_fieldnames)
        writer.writeheader()
        writer.writerows(chinese_rows)
    return output_path


def benchmark_controller(
    label,
    controller_type,
    runs,
    timeout_s,
    env,
    output_root,
    world,
    marker_config,
    detector_type,
    target_xy,
    launch_args=None,
):
    controller_dir = output_root / label
    controller_dir.mkdir(parents=True, exist_ok=True)
    all_runs = []
    launch_args = dict(launch_args or {})

    for run_index in range(1, runs + 1):
        cleanup_processes()
        run_dir = controller_dir / f"run_{run_index:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        launch_log = run_dir / "launch.log"
        local_csv = run_dir / "local_pose.csv"
        aruco_csv = run_dir / "aruco_pose.csv"
        setpoint_csv = run_dir / "setpoint_pose.csv"
        result_json = run_dir / "result.json"
        ros_home = run_dir / "ros_home"
        ros_home.mkdir(parents=True, exist_ok=True)

        launch_cmd = [
            "roslaunch",
            "px4",
            "aruco_search_and_land_benchmark.launch",
            "gui:=false",
            f"controller_type:={controller_type}",
            f"world:={world}",
            f"marker_config_path:={marker_config}",
            f"detector_type:={detector_type}",
            f"success_result_path:={result_json}",
        ]
        for key, value in sorted(launch_args.items()):
            launch_cmd.append(f"{key}:={format_launch_value(value)}")
        launch_env = env.copy()
        launch_env["ROS_HOME"] = str(ros_home)

        with launch_log.open("w", encoding="utf-8") as log_handle:
            launch_proc = run_command(launch_cmd, env=launch_env, stdout=log_handle)
            ros_ready = wait_for_ros_topics(
                launch_env,
                ["/mavros/local_position/pose", "/aruco/pose", "/mavros/setpoint_position/local"],
                timeout_s=60.0,
            )
            if ros_ready:
                with local_csv.open("w", encoding="utf-8") as local_handle, aruco_csv.open(
                    "w", encoding="utf-8"
                ) as aruco_handle, setpoint_csv.open("w", encoding="utf-8") as setpoint_handle:
                    local_proc = run_command(
                        ["rostopic", "echo", "-p", "/mavros/local_position/pose"],
                        env=launch_env,
                        stdout=local_handle,
                    )
                    aruco_proc = run_command(
                        ["rostopic", "echo", "-p", "/aruco/pose"],
                        env=launch_env,
                        stdout=aruco_handle,
                    )
                    setpoint_proc = run_command(
                        ["rostopic", "echo", "-p", "/mavros/setpoint_position/local"],
                        env=launch_env,
                        stdout=setpoint_handle,
                    )

                    success = wait_for_file(result_json, timeout_s)

                    terminate_process(local_proc)
                    terminate_process(aruco_proc)
                    terminate_process(setpoint_proc)
            else:
                success = False

            terminate_process(launch_proc)

        metrics = summarize_run(local_csv, aruco_csv, setpoint_csv, result_json, launch_log, target_xy, run_dir)
        metrics["run"] = run_index
        metrics["success"] = metrics["success"] and success
        all_runs.append(metrics)
        print(f"{label} run {run_index}: {json.dumps(metrics, ensure_ascii=False)}", flush=True)

    return {
        "controller_type": controller_type,
        "detector_type": detector_type,
        "launch_args": {key: format_launch_value(value) for key, value in sorted(launch_args.items())},
        "runs": all_runs,
        "summary": aggregate(all_runs),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ArUco landing controllers in SITL")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default=None, help="Run a single named case instead of baseline+optimized")
    parser.add_argument("--controller", default=None, help="Controller node script for --label")
    parser.add_argument(
        "--world",
        default="$(find mavlink_sitl_gazebo)/worlds/aruco_search_demo.world",
        help="Gazebo world passed to aruco_search_and_land_benchmark.launch",
    )
    parser.add_argument(
        "--marker-config",
        default="$(find px4)/config/aruco_nested_board.yaml",
        help="Marker YAML loaded by the detector",
    )
    parser.add_argument("--detector-type", default="aruco_multi_marker_det.py")
    parser.add_argument("--target-x", type=float, default=1.5, help="Landing marker center in PX4 local frame")
    parser.add_argument("--target-y", type=float, default=1.5, help="Landing marker center in PX4 local frame")
    parser.add_argument("--output-root", default=str(TMP_ROOT))
    parser.add_argument(
        "--launch-arg",
        action="append",
        default=[],
        help="Additional roslaunch args passed to aruco_search_and_land_benchmark.launch, e.g. track_pid_kp=0.8",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    target_xy = (args.target_x, args.target_y)
    launch_args = parse_launch_args(args.launch_arg)

    if args.label or args.controller:
        if not args.label or not args.controller:
            raise SystemExit("--label and --controller must be used together")
        results = {
            args.label: benchmark_controller(
                args.label,
                args.controller,
                args.runs,
                args.timeout,
                env,
                output_root,
                args.world,
                args.marker_config,
                args.detector_type,
                target_xy,
                launch_args=launch_args,
            )
        }
    else:
        results = {
            "baseline": benchmark_controller(
                "baseline",
                "aruco_search_and_detect_baseline.py",
                args.runs,
                args.timeout,
                env,
                output_root,
                args.world,
                args.marker_config,
                args.detector_type,
                target_xy,
                launch_args=launch_args,
            ),
            "optimized": benchmark_controller(
                "optimized",
                "aruco_search_and_detect.py",
                args.runs,
                args.timeout,
                env,
                output_root,
                args.world,
                args.marker_config,
                args.detector_type,
                target_xy,
                launch_args=launch_args,
            ),
        }

    output_path = output_root / "summary.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    landing_results_path = write_landing_results_table(results, output_root)
    print(f"summary saved to {output_path}", flush=True)
    print(f"landing results saved to {landing_results_path}", flush=True)

    cleanup_processes()


if __name__ == "__main__":
    main()
