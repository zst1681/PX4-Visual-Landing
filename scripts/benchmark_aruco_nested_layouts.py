#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/tmp/aruco_nested_layouts")
CORNER_MARKERS = (
    (31, "outer_marker_ne", 1.0, 1.0),
    (32, "outer_marker_nw", -1.0, 1.0),
    (33, "outer_marker_sw", -1.0, -1.0),
    (34, "outer_marker_se", 1.0, -1.0),
)
CENTER_MARKER_ID = 35
ALL_MARKER_IDS = {31, 32, 33, 34, 35}
CORNER_MARKER_IDS = {31, 32, 33, 34}


@dataclass(frozen=True)
class LayoutCase:
    name: str
    board_size: float
    corner_length: float
    center_length: float
    corner_offset: float
    edge_margin: float

    def marker_configs(self):
        configs = []
        for priority, (marker_id, _visual_name, sign_x, sign_y) in enumerate(CORNER_MARKERS, start=1):
            configs.append(
                {
                    "id": marker_id,
                    "length": self.corner_length,
                    "offset_x": sign_x * self.corner_offset,
                    "offset_y": sign_y * self.corner_offset,
                    "priority": priority,
                    "role": "corner",
                }
            )
        configs.append(
            {
                "id": CENTER_MARKER_ID,
                "length": self.center_length,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "priority": 0,
                "role": "center",
            }
        )
        return configs


@dataclass(frozen=True)
class Scenario:
    phase: str
    camera_x: float
    camera_y: float
    height: float
    yaw_rad: float
    tilt_x_rad: float
    tilt_y_rad: float
    contrast: float
    brightness: float
    blur_sigma: float
    noise_sigma: float
    noise_seed: int


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def load_camera_params(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    camera_matrix = np.array(
        [
            [float(config["fx"]), 0.0, float(config["x0"])],
            [0.0, float(config["fy"]), float(config["y0"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(
        [
            [float(config.get("k1", 0.0))],
            [float(config.get("k2", 0.0))],
            [float(config.get("p1", 0.0))],
            [float(config.get("p2", 0.0))],
            [float(config.get("k3", 0.0))],
        ],
        dtype=np.float64,
    )
    return camera_matrix, dist_coeffs


def create_detector_params():
    params = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 40
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    return params


def marker_object_points(marker_length, center_x=0.0, center_y=0.0):
    half = marker_length * 0.5
    return np.array(
        [
            [center_x - half, center_y + half, 0.0],
            [center_x + half, center_y + half, 0.0],
            [center_x + half, center_y - half, 0.0],
            [center_x - half, center_y - half, 0.0],
        ],
        dtype=np.float64,
    )


def create_board(layout, dictionary):
    board_create = getattr(cv2.aruco, "Board_create", None)
    if board_create is None:
        return None
    board_points = [
        marker_object_points(
            config["length"],
            config["offset_x"],
            config["offset_y"],
        ).astype(np.float32)
        for config in layout.marker_configs()
    ]
    board_ids = np.array([[config["id"]] for config in layout.marker_configs()], dtype=np.int32)
    return board_create(board_points, dictionary, board_ids)


def draw_marker(dictionary, marker_id, marker_px):
    marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_px, marker, 1)
    return marker


def make_board_texture(layout, dictionary, pixels_per_meter):
    texture_px = int(round(layout.board_size * pixels_per_meter))
    texture = np.full((texture_px, texture_px), 255, dtype=np.uint8)

    for config in layout.marker_configs():
        marker_px = max(12, int(round(config["length"] * pixels_per_meter)))
        marker = draw_marker(dictionary, config["id"], marker_px)
        center_x_px = int(round((config["offset_x"] + layout.board_size * 0.5) * pixels_per_meter))
        center_y_px = int(round((layout.board_size * 0.5 - config["offset_y"]) * pixels_per_meter))
        left = center_x_px - marker_px // 2
        top = center_y_px - marker_px // 2
        right = left + marker_px
        bottom = top + marker_px
        if left < 0 or top < 0 or right > texture_px or bottom > texture_px:
            raise ValueError(f"marker {config['id']} is outside the board in {layout.name}")
        texture[top:bottom, left:right] = marker

    return texture


def rot_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def camera_to_world_rotation(scenario):
    yaw = scenario.yaw_rad
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-9)
    base = np.column_stack((x_axis, y_axis, z_axis))
    local_tilt = rot_x(scenario.tilt_x_rad).dot(rot_y(scenario.tilt_y_rad))
    return base.dot(local_tilt)


def project_points(world_points, camera_matrix, scenario):
    camera_center = np.array([scenario.camera_x, scenario.camera_y, scenario.height], dtype=np.float64)
    rotation_cw = camera_to_world_rotation(scenario).T
    camera_points = rotation_cw.dot((world_points - camera_center).T).T
    if np.any(camera_points[:, 2] <= 0.03):
        return None, camera_points

    normalized = camera_points[:, :2] / camera_points[:, 2:3]
    image_points = np.empty((world_points.shape[0], 2), dtype=np.float64)
    image_points[:, 0] = camera_matrix[0, 0] * normalized[:, 0] + camera_matrix[0, 2]
    image_points[:, 1] = camera_matrix[1, 1] * normalized[:, 1] + camera_matrix[1, 2]
    return image_points, camera_points


def render_scenario(texture, layout, scenario, camera_matrix, image_size):
    width, height = image_size
    half = layout.board_size * 0.5
    board_corners = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    image_corners, _camera_points = project_points(board_corners, camera_matrix, scenario)
    if image_corners is None:
        return np.full((height, width), 205, dtype=np.uint8)

    texture_height, texture_width = texture.shape[:2]
    texture_corners = np.array(
        [
            [0.0, 0.0],
            [texture_width - 1.0, 0.0],
            [texture_width - 1.0, texture_height - 1.0],
            [0.0, texture_height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(texture_corners, image_corners.astype(np.float32))
    frame = cv2.warpPerspective(
        texture,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=205,
    )

    frame = frame.astype(np.float32) * scenario.contrast + scenario.brightness
    if scenario.blur_sigma > 0.05:
        kernel_size = max(3, int(math.ceil(scenario.blur_sigma * 4.0)) | 1)
        frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), scenario.blur_sigma)
    if scenario.noise_sigma > 0.0:
        rng = np.random.default_rng(scenario.noise_seed)
        noise = rng.normal(0.0, scenario.noise_sigma, frame.shape).astype(np.float32)
        frame += noise

    return np.clip(frame, 0, 255).astype(np.uint8)


def board_reprojection_error(layout, marker_ids, marker_corners, rvec, tvec, camera_matrix, dist_coeffs):
    config_by_id = {config["id"]: config for config in layout.marker_configs()}
    residuals = []
    for index, marker_id in enumerate(marker_ids.flatten()):
        config = config_by_id.get(int(marker_id))
        if config is None:
            continue
        object_points = marker_object_points(config["length"], config["offset_x"], config["offset_y"])
        projected_corners, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected_corners = projected_corners.reshape(-1, 2)
        observed_corners = marker_corners[index].reshape(-1, 2)
        residuals.append(np.sum((projected_corners - observed_corners) ** 2, axis=1))

    if not residuals:
        return float("inf")
    residuals = np.concatenate(residuals)
    return float(np.sqrt(np.mean(residuals)))


def evaluate_detection(
    frame,
    layout,
    dictionary,
    detector_params,
    camera_matrix,
    dist_coeffs,
    board,
    max_board_reprojection_error,
):
    marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
        frame,
        dictionary,
        parameters=detector_params,
    )
    if marker_ids is None or len(marker_ids) == 0:
        return {
            "detected_ids": set(),
            "center_hit": False,
            "corner_count": 0,
            "board_pose_hit": False,
            "nested_hit": False,
            "all_hit": False,
            "board_reprojection_error_px": None,
        }

    detected_ids = {int(marker_id) for marker_id in marker_ids.flatten() if int(marker_id) in ALL_MARKER_IDS}
    center_hit = CENTER_MARKER_ID in detected_ids
    corner_count = len(detected_ids.intersection(CORNER_MARKER_IDS))
    board_pose_hit = False
    reprojection_error = None

    if board is not None and len(detected_ids) >= 2 and hasattr(cv2.aruco, "estimatePoseBoard"):
        known_indices = [
            index
            for index, marker_id in enumerate(marker_ids.flatten())
            if int(marker_id) in ALL_MARKER_IDS
        ]
        known_corners = [marker_corners[index] for index in known_indices]
        known_ids = np.array([[int(marker_ids[index][0])] for index in known_indices], dtype=np.int32)
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            known_corners,
            known_ids,
            board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )
        if retval > 0:
            reprojection_error = board_reprojection_error(
                layout,
                known_ids,
                known_corners,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            board_pose_hit = reprojection_error <= max_board_reprojection_error

    nested_hit = center_hit or board_pose_hit
    return {
        "detected_ids": detected_ids,
        "center_hit": center_hit,
        "corner_count": corner_count,
        "board_pose_hit": board_pose_hit,
        "nested_hit": nested_hit,
        "all_hit": detected_ids == ALL_MARKER_IDS,
        "board_reprojection_error_px": reprojection_error,
    }


def make_scenarios(sample_count, seed):
    rng = random.Random(seed)
    phase_specs = [
        ("search", 0.45, (2.0, 5.0), 1.10, math.radians(12.0), (0.2, 1.5), (1.0, 8.0)),
        ("align", 0.35, (0.8, 2.2), 0.55, math.radians(9.0), (0.1, 1.1), (0.5, 5.5)),
        ("terminal", 0.20, (0.28, 1.0), 0.25, math.radians(6.0), (0.0, 0.8), (0.0, 4.0)),
    ]
    cumulative = []
    total = 0.0
    for spec in phase_specs:
        total += spec[1]
        cumulative.append((total, spec))

    scenarios = []
    for _sample_index in range(sample_count):
        pick = rng.random() * total
        spec = next(item for threshold, item in cumulative if pick <= threshold)
        phase, _weight, height_range, lateral_max, tilt_max, blur_range, noise_range = spec
        radius = lateral_max * math.sqrt(rng.random())
        theta = rng.uniform(0.0, math.tau)
        scenarios.append(
            Scenario(
                phase=phase,
                camera_x=radius * math.cos(theta),
                camera_y=radius * math.sin(theta),
                height=rng.uniform(*height_range),
                yaw_rad=rng.uniform(-math.pi, math.pi),
                tilt_x_rad=rng.uniform(-tilt_max, tilt_max),
                tilt_y_rad=rng.uniform(-tilt_max, tilt_max),
                contrast=rng.uniform(0.72, 1.18),
                brightness=rng.uniform(-18.0, 18.0),
                blur_sigma=rng.uniform(*blur_range),
                noise_sigma=rng.uniform(*noise_range),
                noise_seed=rng.randrange(0, 2**32),
            )
        )
    return scenarios


def make_layouts(args):
    layouts = {}
    board_size = args.board_size
    for corner_length in parse_float_list(args.corner_lengths):
        for center_length in parse_float_list(args.center_lengths):
            for edge_margin in parse_float_list(args.edge_margins):
                corner_offset = board_size * 0.5 - edge_margin - corner_length * 0.5
                if corner_offset <= 0.0:
                    continue
                center_corner_gap = corner_offset - corner_length * 0.5 - center_length * 0.5
                if center_corner_gap < args.min_gap:
                    continue
                side_corner_gap = 2.0 * corner_offset - corner_length
                if side_corner_gap < args.min_gap:
                    continue
                name = "corner{:03d}_center{:03d}_edge{:03d}".format(
                    int(round(corner_length * 1000.0)),
                    int(round(center_length * 1000.0)),
                    int(round(edge_margin * 1000.0)),
                )
                layouts[name] = LayoutCase(
                    name=name,
                    board_size=board_size,
                    corner_length=corner_length,
                    center_length=center_length,
                    corner_offset=corner_offset,
                    edge_margin=edge_margin,
                )

    current_edge_margin = board_size * 0.5 - 0.28 - 0.24 * 0.5
    current = LayoutCase(
        name="current_corner240_center160_edge100",
        board_size=board_size,
        corner_length=0.24,
        center_length=0.16,
        corner_offset=0.28,
        edge_margin=current_edge_margin,
    )
    layouts.pop("corner240_center160_edge100", None)
    layouts[current.name] = current
    ordered = list(layouts.values())
    ordered.sort(key=lambda item: (item.corner_length, item.center_length, item.edge_margin, item.name))
    if args.max_cases is not None:
        ordered = ordered[: args.max_cases]
    return ordered


def evaluate_layout(layout, scenarios, dictionary, detector_params, camera_matrix, dist_coeffs, args):
    texture = make_board_texture(layout, dictionary, args.texture_pixels_per_meter)
    board = create_board(layout, dictionary)
    counts = {
        "any": 0,
        "center": 0,
        "corner_pair": 0,
        "board_pose": 0,
        "nested": 0,
        "all": 0,
    }
    phase_totals = {}
    phase_nested = {}
    detected_marker_total = 0
    per_id_hits = {marker_id: 0 for marker_id in sorted(ALL_MARKER_IDS)}
    reprojection_errors = []

    for scenario in scenarios:
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

        phase_totals[scenario.phase] = phase_totals.get(scenario.phase, 0) + 1
        if result["nested_hit"]:
            phase_nested[scenario.phase] = phase_nested.get(scenario.phase, 0) + 1

    total = len(scenarios)
    phase_rates = {
        phase: (phase_nested.get(phase, 0) / count if count else 0.0)
        for phase, count in sorted(phase_totals.items())
    }
    min_phase_nested_rate = min(phase_rates.values()) if phase_rates else 0.0
    mean_markers = detected_marker_total / total if total else 0.0
    nested_rate = counts["nested"] / total if total else 0.0
    board_pose_rate = counts["board_pose"] / total if total else 0.0
    center_rate = counts["center"] / total if total else 0.0
    all_rate = counts["all"] / total if total else 0.0
    score = (
        0.65 * nested_rate
        + 0.20 * board_pose_rate
        + 0.10 * center_rate
        + 0.05 * min_phase_nested_rate
    )

    return {
        "case": layout.name,
        "score": score,
        "samples": total,
        "corner_length_m": layout.corner_length,
        "center_length_m": layout.center_length,
        "corner_offset_m": layout.corner_offset,
        "edge_margin_m": layout.edge_margin,
        "nested_rate": nested_rate,
        "board_pose_rate": board_pose_rate,
        "any_marker_rate": counts["any"] / total if total else 0.0,
        "center_rate": center_rate,
        "corner_pair_rate": counts["corner_pair"] / total if total else 0.0,
        "all_markers_rate": all_rate,
        "mean_detected_markers": mean_markers,
        "min_phase_nested_rate": min_phase_nested_rate,
        "phase_nested_rates": phase_rates,
        "per_marker_rates": {
            str(marker_id): per_id_hits[marker_id] / total if total else 0.0
            for marker_id in sorted(per_id_hits)
        },
        "board_reprojection_error_mean_px": (
            float(np.mean(reprojection_errors)) if reprojection_errors else None
        ),
        "board_reprojection_error_p95_px": (
            float(np.percentile(reprojection_errors, 95)) if reprojection_errors else None
        ),
    }


def result_sort_key(result):
    return (
        result["score"],
        result["nested_rate"],
        result["board_pose_rate"],
        result["center_rate"],
        result["mean_detected_markers"],
    )


def write_marker_yaml(layout, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"marker_configs": layout.marker_configs()}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_preview(layout, dictionary, path, pixels_per_meter):
    path.parent.mkdir(parents=True, exist_ok=True)
    texture = make_board_texture(layout, dictionary, pixels_per_meter)
    cv2.imwrite(str(path), texture)


def write_gazebo_model(layout, dictionary, model_name, model_root):
    model_dir = model_root / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    texture_dir = model_dir / "materials" / "textures"
    script_dir = model_dir / "materials" / "scripts"
    texture_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in sorted(ALL_MARKER_IDS):
        marker = draw_marker(dictionary, marker_id, 1000)
        cv2.imwrite(str(texture_dir / f"6x6_1000-{marker_id}.png"), marker)

    material_lines = []
    for marker_id in sorted(ALL_MARKER_IDS):
        material_lines.extend(
            [
                f"material px4/{model_name}_{marker_id}",
                "{",
                "  receive_shadows off",
                "  technique",
                "  {",
                "    pass",
                "    {",
                "      texture_unit",
                "      {",
                f"        texture 6x6_1000-{marker_id}.png",
                "      }",
                "    }",
                "  }",
                "}",
                "",
            ]
        )
    (script_dir / f"{model_name}.material").write_text("\n".join(material_lines), encoding="utf-8")

    sdf = build_model_sdf(layout, model_name)
    (model_dir / "model.sdf").write_text(sdf, encoding="utf-8")
    model_config = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>PX4 ArUco nested layout experiment</name>
  </author>
  <description>Generated 1 m nested ArUco board layout for detection probability tests.</description>
</model>
"""
    (model_dir / "model.config").write_text(model_config, encoding="utf-8")
    return model_dir


def visual_sdf(layout, model_name, visual_name, marker_id, x, y, length, z_offset):
    return f"""
      <visual name="{visual_name}">
        <pose>{x:.6f} {y:.6f} {z_offset:.6f} 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box>
            <size>{length:.6f} {length:.6f} 0.001</size>
          </box>
        </geometry>
        <material>
          <script>
            <uri>model://{model_name}/materials/scripts</uri>
            <uri>model://{model_name}/materials/textures</uri>
            <name>px4/{model_name}_{marker_id}</name>
          </script>
        </material>
      </visual>"""


def build_model_sdf(layout, model_name):
    visuals = []
    for marker_id, visual_name, sign_x, sign_y in CORNER_MARKERS:
        visuals.append(
            visual_sdf(
                layout,
                model_name,
                visual_name,
                marker_id,
                sign_x * layout.corner_offset,
                sign_y * layout.corner_offset,
                layout.corner_length,
                0.0055,
            )
        )
    visuals.append(
        visual_sdf(
            layout,
            model_name,
            "inner_marker_center",
            CENTER_MARKER_ID,
            0.0,
            0.0,
            layout.center_length,
            0.0058,
        )
    )
    visuals_text = "\n".join(visuals)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <pose>0 0 0.005 0 0 0</pose>

    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{layout.board_size:.6f} {layout.board_size:.6f} 0.01</size>
          </box>
        </geometry>
      </collision>

      <visual name="board_base">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box>
            <size>{layout.board_size:.6f} {layout.board_size:.6f} 0.01</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>
{visuals_text}
    </link>
  </model>
</sdf>
"""


def write_world(model_name, world_path):
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_text = f"""<?xml version="1.0"?>
<sdf version="1.5">
  <world name="{model_name}_world">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://{model_name}</uri>
      <pose>0 0 0.005 0 0 0</pose>
    </include>

    <physics name="default_physics" default="0" type="ode">
      <gravity>0 0 -9.8066</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
          <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
      <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
    </physics>
  </world>
</sdf>
"""
    world_path.write_text(world_text, encoding="utf-8")


def write_case_assets(ranked_results, layouts_by_name, dictionary, output_root, top_n):
    config_root = output_root / "generated_configs"
    preview_root = output_root / "previews"
    model_root = output_root / "gazebo_models"
    world_root = output_root / "gazebo_worlds"
    assets = []

    for rank, result in enumerate(ranked_results[:top_n], start=1):
        layout = layouts_by_name[result["case"]]
        safe_case = result["case"]
        model_name = f"aruco_nested_board_exp_rank{rank:02d}_{safe_case}"
        config_path = config_root / f"rank{rank:02d}_{safe_case}.yaml"
        preview_path = preview_root / f"rank{rank:02d}_{safe_case}.png"
        world_path = world_root / f"rank{rank:02d}_{safe_case}.world"
        write_marker_yaml(layout, config_path)
        write_preview(layout, dictionary, preview_path, 1200)
        write_gazebo_model(layout, dictionary, model_name, model_root)
        write_world(model_name, world_path)
        assets.append(
            {
                "rank": rank,
                "case": safe_case,
                "model_name": model_name,
                "marker_config": str(config_path),
                "preview": str(preview_path),
                "gazebo_model_root": str(model_root),
                "world": str(world_path),
            }
        )

    return assets


def write_results_csv(ranked_results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "case",
        "score",
        "nested_rate",
        "board_pose_rate",
        "any_marker_rate",
        "center_rate",
        "corner_pair_rate",
        "all_markers_rate",
        "mean_detected_markers",
        "min_phase_nested_rate",
        "corner_length_m",
        "center_length_m",
        "corner_offset_m",
        "edge_margin_m",
        "search_nested_rate",
        "align_nested_rate",
        "terminal_nested_rate",
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
        for rank, result in enumerate(ranked_results, start=1):
            row = dict(result)
            row["rank"] = rank
            row["search_nested_rate"] = result["phase_nested_rates"].get("search", 0.0)
            row["align_nested_rate"] = result["phase_nested_rates"].get("align", 0.0)
            row["terminal_nested_rate"] = result["phase_nested_rates"].get("terminal", 0.0)
            for marker_id in sorted(ALL_MARKER_IDS):
                row[f"marker{marker_id}_rate"] = result["per_marker_rates"].get(str(marker_id), 0.0)
            writer.writerow({field: row.get(field) for field in fieldnames})


def run_sitl_for_assets(assets, args):
    sitl_root = Path(args.output_root) / "sitl_validation"
    env = os.environ.copy()
    model_root = assets[0]["gazebo_model_root"] if assets else ""
    env["GAZEBO_MODEL_PATH"] = f"{model_root}:{env.get('GAZEBO_MODEL_PATH', '')}"

    for asset in assets[: args.run_sitl_top]:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_aruco_landing.py"),
            "--label",
            f"layout_rank{asset['rank']:02d}_{asset['case']}",
            "--controller",
            args.sitl_controller,
            "--detector-type",
            args.sitl_detector_type,
            "--marker-config",
            asset["marker_config"],
            "--world",
            asset["world"],
            "--runs",
            str(args.sitl_runs),
            "--timeout",
            str(args.sitl_timeout),
            "--target-x",
            str(args.target_x),
            "--target-y",
            str(args.target_y),
            "--output-root",
            str(sitl_root),
        ]
        print("running SITL validation:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Scan 1 m nested ArUco board layouts and rank detection probability."
    )
    parser.add_argument("--samples", type=int, default=160, help="Monte Carlo camera samples per layout")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--board-size", type=float, default=1.0)
    parser.add_argument("--corner-lengths", default="0.18,0.22,0.24,0.26,0.30,0.34")
    parser.add_argument("--center-lengths", default="0.10,0.14,0.16,0.18,0.22,0.26,0.30")
    parser.add_argument("--edge-margins", default="0.04,0.07,0.10,0.13")
    parser.add_argument("--min-gap", type=float, default=0.035)
    parser.add_argument("--max-cases", type=int, default=None, help="Debug limit for layout count")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--texture-pixels-per-meter", type=int, default=1800)
    parser.add_argument(
        "--camera-param-path",
        default=str(ROOT / "config" / "camera_monocular_1280x720.yaml"),
    )
    parser.add_argument("--dictionary-id", type=int, default=int(cv2.aruco.DICT_6X6_1000))
    parser.add_argument("--max-board-reprojection-error", type=float, default=5.0)
    parser.add_argument("--top-assets", type=int, default=5)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-sitl-top", type=int, default=0, help="Run full SITL validation for top N layouts")
    parser.add_argument("--sitl-runs", type=int, default=3)
    parser.add_argument("--sitl-timeout", type=float, default=150.0)
    parser.add_argument("--sitl-controller", default="aruco_search_and_detect.py")
    parser.add_argument("--sitl-detector-type", default="aruco_multi_marker_det_weighted.py")
    parser.add_argument("--target-x", type=float, default=1.5)
    parser.add_argument("--target-y", type=float, default=1.5)
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary_id)
    detector_params = create_detector_params()
    camera_matrix, dist_coeffs = load_camera_params(args.camera_param_path)
    layouts = make_layouts(args)
    layouts_by_name = {layout.name: layout for layout in layouts}
    scenarios = make_scenarios(args.samples, args.seed)

    print(
        f"evaluating {len(layouts)} layouts with {len(scenarios)} shared camera samples each",
        flush=True,
    )
    results = []
    for index, layout in enumerate(layouts, start=1):
        result = evaluate_layout(
            layout,
            scenarios,
            dictionary,
            detector_params,
            camera_matrix,
            dist_coeffs,
            args,
        )
        results.append(result)
        print(
            "{}/{} {} nested={:.3f} board_pose={:.3f} center={:.3f}".format(
                index,
                len(layouts),
                layout.name,
                result["nested_rate"],
                result["board_pose_rate"],
                result["center_rate"],
            ),
            flush=True,
        )

    ranked_results = sorted(results, key=result_sort_key, reverse=True)
    assets = write_case_assets(ranked_results, layouts_by_name, dictionary, output_root, args.top_assets)
    write_results_csv(ranked_results, output_root / "layout_results.csv")

    summary = {
        "samples_per_layout": args.samples,
        "seed": args.seed,
        "layout_count": len(layouts),
        "score_formula": "0.65*nested_rate + 0.20*board_pose_rate + 0.10*center_rate + 0.05*min_phase_nested_rate",
        "top_results": ranked_results[: min(10, len(ranked_results))],
        "generated_assets": assets,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if ranked_results:
        best = ranked_results[0]
        print(
            "best layout: {case} corner={corner_length_m:.3f} center={center_length_m:.3f} "
            "offset={corner_offset_m:.3f} edge={edge_margin_m:.3f} nested={nested_rate:.3f}".format(
                **best
            ),
            flush=True,
        )
    print(f"results: {output_root / 'layout_results.csv'}", flush=True)
    print(f"summary: {output_root / 'summary.json'}", flush=True)
    if assets:
        print(f"generated gazebo models: {assets[0]['gazebo_model_root']}", flush=True)
        print(f"top marker config: {assets[0]['marker_config']}", flush=True)
        print(f"top world: {assets[0]['world']}", flush=True)

    if args.run_sitl_top > 0:
        run_sitl_for_assets(assets, args)


if __name__ == "__main__":
    main()
