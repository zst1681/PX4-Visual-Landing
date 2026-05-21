#!/usr/bin/env python3

import math
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header

PX4_ROOT = Path(__file__).resolve().parents[1]


def rotation_matrix_to_quaternion(rotation_matrix):
    trace = np.trace(rotation_matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        qz = 0.25 * scale
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def normalize_quaternion(quaternion):
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class ArucoWeightedMultiMarkerDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_lock = Lock()
        self.latest_frame = None
        self.latest_header = None
        self.last_selection_signature = None

        self.dictionary_id = rospy.get_param("~dictionary_id", int(cv2.aruco.DICT_6X6_1000))
        self.camera_param_path = rospy.get_param(
            "~camera_param_path",
            str(PX4_ROOT / "config" / "camera_monocular_1280x720.yaml"),
        )
        self.sub_image_topic = rospy.get_param("~sub_image_topic", "/iris/camera/image_raw")
        self.default_frame_id = rospy.get_param("~default_frame_id", "camera_link")
        self.debug_axis_length = rospy.get_param("~debug_axis_length", 0.1)
        self.publish_debug_image = rospy.get_param("~publish_debug_image", True)

        self.center_role_weight = rospy.get_param("~center_role_weight", 1.0)
        self.corner_role_weight = rospy.get_param("~corner_role_weight", 1.0)
        self.corner_cluster_bonus = rospy.get_param("~corner_cluster_bonus", 1.6)
        self.center_max_corner_ratio = rospy.get_param("~center_max_corner_ratio", 0.55)
        self.fusion_outlier_scale = rospy.get_param("~fusion_outlier_scale", 0.10)
        self.fusion_consistency_scale = rospy.get_param("~fusion_consistency_scale", 0.08)
        self.single_corner_penalty = rospy.get_param("~single_corner_penalty", 0.45)
        self.min_corner_count_for_fusion = rospy.get_param("~min_corner_count_for_fusion", 2)

        self.enable_board_pose = rospy.get_param("~enable_board_pose", True)
        self.board_pose_min_markers = max(1, int(rospy.get_param("~board_pose_min_markers", 2)))
        self.max_board_reprojection_error = float(rospy.get_param("~max_board_reprojection_error", 5.0))
        self.max_single_marker_reprojection_error = float(
            rospy.get_param("~max_single_marker_reprojection_error", 10.0)
        )
        self.min_marker_area_px2 = float(rospy.get_param("~min_marker_area_px2", 120.0))
        self.edge_margin_px = float(rospy.get_param("~edge_margin_px", 24.0))
        self.min_view_cosine = float(rospy.get_param("~min_view_cosine", 0.12))

        self.marker_configs = self._load_marker_configs()
        self.marker_config_by_id = {config["id"]: config for config in self.marker_configs}
        self.camera_matrix, self.dist_coeffs = self._load_camera_params(self.camera_param_path)
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        self.board = self._create_marker_board()
        self.detector_params = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 5
        self.detector_params.cornerRefinementMaxIterations = 40
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10

        self.pose_pub = rospy.Publisher("/aruco/pose", PoseStamped, queue_size=10)
        self.debug_image_pub = rospy.Publisher("aruco_det_image", Image, queue_size=10)
        rospy.Subscriber(self.sub_image_topic, Image, self._image_cb, queue_size=1)

        marker_summary = ", ".join(
            "id={id} role={role} length={length:.2f} offset=({offset_x:.2f},{offset_y:.2f})".format(**config)
            for config in self.marker_configs
        )
        rospy.loginfo("weighted aruco detector subscribed to %s", self.sub_image_topic)
        rospy.loginfo("loaded weighted marker configs: %s", marker_summary)
        rospy.loginfo(
            "weighted aruco detector board pose=%s min_markers=%d",
            "enabled" if self.board is not None and self.enable_board_pose else "disabled",
            self.board_pose_min_markers,
        )

    def _load_camera_params(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        fx = float(config["fx"])
        fy = float(config["fy"])
        cx = float(config["x0"])
        cy = float(config["y0"])
        k1 = float(config["k1"])
        k2 = float(config["k2"])
        p1 = float(config["p1"])
        p2 = float(config["p2"])
        k3 = float(config["k3"])

        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.array([[k1], [k2], [p1], [p2], [k3]], dtype=np.float64)
        return camera_matrix, dist_coeffs

    def _load_marker_configs(self):
        marker_configs = rospy.get_param("~marker_configs", [])
        if marker_configs:
            configs = []
            for index, config in enumerate(marker_configs):
                offset_x = float(config.get("offset_x", 0.0))
                offset_y = float(config.get("offset_y", 0.0))
                role = config.get("role")
                if role not in ("center", "corner"):
                    role = "center" if abs(offset_x) + abs(offset_y) < 1e-6 else "corner"
                configs.append(
                    {
                        "id": int(config["id"]),
                        "length": float(config["length"]),
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "priority": int(config.get("priority", index)),
                        "role": role,
                    }
                )
            return configs

        return [
            {
                "id": int(rospy.get_param("~aruco_id", 31)),
                "length": float(rospy.get_param("~aruco_length", 0.2)),
                "offset_x": 0.0,
                "offset_y": 0.0,
                "priority": 0,
                "role": "center",
            }
        ]

    def _create_marker_board(self):
        if not self.enable_board_pose:
            return None
        board_create = getattr(cv2.aruco, "Board_create", None)
        if board_create is None:
            return None

        board_points = [
            self._marker_object_points(
                config["length"],
                config["offset_x"],
                config["offset_y"],
            ).astype(np.float32)
            for config in self.marker_configs
        ]
        board_ids = np.array([[config["id"]] for config in self.marker_configs], dtype=np.int32)
        return board_create(board_points, self.dictionary, board_ids)

    def _marker_object_points(self, marker_length, center_x=0.0, center_y=0.0):
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

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "cv_bridge error: %s", exc)
            return

        with self.frame_lock:
            self.latest_frame = frame
            self.latest_header = msg.header

    def _publish_debug_image(self, frame, header):
        if not self.publish_debug_image:
            return
        try:
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "failed to publish debug image: %s", exc)
            return
        image_msg.header = header
        self.debug_image_pub.publish(image_msg)

    def _header_or_default(self, header):
        if header is not None:
            return header
        return Header(stamp=rospy.Time.now(), frame_id=self.default_frame_id)

    def _polygon_area(self, corner_points):
        x_coords = corner_points[:, 0]
        y_coords = corner_points[:, 1]
        return 0.5 * abs(
            np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))
        )

    def _border_margin_px(self, corner_points, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        min_x = float(np.min(corner_points[:, 0]))
        min_y = float(np.min(corner_points[:, 1]))
        max_x = float(np.max(corner_points[:, 0]))
        max_y = float(np.max(corner_points[:, 1]))
        return min(
            min_x,
            min_y,
            (frame_width - 1) - max_x,
            (frame_height - 1) - max_y,
        )

    def _reprojection_error(self, observed_corners, rvec, tvec, marker_length):
        object_points = self._marker_object_points(marker_length)
        projected_corners, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected_corners = projected_corners.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1))))

    def _board_reprojection_error(self, candidates, rvec, tvec):
        residuals = []
        for candidate in candidates:
            config = self.marker_config_by_id[candidate["id"]]
            object_points = self._marker_object_points(
                config["length"],
                config["offset_x"],
                config["offset_y"],
            )
            projected_corners, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
            projected_corners = projected_corners.reshape(-1, 2)
            observed_corners = candidate["observed_corners"]
            residuals.append(np.sum((projected_corners - observed_corners) ** 2, axis=1))

        if not residuals:
            return float("inf")
        residuals = np.concatenate(residuals)
        return float(np.sqrt(np.mean(residuals)))

    def _candidate_is_usable(self, candidate):
        return (
            candidate["area_px2"] >= self.min_marker_area_px2
            and candidate["reprojection_error"] <= self.max_single_marker_reprojection_error
            and candidate["border_margin_px"] >= -0.5 * self.edge_margin_px
            and candidate["view_cosine"] >= self.min_view_cosine
        )

    def _candidate_quality_weight(self, candidate):
        area_term = math.sqrt(max(candidate["area_px2"], 1.0) / 1000.0)
        distance_term = 1.0 / max(candidate["pad_tvec"][2], 0.10)
        reprojection_term = 1.0 / (1.0 + candidate["reprojection_error"])
        edge_term = 0.25 + 0.75 * clamp(candidate["border_margin_px"] / max(self.edge_margin_px, 1.0), 0.0, 1.0)
        view_term = 0.20 + 0.80 * clamp(
            (candidate["view_cosine"] - self.min_view_cosine) / max(1.0 - self.min_view_cosine, 1e-6),
            0.0,
            1.0,
        )
        return max(area_term * math.sqrt(distance_term) * reprojection_term * edge_term * view_term, 1e-6)

    def _role_weight(self, candidate):
        return self.center_role_weight if candidate["role"] == "center" else self.corner_role_weight

    def _weighted_average_vectors(self, vectors, weights):
        normalized_weights = np.asarray(weights, dtype=np.float64)
        weight_sum = float(np.sum(normalized_weights))
        if weight_sum <= 1e-9:
            return np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
        normalized_weights /= weight_sum
        return np.sum(np.asarray(vectors, dtype=np.float64) * normalized_weights[:, None], axis=0)

    def _weighted_average_quaternions(self, quaternions, weights):
        reference = None
        aligned = []
        for quaternion in quaternions:
            normalized_quaternion = normalize_quaternion(quaternion)
            if reference is None:
                reference = normalized_quaternion
            elif np.dot(reference, normalized_quaternion) < 0.0:
                normalized_quaternion = -normalized_quaternion
            aligned.append(normalized_quaternion)
        return normalize_quaternion(self._weighted_average_vectors(aligned, weights))

    def _build_candidates(self, marker_ids, marker_corners, frame_shape):
        candidates = []
        for index, marker_id in enumerate(marker_ids.flatten()):
            config = self.marker_config_by_id.get(int(marker_id))
            if config is None:
                continue

            raw_corners = marker_corners[index]
            observed_corners = raw_corners.reshape(-1, 2)
            rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
                [raw_corners],
                config["length"],
                self.camera_matrix,
                self.dist_coeffs,
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            pad_center_in_marker = np.array(
                [-config["offset_x"], -config["offset_y"], 0.0],
                dtype=np.float64,
            )
            pad_tvec = rotation_matrix.dot(pad_center_in_marker) + tvec
            quaternion = rotation_matrix_to_quaternion(rotation_matrix)
            reprojection_error = self._reprojection_error(observed_corners, rvec, tvec, config["length"])
            marker_area_px2 = self._polygon_area(observed_corners)
            border_margin_px = self._border_margin_px(observed_corners, frame_shape)
            view_cosine = abs(float(rotation_matrix[2, 2]))

            candidate = {
                "id": int(marker_id),
                "priority": config["priority"],
                "role": config["role"],
                "length": config["length"],
                "rvec": rvec,
                "marker_tvec": tvec,
                "pad_tvec": pad_tvec,
                "quaternion": quaternion,
                "reprojection_error": reprojection_error,
                "area_px2": marker_area_px2,
                "border_margin_px": border_margin_px,
                "view_cosine": view_cosine,
                "raw_corners": raw_corners,
                "observed_corners": observed_corners,
            }
            candidate["usable"] = self._candidate_is_usable(candidate)
            candidates.append(candidate)

        return candidates

    def _estimate_board_pose(self, candidates):
        if self.board is None or not hasattr(cv2.aruco, "estimatePoseBoard"):
            return None

        usable_candidates = [candidate for candidate in candidates if candidate["usable"]]
        if len(usable_candidates) < self.board_pose_min_markers:
            return None

        board_corners = [candidate["raw_corners"] for candidate in usable_candidates]
        board_ids = np.array([[candidate["id"]] for candidate in usable_candidates], dtype=np.int32)

        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            board_corners,
            board_ids,
            self.board,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            None,
        )
        if retval <= 0:
            return None

        board_reprojection_error = self._board_reprojection_error(usable_candidates, rvec, tvec)
        if board_reprojection_error > self.max_board_reprojection_error:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return {
            "selected": usable_candidates,
            "pad_tvec": tvec.reshape(3),
            "quaternion": rotation_matrix_to_quaternion(rotation_matrix),
            "rvec": rvec.reshape(3),
            "reprojection_error": board_reprojection_error,
        }

    def _fuse_candidate_group(self, candidates):
        base_weights = [
            self._candidate_quality_weight(candidate) * self._role_weight(candidate)
            for candidate in candidates
        ]

        pad_tvec = self._weighted_average_vectors([candidate["pad_tvec"] for candidate in candidates], base_weights)
        fused_weights = list(base_weights)
        for _ in range(2):
            residuals = [
                float(np.linalg.norm(candidate["pad_tvec"] - pad_tvec))
                for candidate in candidates
            ]
            robust_weights = [
                1.0 / (1.0 + (residual / self.fusion_outlier_scale) ** 2)
                for residual in residuals
            ]
            fused_weights = [
                max(base_weight * robust_weight, 1e-6)
                for base_weight, robust_weight in zip(base_weights, robust_weights)
            ]
            pad_tvec = self._weighted_average_vectors(
                [candidate["pad_tvec"] for candidate in candidates],
                fused_weights,
            )

        quaternion = self._weighted_average_quaternions(
            [candidate["quaternion"] for candidate in candidates],
            fused_weights,
        )
        return pad_tvec, quaternion, fused_weights

    def _merge_group_pose(self, group_poses):
        total_weights = [group_pose["weight"] for group_pose in group_poses]
        merged_tvec = self._weighted_average_vectors(
            [group_pose["pad_tvec"] for group_pose in group_poses],
            total_weights,
        )
        merged_quaternion = self._weighted_average_quaternions(
            [group_pose["quaternion"] for group_pose in group_poses],
            total_weights,
        )
        return merged_tvec, merged_quaternion

    def _select_pose(self, candidates):
        if not candidates:
            return None

        board_pose = self._estimate_board_pose(candidates)
        if board_pose is not None:
            selected = board_pose["selected"]
            signature = ("board_pose", tuple(sorted(candidate["id"] for candidate in selected)))
            summary = "markers={} reproj={:.2f}".format(
                len(selected),
                board_pose["reprojection_error"],
            )
            return (
                selected,
                board_pose["pad_tvec"],
                board_pose["quaternion"],
                signature,
                summary,
                board_pose["rvec"],
            )

        usable_candidates = [candidate for candidate in candidates if candidate["usable"]] or candidates
        center_candidates = [candidate for candidate in usable_candidates if candidate["role"] == "center"]
        corner_candidates = [candidate for candidate in usable_candidates if candidate["role"] == "corner"]

        fusion_groups = []
        fusion_mode = "center_only"
        summary = "usable={}/{}".format(
            sum(1 for candidate in candidates if candidate["usable"]),
            len(candidates),
        )

        if center_candidates:
            center_tvec, center_quaternion, center_weights = self._fuse_candidate_group(center_candidates)
            fusion_groups.append(
                {
                    "label": "center",
                    "pad_tvec": center_tvec,
                    "quaternion": center_quaternion,
                    "weight": sum(center_weights),
                }
            )

        if corner_candidates:
            corner_tvec, corner_quaternion, corner_weights = self._fuse_candidate_group(corner_candidates)
            corner_weight = sum(corner_weights)
            if len(corner_candidates) == 1:
                corner_weight *= self.single_corner_penalty
            elif len(corner_candidates) >= self.min_corner_count_for_fusion:
                corner_gain = 1.0 + (self.corner_cluster_bonus - 1.0) * clamp(len(corner_candidates) / 4.0, 0.0, 1.0)
                corner_weight *= corner_gain
            fusion_groups.append(
                {
                    "label": "corners",
                    "pad_tvec": corner_tvec,
                    "quaternion": corner_quaternion,
                    "weight": corner_weight,
                }
            )

        if len(fusion_groups) == 2:
            center_group = next(group for group in fusion_groups if group["label"] == "center")
            corner_group = next(group for group in fusion_groups if group["label"] == "corners")
            center_corner_residual = float(np.linalg.norm(center_group["pad_tvec"] - corner_group["pad_tvec"]))
            consistency_weight = 1.0 / (1.0 + (center_corner_residual / self.fusion_consistency_scale) ** 2)
            center_group["weight"] *= consistency_weight
            if len(corner_candidates) >= 3:
                center_group["weight"] = min(
                    center_group["weight"],
                    corner_group["weight"] * self.center_max_corner_ratio,
                )
            fusion_mode = "center_corner_weighted"
            summary = "{} residual={:.3f} center_w={:.3f} corner_w={:.3f}".format(
                summary,
                center_corner_residual,
                center_group["weight"],
                corner_group["weight"],
            )
        elif corner_candidates:
            fusion_mode = "corner_only" if not center_candidates else "center_corner_weighted"

        fused_tvec, fused_quaternion = self._merge_group_pose(fusion_groups)
        selected = center_candidates + corner_candidates
        signature = (fusion_mode, tuple(sorted(candidate["id"] for candidate in selected)))
        return selected, fused_tvec, fused_quaternion, signature, summary, None

    def _log_selection(self, selected, signature, summary):
        if signature == self.last_selection_signature:
            return
        self.last_selection_signature = signature
        selected_ids = [candidate["id"] for candidate in selected]
        rospy.loginfo(
            "weighted aruco fusion mode=%s ids=%s %s",
            signature[0],
            selected_ids,
            summary,
        )

    def spin(self):
        rate = rospy.Rate(15.0)
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_frame is None:
                    frame = None
                    header = None
                else:
                    frame = self.latest_frame.copy()
                    header = self.latest_header

            if frame is None:
                rate.sleep()
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.detector_params,
            )

            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                candidates = self._build_candidates(marker_ids, marker_corners, frame.shape)
                selection = self._select_pose(candidates)

                if selection is not None:
                    selected, pad_tvec, quaternion, signature, summary, board_rvec = selection
                    self._log_selection(selected, signature, summary)

                    for candidate in candidates:
                        text_color = (0, 255, 0) if candidate["usable"] else (0, 0, 255)
                        cv2.aruco.drawAxis(
                            frame,
                            self.camera_matrix,
                            self.dist_coeffs,
                            candidate["rvec"],
                            candidate["marker_tvec"],
                            self.debug_axis_length,
                        )
                        center = np.mean(candidate["observed_corners"], axis=0).astype(int)
                        cv2.putText(
                            frame,
                            "{}{}".format(candidate["id"], "" if candidate["usable"] else "!"),
                            tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            text_color,
                            2,
                            cv2.LINE_AA,
                        )

                    if board_rvec is not None:
                        cv2.aruco.drawAxis(
                            frame,
                            self.camera_matrix,
                            self.dist_coeffs,
                            board_rvec,
                            pad_tvec,
                            self.debug_axis_length * 1.2,
                        )

                    pose_msg = PoseStamped()
                    pose_msg.header = self._header_or_default(header)
                    pose_msg.pose.position.x = float(pad_tvec[0])
                    pose_msg.pose.position.y = float(pad_tvec[1])
                    pose_msg.pose.position.z = float(pad_tvec[2])
                    pose_msg.pose.orientation.x = float(quaternion[0])
                    pose_msg.pose.orientation.y = float(quaternion[1])
                    pose_msg.pose.orientation.z = float(quaternion[2])
                    pose_msg.pose.orientation.w = float(quaternion[3])
                    self.pose_pub.publish(pose_msg)

                    cv2.putText(
                        frame,
                        "fusion: {} ids={}".format(
                            signature[0],
                            ",".join(str(candidate["id"]) for candidate in selected),
                        ),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "pad xyz: {:.2f} {:.2f} {:.2f}".format(
                            pad_tvec[0],
                            pad_tvec[1],
                            pad_tvec[2],
                        ),
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        summary,
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if header is None:
                header = self._header_or_default(header)
            self._publish_debug_image(frame, header)
            rate.sleep()


def main():
    rospy.init_node("aruco_multi_marker_det", anonymous=False)
    detector = ArucoWeightedMultiMarkerDetector()
    detector.spin()


if __name__ == "__main__":
    main()
