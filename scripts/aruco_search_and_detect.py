#!/usr/bin/env python3

import json
import math
import sys
from collections import deque
from pathlib import Path

PX4_ROOT = Path(__file__).resolve().parents[1]
MAVLINK_PYTHON = PX4_ROOT / "src" / "modules" / "mavlink" / "mavlink"
sys.path.insert(0, str(MAVLINK_PYTHON))

import rospy
from geometry_msgs.msg import PoseStamped
from mavros import mavlink as mavlink_convert
from mavros_msgs.msg import ExtendedState, Mavlink, ParamValue, State
from mavros_msgs.srv import CommandBool, ParamSet, SetMode
from pymavlink import mavutil


TBC_ROT = (
    (0.0, -1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def quat_to_matrix(orientation):
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat_vec_mul(matrix, vector):
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class ArucoSearchController:
    def __init__(self):
        self.auto_arm_offboard = rospy.get_param("~auto_arm_offboard", True)
        self.land_after_detect = rospy.get_param("~land_after_detect", False)
        self.takeoff_altitude = rospy.get_param("~takeoff_altitude", 1.8)
        self.search_spacing = rospy.get_param("~search_spacing", 1.0)
        self.search_radius = rospy.get_param("~search_radius", 3.0)
        self.position_tolerance = rospy.get_param("~position_tolerance", 0.28)
        self.altitude_tolerance = rospy.get_param("~altitude_tolerance", 0.22)
        self.center_tolerance = rospy.get_param("~center_tolerance", 0.08)
        self.land_center_tolerance = rospy.get_param("~land_center_tolerance", 0.12)
        self.track_lost_timeout = rospy.get_param("~track_lost_timeout", 2.0)
        self.offboard_retry_period = rospy.get_param("~offboard_retry_period", 2.0)
        self.arm_retry_period = rospy.get_param("~arm_retry_period", 2.0)
        self.search_dwell_time = rospy.get_param("~search_dwell_time", 0.6)
        self.detection_confirm_count = rospy.get_param("~detection_confirm_count", 5)
        self.success_hold_count = rospy.get_param("~success_hold_count", 10)
        self.land_align_hold_count = rospy.get_param("~land_align_hold_count", 8)
        self.land_descent_rate = rospy.get_param("~land_descent_rate", 0.15)
        self.land_align_min_altitude = rospy.get_param("~land_align_min_altitude", 0.45)
        self.auto_land_trigger_altitude = rospy.get_param("~auto_land_trigger_altitude", 0.35)
        self.landing_commit_altitude = rospy.get_param(
            "~landing_commit_altitude",
            max(self.land_align_min_altitude, self.auto_land_trigger_altitude),
        )
        self.land_commit_tolerance = rospy.get_param("~land_commit_tolerance", 0.05)
        self.auto_land_settle_time = rospy.get_param("~auto_land_settle_time", 1.0)
        self.success_result_path = rospy.get_param(
            "~success_result_path", "/tmp/px4_aruco_search_result.json"
        )
        self.marker_estimate_window = rospy.get_param("~marker_estimate_window", 0.8)
        self.marker_estimate_trim_ratio = rospy.get_param("~marker_estimate_trim_ratio", 0.2)
        self.marker_estimate_alpha = rospy.get_param("~marker_estimate_alpha", 0.65)
        self.fixed_yaw = rospy.get_param("~fixed_yaw", 0.0)
        self.track_pid_kp = rospy.get_param("~track_pid_kp", 0.72)
        self.track_pid_ki = rospy.get_param("~track_pid_ki", 0.0)
        self.track_pid_kd = rospy.get_param("~track_pid_kd", 0.18)
        self.land_pid_kp = rospy.get_param("~land_pid_kp", 0.58)
        self.land_pid_ki = rospy.get_param("~land_pid_ki", 0.02)
        self.land_pid_kd = rospy.get_param("~land_pid_kd", 0.16)
        self.pid_integral_limit = rospy.get_param("~pid_integral_limit", 0.35)
        self.track_output_limit = rospy.get_param("~track_output_limit", 1.0)
        self.land_output_limit = rospy.get_param("~land_output_limit", 0.45)
        self.track_target_alpha = rospy.get_param("~track_target_alpha", 0.45)
        self.land_target_alpha = rospy.get_param("~land_target_alpha", 0.25)
        self.track_target_step_limit = rospy.get_param("~track_target_step_limit", 0.20)
        self.land_target_step_limit = rospy.get_param("~land_target_step_limit", 0.10)
        self.land_hold_error = rospy.get_param("~land_hold_error", 0.18)
        self.land_slow_error = rospy.get_param("~land_slow_error", 0.08)
        self.land_slow_descent_factor = rospy.get_param("~land_slow_descent_factor", 0.35)
        self.land_z_pid_kp = rospy.get_param("~land_z_pid_kp", 1.05)
        self.land_z_pid_ki = rospy.get_param("~land_z_pid_ki", 0.06)
        self.land_z_pid_kd = rospy.get_param("~land_z_pid_kd", 0.18)
        self.land_z_integral_limit = rospy.get_param("~land_z_integral_limit", 0.18)
        self.land_z_output_limit = rospy.get_param("~land_z_output_limit", 0.10)
        self.land_descent_gate_alpha = rospy.get_param("~land_descent_gate_alpha", 0.35)
        self.descent_realign_tolerance = rospy.get_param(
            "~descent_realign_tolerance",
            max(self.land_center_tolerance, 0.10),
        )
        self.low_altitude_xy_slowdown_start = rospy.get_param(
            "~low_altitude_xy_slowdown_start",
            max(self.land_align_min_altitude + 0.35, 0.9),
        )
        self.track_low_altitude_xy_scale_min = rospy.get_param("~track_low_altitude_xy_scale_min", 0.70)
        self.track_low_altitude_step_scale_min = rospy.get_param("~track_low_altitude_step_scale_min", 0.75)
        self.land_low_altitude_xy_scale_min = rospy.get_param("~land_low_altitude_xy_scale_min", 0.40)
        self.land_low_altitude_step_scale_min = rospy.get_param("~land_low_altitude_step_scale_min", 0.45)

        self.state = State()
        self.extended_state = ExtendedState()
        self.local_pose = None
        self.home_pose = None
        self.flight_phase = "WAIT_FOR_HOME"
        self.search_waypoints = []
        self.search_index = 0
        self.search_arrival_time = None
        self.marker_world_estimate = None
        self.landing_target_xy = None
        self.last_detection_time = None
        self.last_relative_pose = None
        self.recent_detections = deque(maxlen=40)
        self.success_hits = 0
        self.land_align_hits = 0
        self.control_started_at = None
        self.offboard_requested_at = rospy.Time(0)
        self.arm_requested_at = rospy.Time(0)
        self.land_started_at = None
        self.land_commit_reached_at = None
        self.land_align_loss_announced = False
        self.auto_land_requested_at = rospy.Time(0)
        self.param_configured = False
        self.param_config_attempted = False
        self.success_logged = False
        self.pid_integral_xy = [0.0, 0.0]
        self.pid_prev_error_xy = None
        self.pid_integral_z = 0.0
        self.pid_prev_error_z = None
        self.pid_prev_time = None
        self.guided_target_xy = None
        self.land_last_guidance_at = None
        self.land_descent_gate = 0.0
        self.land_desired_z = None
        self.land_align_reference_z = None
        self.descent_target_xy = None
        self.descent_last_update_at = None

        self.target_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=20)
        self.mavlink_pub = rospy.Publisher("/mavlink/to", Mavlink, queue_size=10)
        self.mav = mavutil.mavlink.MAVLink(_MavFile(), srcSystem=250, srcComponent=190)

        rospy.Subscriber("mavros/state", State, self._state_cb, queue_size=10)
        rospy.Subscriber("mavros/extended_state", ExtendedState, self._extended_state_cb, queue_size=10)
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self._local_pose_cb, queue_size=10)
        rospy.Subscriber("/aruco/pose", PoseStamped, self._aruco_pose_cb, queue_size=10)

        rospy.loginfo("waiting for MAVROS services")
        rospy.wait_for_service("mavros/set_mode", 30)
        rospy.wait_for_service("mavros/cmd/arming", 30)
        rospy.wait_for_service("mavros/param/set", 30)
        self.set_mode_srv = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arm_srv = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.param_set_srv = rospy.ServiceProxy("mavros/param/set", ParamSet)
        rospy.loginfo("MAVROS services are ready")

    def _state_cb(self, msg):
        if self.state.mode != msg.mode:
            rospy.loginfo("FCU mode: %s -> %s", self.state.mode, msg.mode)
        if self.state.armed != msg.armed:
            rospy.loginfo("FCU armed: %s -> %s", self.state.armed, msg.armed)
        self.state = msg

    def _extended_state_cb(self, msg):
        self.extended_state = msg

    def _local_pose_cb(self, msg):
        self.local_pose = msg
        if self.home_pose is None:
            self.home_pose = PoseStamped()
            self.home_pose.pose = msg.pose
            self.search_waypoints = self._build_search_waypoints()
            self.control_started_at = rospy.Time.now()
            rospy.loginfo(
                "home pose captured at x=%.2f y=%.2f z=%.2f, planned %d search waypoints",
                self.home_pose.pose.position.x,
                self.home_pose.pose.position.y,
                self.home_pose.pose.position.z,
                len(self.search_waypoints),
            )

    def _aruco_pose_cb(self, msg):
        if self.local_pose is None:
            return

        marker_world = self._marker_world_from_camera(msg)
        self.last_detection_time = rospy.Time.now()
        self.last_relative_pose = msg.pose
        self.recent_detections.append((self.last_detection_time, marker_world))
        self._update_marker_world_estimate()
        self._update_landing_target_xy()

        if self.flight_phase in ("TAKEOFF", "SEARCH"):
            previous_phase = self.flight_phase
            if self._recent_detection_count() >= self.detection_confirm_count:
                self.flight_phase = "TRACK"
                self.success_hits = 0
                self.land_align_hits = 0
                self._reset_visual_guidance()
                rospy.loginfo(
                    "ArUco confirmed during %s, switching to TRACK at world estimate x=%.2f y=%.2f",
                    "takeoff" if previous_phase == "TAKEOFF" else "search",
                    self.marker_world_estimate[0],
                    self.marker_world_estimate[1],
                )

        if self.flight_phase in ("TRACK", "LAND_ALIGN", "DESCEND"):
            if abs(msg.pose.position.x) <= self.center_tolerance and abs(msg.pose.position.y) <= self.center_tolerance:
                self.success_hits += 1
            else:
                self.success_hits = 0

            if (
                abs(msg.pose.position.x) <= self.land_center_tolerance
                and abs(msg.pose.position.y) <= self.land_center_tolerance
            ):
                self.land_align_hits += 1
                self.land_align_loss_announced = False
            else:
                self.land_align_hits = 0

    def _build_search_waypoints(self):
        home_x = self.home_pose.pose.position.x
        home_y = self.home_pose.pose.position.y
        radius_steps = max(1, int(math.ceil(self.search_radius / self.search_spacing)))
        offsets = [(0, 0)]

        for ring in range(1, radius_steps + 1):
            x = ring
            y = -(ring - 1)
            while y <= ring:
                offsets.append((x, y))
                y += 1
            x = ring - 1
            y = ring
            while x >= -ring:
                offsets.append((x, y))
                x -= 1
            x = -ring
            y = ring - 1
            while y >= -ring:
                offsets.append((x, y))
                y -= 1
            x = -(ring - 1)
            y = -ring
            while x <= ring:
                offsets.append((x, y))
                x += 1

        waypoints = []
        seen = set()
        for dx_step, dy_step in offsets:
            key = (dx_step, dy_step)
            if key in seen:
                continue
            seen.add(key)
            waypoints.append(
                (
                    home_x + dx_step * self.search_spacing,
                    home_y + dy_step * self.search_spacing,
                )
            )
        return waypoints

    def _marker_world_from_camera(self, msg):
        position = self.local_pose.pose.position
        rotation_wb = quat_to_matrix(self.local_pose.pose.orientation)
        t_ca = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        t_ba = mat_vec_mul(TBC_ROT, t_ca)
        t_wa_offset = mat_vec_mul(rotation_wb, t_ba)
        return (
            position.x + t_wa_offset[0],
            position.y + t_wa_offset[1],
            position.z + t_wa_offset[2],
        )

    def _update_marker_world_estimate(self):
        now = rospy.Time.now()
        recent = [
            detection
            for stamp, detection in self.recent_detections
            if (now - stamp).to_sec() <= self.marker_estimate_window
        ]
        if not recent:
            return
        filtered_estimate = (
            self._trimmed_mean(item[0] for item in recent),
            self._trimmed_mean(item[1] for item in recent),
            self._trimmed_mean(item[2] for item in recent),
        )
        if self.marker_world_estimate is None:
            self.marker_world_estimate = filtered_estimate
            return
        alpha = clamp(self.marker_estimate_alpha, 0.0, 1.0)
        self.marker_world_estimate = tuple(
            previous + alpha * (current - previous)
            for previous, current in zip(self.marker_world_estimate, filtered_estimate)
        )

    def _trimmed_mean(self, values):
        ordered = sorted(values)
        if not ordered:
            return 0.0
        trim_count = min(
            int(len(ordered) * self.marker_estimate_trim_ratio),
            max(0, (len(ordered) - 1) // 2),
        )
        if trim_count > 0:
            ordered = ordered[trim_count : len(ordered) - trim_count]
        return sum(ordered) / len(ordered)

    def _recent_detection_count(self):
        now = rospy.Time.now()
        return sum(1 for stamp, _ in self.recent_detections if (now - stamp).to_sec() <= 1.5)

    def _target_altitude(self):
        return self.home_pose.pose.position.z + self.takeoff_altitude

    def _publish_target(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        qx, qy, qz, qw = yaw_to_quaternion(self.fixed_yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.target_pub.publish(msg)

    def _configure_offboard_failsafe(self):
        if self.param_configured or self.param_config_attempted:
            return
        self.param_config_attempted = True
        value = ParamValue(integer=(1 << 2), real=0.0)
        try:
            response = self.param_set_srv("COM_RCL_EXCEPT", value)
        except rospy.ServiceException as exc:
            rospy.logwarn("skipping COM_RCL_EXCEPT setup: %s", exc)
            self.param_configured = True
            return
        if not response.success:
            rospy.logwarn("failed to set COM_RCL_EXCEPT, continuing without it")
            self.param_configured = True
            return
        self.param_configured = True
        rospy.loginfo("COM_RCL_EXCEPT set to %d to allow RC-less OFFBOARD", 1 << 2)

    def _publish_mavlink(self, mav_msg):
        mav_msg.pack(self.mav)
        self.mavlink_pub.publish(mavlink_convert.convert_to_rosmsg(mav_msg))

    def _arm_via_raw_mavlink(self):
        mav_msg = mavutil.mavlink.MAVLink_command_long_message(
            target_system=1,
            target_component=1,
            command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            confirmation=0,
            param1=1.0,
            param2=0.0,
            param3=0.0,
            param4=0.0,
            param5=0.0,
            param6=0.0,
            param7=0.0,
        )
        self._publish_mavlink(mav_msg)

    def _maybe_enable_offboard_and_arm(self):
        if not self.auto_arm_offboard:
            return

        if self.flight_phase in ("AUTO_LAND", "SUCCESS"):
            return

        now = rospy.Time.now()
        if self.state.mode != "OFFBOARD":
            if (now - self.offboard_requested_at).to_sec() >= self.offboard_retry_period:
                response = self.set_mode_srv(0, "OFFBOARD")
                if response.mode_sent:
                    rospy.loginfo("OFFBOARD request sent")
                else:
                    rospy.logwarn("OFFBOARD request rejected by MAVROS")
                self.offboard_requested_at = now
            return

        if not self.state.armed:
            if (now - self.arm_requested_at).to_sec() >= self.arm_retry_period:
                response = self.arm_srv(True)
                if response.success:
                    rospy.loginfo("arm request sent")
                else:
                    rospy.logwarn("arm request rejected by MAVROS, sending raw MAVLink arm command")
                    self._arm_via_raw_mavlink()
                self.arm_requested_at = now

    def _distance_to(self, x, y, z):
        if self.local_pose is None:
            return float("inf"), float("inf")
        pos = self.local_pose.pose.position
        lateral = math.hypot(pos.x - x, pos.y - y)
        vertical = abs(pos.z - z)
        return lateral, vertical

    def _current_xy(self):
        position = self.local_pose.pose.position
        return position.x, position.y

    def _reset_visual_guidance(self, keep_target=False):
        self.pid_integral_xy = [0.0, 0.0]
        self.pid_prev_error_xy = None
        self.pid_integral_z = 0.0
        self.pid_prev_error_z = None
        self.pid_prev_time = None
        self.land_last_guidance_at = None
        self.land_descent_gate = 0.0
        self.land_desired_z = None
        if not keep_target:
            self.guided_target_xy = None

    def _current_visual_error(self):
        if self.last_relative_pose is None:
            return float("inf")
        return math.hypot(self.last_relative_pose.position.x, self.last_relative_pose.position.y)

    def _phase_pid_settings(self, phase):
        output_scale, step_scale = self._low_altitude_guidance_scale(phase)
        if phase == "LAND_ALIGN":
            return (
                self.land_pid_kp,
                self.land_pid_ki,
                self.land_pid_kd,
                self.land_output_limit * output_scale,
                self.land_target_alpha,
                self.land_target_step_limit * step_scale,
            )
        return (
            self.track_pid_kp,
            self.track_pid_ki,
            self.track_pid_kd,
            self.track_output_limit * output_scale,
            self.track_target_alpha,
            self.track_target_step_limit * step_scale,
        )

    def _low_altitude_guidance_scale(self, phase):
        if self.local_pose is None or self.home_pose is None:
            return 1.0, 1.0

        slowdown_start = max(self.low_altitude_xy_slowdown_start, self.landing_commit_altitude + 0.05)
        altitude_ratio = clamp(max(self._above_ground(), 0.0) / slowdown_start, 0.0, 1.0)

        if phase == "LAND_ALIGN":
            output_floor = self.land_low_altitude_xy_scale_min
            step_floor = self.land_low_altitude_step_scale_min
        else:
            output_floor = self.track_low_altitude_xy_scale_min
            step_floor = self.track_low_altitude_step_scale_min

        output_scale = output_floor + (1.0 - output_floor) * altitude_ratio
        step_scale = step_floor + (1.0 - step_floor) * altitude_ratio
        return output_scale, step_scale

    def _blend_target_xy(self, commanded_x, commanded_y, alpha, step_limit):
        if self.guided_target_xy is None:
            self.guided_target_xy = (commanded_x, commanded_y)
            return self.guided_target_xy

        prev_x, prev_y = self.guided_target_xy
        blended_x = prev_x + alpha * (commanded_x - prev_x)
        blended_y = prev_y + alpha * (commanded_y - prev_y)

        step_dx = blended_x - prev_x
        step_dy = blended_y - prev_y
        step_norm = math.hypot(step_dx, step_dy)
        if step_limit > 0.0 and step_norm > step_limit:
            scale = step_limit / step_norm
            blended_x = prev_x + step_dx * scale
            blended_y = prev_y + step_dy * scale

        self.guided_target_xy = (blended_x, blended_y)
        return self.guided_target_xy

    def _guided_visual_target(self, measurement_xy, phase):
        current_x, current_y = self._current_xy()
        error_x = measurement_xy[0] - current_x
        error_y = measurement_xy[1] - current_y

        now = rospy.Time.now()
        dt = 0.0
        if self.pid_prev_time != rospy.Time(0) and self.pid_prev_time is not None:
            dt = (now - self.pid_prev_time).to_sec()
        if dt < 0.0 or dt > 0.5:
            dt = 0.0

        kp, ki, kd, output_limit, alpha, step_limit = self._phase_pid_settings(phase)

        if dt > 0.0:
            self.pid_integral_xy[0] = clamp(
                self.pid_integral_xy[0] + error_x * dt,
                -self.pid_integral_limit,
                self.pid_integral_limit,
            )
            self.pid_integral_xy[1] = clamp(
                self.pid_integral_xy[1] + error_y * dt,
                -self.pid_integral_limit,
                self.pid_integral_limit,
            )
            prev_error_x, prev_error_y = self.pid_prev_error_xy or (0.0, 0.0)
            derivative_x = (error_x - prev_error_x) / dt
            derivative_y = (error_y - prev_error_y) / dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0

        output_x = kp * error_x + ki * self.pid_integral_xy[0] + kd * derivative_x
        output_y = kp * error_y + ki * self.pid_integral_xy[1] + kd * derivative_y
        output_norm = math.hypot(output_x, output_y)
        if output_limit > 0.0 and output_norm > output_limit:
            scale = output_limit / output_norm
            output_x *= scale
            output_y *= scale

        commanded_x = current_x + output_x
        commanded_y = current_y + output_y
        commanded_x, commanded_y = self._clamp_marker_xy(commanded_x, commanded_y)

        self.pid_prev_error_xy = (error_x, error_y)
        self.pid_prev_time = now
        target_x, target_y = self._blend_target_xy(commanded_x, commanded_y, alpha, step_limit)
        return target_x, target_y, math.hypot(error_x, error_y)

    def _descent_gate_target(self, lateral_error, marker_visible):
        if not marker_visible and self._above_ground() > self.landing_commit_altitude + self.land_commit_tolerance:
            raw_gate = 0.0
        elif lateral_error <= self.land_slow_error:
            raw_gate = 1.0
        elif lateral_error >= self.land_hold_error:
            raw_gate = 0.0
        else:
            span = max(self.land_hold_error - self.land_slow_error, 1e-3)
            raw_gate = (self.land_hold_error - lateral_error) / span

        raw_gate = clamp(raw_gate, 0.0, 1.0)
        return raw_gate * (
            self.land_slow_descent_factor + (1.0 - self.land_slow_descent_factor) * raw_gate
        )

    def _guided_land_altitude(self, current_z, lateral_error, marker_visible, dt):
        commit_z = self.home_pose.pose.position.z + self.landing_commit_altitude
        if self.land_desired_z is None:
            self.land_desired_z = current_z
        self.land_desired_z = max(min(self.land_desired_z, current_z), commit_z)

        target_gate = self._descent_gate_target(lateral_error, marker_visible)
        if target_gate <= 1e-3:
            self.land_desired_z = current_z
        gate_alpha = clamp(self.land_descent_gate_alpha, 0.0, 1.0)
        self.land_descent_gate += gate_alpha * (target_gate - self.land_descent_gate)
        self.land_descent_gate = clamp(self.land_descent_gate, 0.0, 1.0)

        if dt > 0.0 and self.land_descent_gate > 0.0:
            self.land_desired_z = max(
                self.land_desired_z - self.land_descent_rate * self.land_descent_gate * dt,
                commit_z,
            )

        altitude_error = self.land_desired_z - current_z
        if dt > 0.0:
            if self.land_descent_gate > 1e-3:
                self.pid_integral_z = clamp(
                    self.pid_integral_z + altitude_error * dt,
                    -self.land_z_integral_limit,
                    self.land_z_integral_limit,
                )
            else:
                self.pid_integral_z *= 0.6
            previous_error = self.pid_prev_error_z if self.pid_prev_error_z is not None else altitude_error
            derivative = (altitude_error - previous_error) / dt
        else:
            derivative = 0.0

        z_adjust = (
            self.land_z_pid_kp * altitude_error
            + self.land_z_pid_ki * self.pid_integral_z
            + self.land_z_pid_kd * derivative
        )
        z_adjust = clamp(z_adjust, -self.land_z_output_limit, 0.0)
        self.pid_prev_error_z = altitude_error

        desired_z = clamp(current_z + z_adjust, commit_z, current_z)
        return desired_z

    def _clamp_marker_xy(self, x, y):
        search_limit = self.search_radius + self.search_spacing
        home_x = self.home_pose.pose.position.x
        home_y = self.home_pose.pose.position.y
        clamped_x = clamp(x, home_x - search_limit, home_x + search_limit)
        clamped_y = clamp(y, home_y - search_limit, home_y + search_limit)
        return clamped_x, clamped_y

    def _update_landing_target_xy(self):
        if self.marker_world_estimate is None or self.home_pose is None:
            return
        marker_x, marker_y = self._clamp_marker_xy(
            self.marker_world_estimate[0],
            self.marker_world_estimate[1],
        )
        self.landing_target_xy = (marker_x, marker_y)

    def _landing_target(self):
        if self.landing_target_xy is not None:
            return self.landing_target_xy
        if self.marker_world_estimate is not None:
            return self._clamp_marker_xy(
                self.marker_world_estimate[0],
                self.marker_world_estimate[1],
            )
        if self.descent_target_xy is not None:
            return self.descent_target_xy
        return self.home_pose.pose.position.x, self.home_pose.pose.position.y

    def _marker_recently_visible(self):
        if self.last_detection_time is None:
            return False
        return (rospy.Time.now() - self.last_detection_time).to_sec() <= self.track_lost_timeout

    def _above_ground(self):
        return self.local_pose.pose.position.z - self.home_pose.pose.position.z

    def _start_land_align(self):
        target_x, target_y = self._landing_target()
        self.flight_phase = "LAND_ALIGN"
        self.land_started_at = None
        self.land_commit_reached_at = None
        self.land_align_hits = 0
        self.land_align_loss_announced = False
        self.land_align_reference_z = self.local_pose.pose.position.z
        self.landing_target_xy = (target_x, target_y)
        self.descent_target_xy = None
        self._reset_visual_guidance(keep_target=True)
        self.guided_target_xy = (target_x, target_y)
        rospy.loginfo(
            "marker centered consistently, switching to LAND_ALIGN at x=%.2f y=%.2f",
            target_x,
            target_y,
        )

    def _start_vertical_descent(self):
        self.flight_phase = "DESCEND"
        self.descent_target_xy = self._landing_target()
        self.land_desired_z = self.local_pose.pose.position.z
        self.descent_last_update_at = rospy.Time.now()
        self.guided_target_xy = None
        rospy.loginfo(
            "landing target locked at x=%.2f y=%.2f, switching to DESCEND",
            self.descent_target_xy[0],
            self.descent_target_xy[1],
        )

    def _desired_search_target(self):
        target_z = self._target_altitude()
        waypoint_x, waypoint_y = self.search_waypoints[self.search_index]
        lateral, vertical = self._distance_to(waypoint_x, waypoint_y, target_z)

        if lateral <= self.position_tolerance and vertical <= self.altitude_tolerance:
            if self.search_arrival_time is None:
                self.search_arrival_time = rospy.Time.now()
            elif (rospy.Time.now() - self.search_arrival_time).to_sec() >= self.search_dwell_time:
                previous_index = self.search_index
                self.search_index = (self.search_index + 1) % len(self.search_waypoints)
                self.search_arrival_time = None
                next_x, next_y = self.search_waypoints[self.search_index]
                rospy.loginfo(
                    "search waypoint %d reached, advancing to %d at x=%.2f y=%.2f",
                    previous_index,
                    self.search_index,
                    next_x,
                    next_y,
                )
        else:
            self.search_arrival_time = None

        waypoint_x, waypoint_y = self.search_waypoints[self.search_index]
        return waypoint_x, waypoint_y, target_z

    def _desired_track_target(self):
        if self.marker_world_estimate is None:
            home_x = self.home_pose.pose.position.x
            home_y = self.home_pose.pose.position.y
            return home_x, home_y, self._target_altitude()

        marker_x, marker_y = self._clamp_marker_xy(
            self.marker_world_estimate[0],
            self.marker_world_estimate[1],
        )
        target_x, target_y, _error = self._guided_visual_target((marker_x, marker_y), "TRACK")
        return target_x, target_y, self._target_altitude()

    def _desired_land_align_target(self):
        marker_visible = self._marker_recently_visible()
        marker_x, marker_y = self._landing_target()
        now = rospy.Time.now()

        if self.land_started_at is None:
            self.land_started_at = now

        self.land_last_guidance_at = now

        if marker_visible:
            target_x, target_y, lateral_error = self._guided_visual_target((marker_x, marker_y), "LAND_ALIGN")
        else:
            if self.guided_target_xy is not None:
                target_x, target_y = self.guided_target_xy
            else:
                target_x, target_y = marker_x, marker_y
            lateral_error = self._current_visual_error()

        desired_z = self.land_align_reference_z
        return target_x, target_y, desired_z, lateral_error

    def _desired_vertical_descent_target(self):
        target_x, target_y = self.descent_target_xy if self.descent_target_xy is not None else self._landing_target()
        now = rospy.Time.now()
        if self.descent_last_update_at is None:
            dt = 0.0
        else:
            dt = (now - self.descent_last_update_at).to_sec()
        self.descent_last_update_at = now

        if dt < 0.0 or dt > 0.5:
            dt = 0.0

        current_z = self.local_pose.pose.position.z
        commit_z = self.home_pose.pose.position.z + self.landing_commit_altitude
        if self.land_desired_z is None:
            self.land_desired_z = current_z
        self.land_desired_z = max(self.land_desired_z, commit_z)
        if dt > 0.0:
            self.land_desired_z = max(self.land_desired_z - self.land_descent_rate * dt, commit_z)
        return target_x, target_y, self.land_desired_z

    def _maybe_request_auto_land(self):
        if self.state.mode == "AUTO.LAND":
            return True

        now = rospy.Time.now()
        if (now - self.auto_land_requested_at).to_sec() < self.offboard_retry_period:
            return False

        response = self.set_mode_srv(0, "AUTO.LAND")
        if response.mode_sent:
            rospy.loginfo("AUTO.LAND request sent")
        else:
            rospy.logwarn("AUTO.LAND request rejected by MAVROS")
        self.auto_land_requested_at = now
        return response.mode_sent

    def _write_success_result(self):
        if self.success_logged or self.marker_world_estimate is None:
            return

        result = {
            "success": True,
            "phase": self.flight_phase,
            "mode": self.state.mode,
            "armed": self.state.armed,
            "marker_world_estimate": {
                "x": self.marker_world_estimate[0],
                "y": self.marker_world_estimate[1],
                "z": self.marker_world_estimate[2],
            },
            "vehicle_position": {
                "x": self.local_pose.pose.position.x,
                "y": self.local_pose.pose.position.y,
                "z": self.local_pose.pose.position.z,
            },
            "relative_pose": {
                "x": self.last_relative_pose.position.x,
                "y": self.last_relative_pose.position.y,
                "z": self.last_relative_pose.position.z,
            },
        }
        with open(self.success_result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        self.success_logged = True
        rospy.loginfo(
            "search succeeded, marker centered with relative x=%.3f y=%.3f z=%.3f",
            self.last_relative_pose.position.x,
            self.last_relative_pose.position.y,
            self.last_relative_pose.position.z,
        )
        rospy.loginfo("success result written to %s", self.success_result_path)

    def spin(self):
        rate = rospy.Rate(20.0)

        while not rospy.is_shutdown():
            if not self.state.connected or self.home_pose is None:
                rate.sleep()
                continue

            self._configure_offboard_failsafe()

            if self.flight_phase == "WAIT_FOR_HOME":
                self.flight_phase = "TAKEOFF"
                rospy.loginfo("switching to TAKEOFF")

            target_x = self.home_pose.pose.position.x
            target_y = self.home_pose.pose.position.y
            target_z = self._target_altitude()

            if self.flight_phase == "TAKEOFF":
                lateral, vertical = self._distance_to(target_x, target_y, target_z)
                if (
                    self.state.armed
                    and self.state.mode == "OFFBOARD"
                    and lateral <= self.position_tolerance
                    and vertical <= self.altitude_tolerance
                ):
                    self.flight_phase = "SEARCH"
                    rospy.loginfo("takeoff complete, switching to SEARCH")

            elif self.flight_phase == "SEARCH":
                target_x, target_y, target_z = self._desired_search_target()

            elif self.flight_phase == "TRACK":
                target_x, target_y, target_z = self._desired_track_target()
                if not self._marker_recently_visible():
                    self.flight_phase = "SEARCH"
                    self.success_hits = 0
                    self.landing_target_xy = None
                    self._reset_visual_guidance()
                    rospy.logwarn("marker lost during TRACK, returning to SEARCH")
                if self.success_hits >= self.success_hold_count:
                    if self.land_after_detect:
                        self._start_land_align()
                    else:
                        self.flight_phase = "SUCCESS"
                        rospy.loginfo("marker centered consistently, switching to SUCCESS hold")

            elif self.flight_phase == "LAND_ALIGN":
                marker_visible = self._marker_recently_visible()
                target_x, target_y, target_z, lateral_error = self._desired_land_align_target()
                if not marker_visible and not self.land_align_loss_announced:
                    self.land_align_loss_announced = True
                    rospy.logwarn(
                        "marker lost during LAND_ALIGN, holding altitude until center is visible again"
                    )

                if marker_visible and self.land_align_hits >= self.land_align_hold_count:
                    self._start_vertical_descent()

            elif self.flight_phase == "DESCEND":
                marker_visible = self._marker_recently_visible()
                target_x, target_y, target_z = self._desired_vertical_descent_target()

                if not marker_visible and self._above_ground() > self.landing_commit_altitude + self.land_commit_tolerance:
                    rospy.logwarn("marker lost during DESCEND, pausing descent to re-align")
                    self._start_land_align()
                    target_x, target_y, target_z, _lateral_error = self._desired_land_align_target()
                elif (
                    marker_visible
                    and self.last_relative_pose is not None
                    and self._current_visual_error() > self.descent_realign_tolerance
                    and self._above_ground() > self.landing_commit_altitude + self.land_commit_tolerance
                ):
                    rospy.logwarn(
                        "marker drifted during DESCEND (error=%.3f m), pausing descent to re-align",
                        self._current_visual_error(),
                    )
                    self._start_land_align()
                    target_x, target_y, target_z, _lateral_error = self._desired_land_align_target()
                elif self._above_ground() <= self.landing_commit_altitude + self.land_commit_tolerance:
                    if marker_visible and self._current_visual_error() <= self.descent_realign_tolerance:
                        self.flight_phase = "AUTO_LAND"
                        self.auto_land_requested_at = rospy.Time(0)
                        rospy.loginfo("center locked and descent complete, switching to AUTO_LAND")
                    else:
                        rospy.logwarn("final center check failed, pausing descent to re-align")
                        self._start_land_align()
                        target_x, target_y, target_z, _lateral_error = self._desired_land_align_target()

            elif self.flight_phase == "AUTO_LAND":
                target_x, target_y = self.descent_target_xy if self.descent_target_xy is not None else self._landing_target()
                target_z = self.local_pose.pose.position.z
                self._maybe_request_auto_land()
                landed_or_disarmed = (not self.state.armed) or self.extended_state.landed_state == 1
                if landed_or_disarmed:
                    self.flight_phase = "SUCCESS"
                    rospy.loginfo("landing complete, switching to SUCCESS")

            elif self.flight_phase == "SUCCESS":
                target_x, target_y = self.descent_target_xy if self.descent_target_xy is not None else self._landing_target()
                target_z = self.local_pose.pose.position.z
                if self.last_relative_pose is not None and self.marker_world_estimate is not None:
                    self._write_success_result()

            self._publish_target(target_x, target_y, target_z)

            if self.control_started_at is not None:
                warmup_done = (rospy.Time.now() - self.control_started_at).to_sec() >= 2.0
                if warmup_done:
                    self._maybe_enable_offboard_and_arm()

            rate.sleep()


def main():
    rospy.init_node("aruco_search_and_detect", anonymous=False)
    controller = ArucoSearchController()
    controller.spin()


class _MavFile:
    def write(self, _buf):
        pass


if __name__ == "__main__":
    main()
