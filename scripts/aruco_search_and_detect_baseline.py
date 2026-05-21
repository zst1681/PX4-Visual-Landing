#!/usr/bin/env python3

import rospy

from aruco_search_and_detect import ArucoSearchController


class BaselineArucoSearchController(ArucoSearchController):
    def _update_marker_world_estimate(self):
        now = rospy.Time.now()
        recent = [
            detection
            for stamp, detection in self.recent_detections
            if (now - stamp).to_sec() <= 1.5
        ]
        if not recent:
            return
        avg_x = sum(item[0] for item in recent) / len(recent)
        avg_y = sum(item[1] for item in recent) / len(recent)
        avg_z = sum(item[2] for item in recent) / len(recent)
        self.marker_world_estimate = (avg_x, avg_y, avg_z)

    def _desired_land_align_target(self):
        marker_visible = self._marker_recently_visible()
        marker_x, marker_y = self._landing_target()
        current_z = self.local_pose.pose.position.z
        now = rospy.Time.now()

        if self.land_started_at is None:
            self.land_started_at = now

        if self.land_last_guidance_at is None:
            dt = 0.0
        else:
            dt = (now - self.land_last_guidance_at).to_sec()
        if dt < 0.0 or dt > 0.5:
            dt = 0.0
        self.land_last_guidance_at = now

        if marker_visible:
            target_x, target_y, lateral_error = self._guided_visual_target((marker_x, marker_y), "LAND_ALIGN")
        else:
            if self.guided_target_xy is not None:
                target_x, target_y = self.guided_target_xy
            else:
                target_x, target_y = marker_x, marker_y
            lateral_error = self._current_visual_error()

        descent_rate = 0.0
        if marker_visible or self._above_ground() <= self.landing_commit_altitude + self.land_commit_tolerance:
            if lateral_error <= self.land_slow_error:
                descent_rate = self.land_descent_rate
            elif lateral_error <= self.land_hold_error:
                descent_rate = self.land_descent_rate * self.land_slow_descent_factor

        desired_z = current_z
        if dt > 0.0 and descent_rate > 0.0:
            desired_z = current_z - descent_rate * dt
        desired_z = max(desired_z, self.home_pose.pose.position.z + self.landing_commit_altitude)
        desired_z = max(desired_z, current_z - 0.05)
        return target_x, target_y, desired_z, lateral_error


def main():
    rospy.init_node("aruco_search_and_detect", anonymous=False)
    controller = BaselineArucoSearchController()
    controller.spin()


if __name__ == "__main__":
    main()
