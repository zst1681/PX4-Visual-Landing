#!/usr/bin/env python3

import math

import rospy

from aruco_search_and_detect import ArucoSearchController


class NoPidArucoSearchController(ArucoSearchController):
    def _guided_visual_target(self, measurement_xy, phase):
        del phase
        current_x, current_y = self._current_xy()
        error_x = measurement_xy[0] - current_x
        error_y = measurement_xy[1] - current_y

        target_x, target_y = self._clamp_marker_xy(measurement_xy[0], measurement_xy[1])
        self.guided_target_xy = (target_x, target_y)

        # No PID outer loop: directly follow the visual target with only workspace clamping.
        self.pid_prev_error_xy = None
        self.pid_prev_time = rospy.Time.now()
        return target_x, target_y, math.hypot(error_x, error_y)


def main():
    rospy.init_node("aruco_search_and_detect", anonymous=False)
    controller = NoPidArucoSearchController()
    controller.spin()


if __name__ == "__main__":
    main()
