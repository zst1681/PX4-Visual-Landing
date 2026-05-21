#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64


class RealtimeErrorMetrics:
    def __init__(self):
        self.local_topic = rospy.get_param("~local_topic", "/mavros/local_position/pose")
        self.setpoint_topic = rospy.get_param("~setpoint_topic", "/mavros/setpoint_position/local")
        self.home_z = None
        self.latest_local = None
        self.latest_setpoint = None

        self.altitude_pub = rospy.Publisher("~altitude_m", Float64, queue_size=10)
        self.setpoint_altitude_pub = rospy.Publisher("~setpoint_altitude_m", Float64, queue_size=10)
        self.vertical_error_pub = rospy.Publisher("~vertical_error_m", Float64, queue_size=10)
        self.x_error_pub = rospy.Publisher("~x_error_m", Float64, queue_size=10)
        self.y_error_pub = rospy.Publisher("~y_error_m", Float64, queue_size=10)
        self.horizontal_error_pub = rospy.Publisher("~horizontal_error_m", Float64, queue_size=10)

        rospy.Subscriber(self.local_topic, PoseStamped, self._local_cb, queue_size=20)
        rospy.Subscriber(self.setpoint_topic, PoseStamped, self._setpoint_cb, queue_size=20)

        rospy.loginfo(
            "Realtime metrics from %s and %s",
            self.local_topic,
            self.setpoint_topic,
        )

    def _local_cb(self, msg):
        self.latest_local = msg
        if self.home_z is None:
            self.home_z = msg.pose.position.z
        self._publish_metrics()

    def _setpoint_cb(self, msg):
        self.latest_setpoint = msg
        self._publish_metrics()

    def _publish_float(self, publisher, value):
        publisher.publish(Float64(data=value))

    def _publish_metrics(self):
        if self.latest_local is None or self.home_z is None:
            return

        local = self.latest_local.pose.position
        altitude = local.z - self.home_z
        self._publish_float(self.altitude_pub, altitude)

        if self.latest_setpoint is None:
            return

        target = self.latest_setpoint.pose.position
        setpoint_altitude = target.z - self.home_z
        x_error = local.x - target.x
        y_error = local.y - target.y
        vertical_error = local.z - target.z
        horizontal_error = math.hypot(x_error, y_error)

        self._publish_float(self.setpoint_altitude_pub, setpoint_altitude)
        self._publish_float(self.vertical_error_pub, vertical_error)
        self._publish_float(self.x_error_pub, x_error)
        self._publish_float(self.y_error_pub, y_error)
        self._publish_float(self.horizontal_error_pub, horizontal_error)


def main():
    rospy.init_node("realtime_error_metrics")
    RealtimeErrorMetrics()
    rospy.spin()


if __name__ == "__main__":
    main()
