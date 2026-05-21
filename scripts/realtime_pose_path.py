#!/usr/bin/env python3

from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class PoseToPathPublisher:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/mavros/local_position/pose")
        self.output_topic = rospy.get_param("~output_topic", "/trajectory/path")
        self.frame_id = rospy.get_param("~frame_id", "")
        self.max_poses = max(1, int(rospy.get_param("~max_poses", 2000)))
        self.sample_stride = max(1, int(rospy.get_param("~sample_stride", 1)))
        self.latch_output = bool(rospy.get_param("~latch_output", True))

        self._poses = deque(maxlen=self.max_poses)
        self._message_count = 0
        self._last_frame_id = self.frame_id or "map"

        self._path_pub = rospy.Publisher(
            self.output_topic,
            Path,
            queue_size=1,
            latch=self.latch_output,
        )
        self._pose_sub = rospy.Subscriber(
            self.input_topic,
            PoseStamped,
            self._pose_cb,
            queue_size=50,
        )

        rospy.loginfo(
            "Publishing path %s from pose topic %s (max_poses=%d, sample_stride=%d)",
            self.output_topic,
            self.input_topic,
            self.max_poses,
            self.sample_stride,
        )

    def _pose_cb(self, msg):
        self._message_count += 1
        if (self._message_count - 1) % self.sample_stride != 0:
            return

        effective_frame_id = self.frame_id or msg.header.frame_id or self._last_frame_id or "map"
        self._last_frame_id = effective_frame_id

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = effective_frame_id
        pose.pose = msg.pose
        self._poses.append(pose)

        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        path_msg.header.frame_id = effective_frame_id
        path_msg.poses = list(self._poses)
        self._path_pub.publish(path_msg)


def main():
    rospy.init_node("realtime_pose_path")
    PoseToPathPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
