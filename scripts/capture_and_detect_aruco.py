#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image


def collect_frame(topic: str, timeout: float):
    bridge = CvBridge()
    result = {"frame": None}

    def callback(msg):
        if result["frame"] is None:
            result["frame"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    sub = rospy.Subscriber(topic, Image, callback, queue_size=1)
    rate = rospy.Rate(20)
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and result["frame"] is None:
        if time.monotonic() > deadline:
            break
        rate.sleep()

    sub.unregister()
    return result["frame"]


def aruco_dict_candidates():
    return [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
        ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
        ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
        ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
        ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
        ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000),
        ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
        ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
        ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
        ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000),
        ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ]


def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    matches = []

    for name, dict_id in aruco_dict_candidates():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        if ids is not None and len(ids) > 0:
            matches.append((name, ids.flatten().tolist(), len(corners)))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/rgb/image_raw")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", default="/tmp/aruco_frame.png")
    args = parser.parse_args()

    rospy.init_node("capture_and_detect_aruco", anonymous=True)
    frame = collect_frame(args.topic, args.timeout)

    if frame is None:
        print(f"Timed out waiting for image on {args.topic}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved frame to {output_path}")

    matches = detect_markers(frame)
    if not matches:
        print("No ArUco markers detected in the captured frame.")
        return 2

    for name, ids, count in matches:
        print(f"{name}: ids={ids} count={count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
