#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

PX4_ROOT = Path(__file__).resolve().parents[1]
MAVLINK_PYTHON = PX4_ROOT / "src" / "modules" / "mavlink" / "mavlink"
sys.path.insert(0, str(MAVLINK_PYTHON))

import rospy
from mavros import mavlink as mavlink_convert
from mavros_msgs.msg import Mavlink, State
from pymavlink import mavutil


class _MavFile:
    def write(self, _buf):
        pass


class OffboardBridge:
    def __init__(self):
        self.state = State()
        self.pub = rospy.Publisher("/mavlink/to", Mavlink, queue_size=10)
        self.sub = rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=10)
        self.mav = mavutil.mavlink.MAVLink(_MavFile(), srcSystem=250, srcComponent=190)

    def _state_cb(self, msg):
        self.state = msg

    def _publish(self, mav_msg):
        mav_msg.pack(self.mav)
        self.pub.publish(mavlink_convert.convert_to_rosmsg(mav_msg))

    def set_mode_offboard(self):
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | \
            mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED | \
            mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
        custom_mode = mavutil.PX4_CUSTOM_MAIN_MODE_OFFBOARD << 16
        msg = mavutil.mavlink.MAVLink_set_mode_message(
            target_system=1,
            base_mode=base_mode,
            custom_mode=custom_mode,
        )
        self._publish(msg)

    def arm(self):
        msg = mavutil.mavlink.MAVLink_command_long_message(
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
        self._publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-mode-timeout", type=float, default=12.0)
    parser.add_argument("--arm-timeout", type=float, default=12.0)
    args = parser.parse_args()

    rospy.init_node("mavlink_offboard_bridge", anonymous=True)
    bridge = OffboardBridge()

    wait_deadline = time.time() + 10.0
    rate = rospy.Rate(5)
    while not rospy.is_shutdown() and not bridge.state.connected and time.time() < wait_deadline:
        rate.sleep()

    if not bridge.state.connected:
        print("FCU is not connected", file=sys.stderr)
        return 1

    set_mode_deadline = time.time() + args.set_mode_timeout
    while not rospy.is_shutdown() and bridge.state.mode != "OFFBOARD" and time.time() < set_mode_deadline:
        bridge.set_mode_offboard()
        print(f"sent OFFBOARD request, current mode={bridge.state.mode}", file=sys.stderr)
        time.sleep(0.5)

    arm_deadline = time.time() + args.arm_timeout
    while not rospy.is_shutdown() and not bridge.state.armed and time.time() < arm_deadline:
        bridge.arm()
        print(f"sent ARM request, current armed={bridge.state.armed}", file=sys.stderr)
        time.sleep(0.5)

    print(f"final state: connected={bridge.state.connected} mode={bridge.state.mode} armed={bridge.state.armed}")
    return 0 if bridge.state.mode == "OFFBOARD" and bridge.state.armed else 2


if __name__ == "__main__":
    raise SystemExit(main())
