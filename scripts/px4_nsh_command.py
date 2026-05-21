#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

PX4_ROOT = Path(__file__).resolve().parents[1]
MAVLINK_PYTHON = PX4_ROOT / "src" / "modules" / "mavlink" / "mavlink"
sys.path.insert(0, str(MAVLINK_PYTHON))

from pymavlink import mavutil  # noqa: E402


class MavlinkSerialPort:
    def __init__(self, portname: str, baudrate: int, devnum: int = 10):
        self.buf = ""
        self.port = devnum
        self.mav = mavutil.mavlink_connection(portname, autoreconnect=True, baud=baudrate)
        self.mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GENERIC,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )
        self.mav.wait_heartbeat()

    def write(self, text: str):
        remaining = text
        while remaining:
            chunk = remaining[:70]
            remaining = remaining[70:]
            buf = [ord(ch) for ch in chunk]
            buf.extend([0] * (70 - len(buf)))
            self.mav.mav.serial_control_send(
                self.port,
                mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
                | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
                0,
                0,
                len(chunk),
                buf,
            )

    def read_available(self, timeout: float = 0.05) -> str:
        msg = self.mav.recv_match(
            condition="SERIAL_CONTROL.count!=0",
            type="SERIAL_CONTROL",
            blocking=True,
            timeout=timeout,
        )
        if msg is None:
            return ""
        data = msg.data[: msg.count]
        return "".join(chr(x) for x in data)

    def close(self):
        self.mav.mav.serial_control_send(self.port, 0, 0, 0, 0, [0] * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("commands", nargs="+", help="PX4 nsh commands to execute")
    parser.add_argument("--port", default="127.0.0.1:14550")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--wait", type=float, default=2.0, help="Seconds to collect output")
    args = parser.parse_args()

    port = MavlinkSerialPort(args.port, args.baudrate)
    try:
        port.write("\n")
        time.sleep(0.2)
        output = []
        for command in args.commands:
            port.write(command + "\n")
            time.sleep(0.2)
        deadline = time.time() + args.wait
        while time.time() < deadline:
            chunk = port.read_available(timeout=0.1)
            if chunk:
                output.append(chunk)
        print("".join(output), end="")
    finally:
        port.close()


if __name__ == "__main__":
    raise SystemExit(main())
