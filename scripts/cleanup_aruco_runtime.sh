#!/usr/bin/env bash

set -euo pipefail

PATTERNS=(
  "roslaunch px4 aruco_search_and_land_demo.launch"
  "roslaunch px4 aruco_search_and_land_weighted_demo.launch"
  "roslaunch px4 aruco_search_demo.launch"
  "roslaunch px4 aruco_detect_and_search.launch"
  "roslaunch px4 aruco_detect_and_search_weighted.launch"
  "aruco_search_and_detect.py"
  "aruco_multi_marker_det.py"
  "aruco_multi_marker_det_weighted.py"
  "mavros_node"
  "px4_sitl_default/bin/px4"
  "gzserver --verbose -e ode .*aruco_search_demo.world"
  "gazebo_ros/scripts/gzserver --verbose -e ode .*aruco_search_demo.world"
  "rosmaster --core -p 11311"
)

for pattern in "${PATTERNS[@]}"; do
  pkill -f "${pattern}" 2>/dev/null || true
done

pkill -f "gzclient" 2>/dev/null || true

rm -f /tmp/px4-sock-0

# Let ROS/Gazebo tear down and release ports.
sleep 2
