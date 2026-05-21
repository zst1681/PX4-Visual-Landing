#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script is meant to be sourced:" >&2
  echo "  source scripts/setup_aruco_runtime.bash" >&2
  exit 1
fi

PX4_REAL_HOME="${PX4_ARUCO_USER_HOME:-${HOME}}"
PX4_DIR="${PX4_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
GAZEBO_WS="${GAZEBO_WS:-${CATKIN_WS:-${PX4_REAL_HOME}/catkin_ws}}"
ARUCO_WS="${ARUCO_WS:-${PX4_REAL_HOME}/ros_gazebo_px4_sim_ws-master}"
XTDRONE_MODELS="${XTDRONE_MODELS:-${PX4_REAL_HOME}/XTDrone/sitl_config/models}"
TMP_HOME="${PX4_ARUCO_HOME:-/tmp/px4_aruco_home}"

source_required() {
  local setup_file="$1"
  shift
  if [[ ! -f "${setup_file}" ]]; then
    echo "Missing required setup file: ${setup_file}" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${setup_file}" "$@"
}

source_optional() {
  local setup_file="$1"
  local label="$2"
  shift 2
  if [[ -f "${setup_file}" ]]; then
    # shellcheck source=/dev/null
    source "${setup_file}" "$@"
  else
    echo "Skipping optional ${label}: ${setup_file} not found" >&2
  fi
}

mkdir -p "${TMP_HOME}/.ros" "${TMP_HOME}/.gazebo"

export ROS_HOME="${TMP_HOME}/.ros"

if [[ "${PX4_ARUCO_USE_TMP_HOME:-0}" == "1" ]]; then
  export HOME="${TMP_HOME}"
fi

source_required "/opt/ros/${ROS_DISTRO}/setup.bash" || return 1
source_optional "${GAZEBO_WS}/devel/setup.bash" "Gazebo catkin workspace"
source_optional "${ARUCO_WS}/devel/setup.bash" "ArUco catkin workspace"
source_required "${PX4_DIR}/Tools/setup_gazebo.bash" "${PX4_DIR}" "${PX4_DIR}/build/px4_sitl_default" || return 1

export ROS_PACKAGE_PATH="${PX4_DIR}:${PX4_DIR}/Tools/sitl_gazebo:${ROS_PACKAGE_PATH}"

MAXI_PKG_DIR="$(rospack find maxi_aruco_det_pkg 2>/dev/null || true)"
if [[ -n "${MAXI_PKG_DIR}" ]]; then
  export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:${MAXI_PKG_DIR}/models:${MAXI_PKG_DIR}/models_for_worlds"
fi

if [[ -d "${XTDRONE_MODELS}" ]]; then
  export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:${XTDRONE_MODELS}"
fi

# Gazebo looks at HOME for logs; the temp HOME above keeps sandbox writes local.
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11345}"
