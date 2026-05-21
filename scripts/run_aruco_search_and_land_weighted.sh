#!/usr/bin/env bash

set -euo pipefail

PX4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PX4_DIR}"

source scripts/setup_aruco_runtime.bash

LAUNCH_FILE="aruco_search_and_land_weighted_demo.launch"
if [[ $# -gt 0 && "$1" == *.launch ]]; then
  LAUNCH_FILE="$1"
  shift
fi

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "-${LAUNCH_PID}" 2>/dev/null || true
    sleep 2
    kill -TERM "-${LAUNCH_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "-${LAUNCH_PID}" 2>/dev/null || true
  fi

  "${PX4_DIR}/scripts/cleanup_aruco_runtime.sh" >/dev/null 2>&1 || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

"${PX4_DIR}/scripts/cleanup_aruco_runtime.sh"

GUI_VALUE="${GUI:-true}"

setsid roslaunch px4 "${LAUNCH_FILE}" gui:="${GUI_VALUE}" "$@" &
LAUNCH_PID=$!
wait "${LAUNCH_PID}"
