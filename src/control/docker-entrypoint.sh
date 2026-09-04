#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다 (src/web/docker-entrypoint.sh와 같은 패턴).
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
