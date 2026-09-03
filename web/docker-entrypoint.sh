#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다.
#
# MOCK_MODE=1이어도 소싱은 한다 — 소싱 자체는 환경변수만 바꾸고, mock 모드에서는
# executor.create_executor()가 ros_bridge를 아예 import하지 않으므로 비용이 없다.
# 모드에 따라 기동 경로가 갈리지 않게 하는 편이 낫다.
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
