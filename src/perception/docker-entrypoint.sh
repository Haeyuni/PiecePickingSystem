#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다 (src/grasp/docker-entrypoint.sh와 같은 이유로
# dsr_msgs2가 필요한 doosan 워크스페이스도 함께 source한다).
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
if [ -f /doosan_ws/setup.bash ]; then
    source /doosan_ws/setup.bash
else
    echo "경고: /doosan_ws/setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "get_current_posx 없이는 eye-in-hand 변환을 할 수 없다." >&2
fi
exec "$@"
