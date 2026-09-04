#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다. dsr_msgs2 패키지 하나만 마운트해 source하는
# 이유는 src/grasp/docker-entrypoint.sh 참조 — 워크스페이스 전체를 마운트하면 그 안의
# 옛 control/perception/sort_msgs 빌드가 이 이미지의 새 빌드를 덮어써 버린다(실제로 겪음).
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
if [ -f /doosan_ws/share/dsr_msgs2/local_setup.bash ]; then
    source /doosan_ws/share/dsr_msgs2/local_setup.bash
else
    echo "경고: dsr_msgs2 local_setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "get_current_posx 없이는 eye-in-hand 변환을 할 수 없다." >&2
fi
exec "$@"
