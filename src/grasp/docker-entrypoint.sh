#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다 (src/web/docker-entrypoint.sh와 같은 패턴).
#
# perception_common.robot_pose가 dsr_msgs2(get_current_posx 서비스 타입)를 import한다.
# dsr_msgs2는 이 저장소가 아니라 별도 워크스페이스(doosan-robot2 드라이버, 기본
# ~/cobot2_ws)에 있어 이미지에 굽지 않고 호스트 설치본을 읽기 전용으로 마운트해 쓴다
# (data/calibration, CycloneDDS 설정과 같은 원칙 — 로봇 SDK는 호스트 정책이지 앱의
# 성질이 아니다). dsr_msgs2 하나만 마운트하지 않고 워크스페이스 전체(install/)를
# source하는 이유는 typesupport .so 등 dsr_msgs2의 나머지 의존 패키지까지 AMENT_PREFIX_PATH/
# LD_LIBRARY_PATH에 실려야 import가 끝까지 성공하기 때문이다 — 호스트에서
# `source ~/cobot2_ws/install/setup.bash`로 확인한 것과 같다.
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
if [ -f /doosan_ws/setup.bash ]; then
    source /doosan_ws/setup.bash
else
    echo "경고: /doosan_ws/setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "get_current_posx 조회 없이는 grasp가 base 좌표를 낼 수 없다." >&2
fi
exec "$@"
