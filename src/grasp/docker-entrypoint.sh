#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다 (src/web/docker-entrypoint.sh와 같은 패턴).
#
# perception_common.robot_pose가 dsr_msgs2(get_current_posx 서비스 타입)를 import한다.
# dsr_msgs2는 이 저장소가 아니라 별도 워크스페이스(doosan-robot2 드라이버, 기본
# ~/cobot2_ws)에 있어 이미지에 굽지 않고 호스트 설치본을 읽기 전용으로 마운트해 쓴다
# (data/calibration, CycloneDDS 설정과 같은 원칙 — 로봇 SDK는 호스트 정책이지 앱의
# 성질이 아니다). **dsr_msgs2 패키지 하나만** 마운트한다(워크스페이스 전체 setup.bash를
# source했다가 그 워크스페이스의 옛 control/perception/sort_msgs 빌드가 이 이미지의 새
# 빌드를 덮어써 버리는 사고를 실제로 겪었다). dsr_msgs2는 rclcpp/std_msgs/builtin_interfaces
# 정도만 의존해 이 이미지의 ROS 베이스만으로 충분하고, 자기 패키지의 local_setup.bash로도
# typesupport .so를 포함해 필요한 게 다 실린다 — 워크스페이스 전체를 끌어올 이유가 없다.
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
if [ -f /doosan_ws/share/dsr_msgs2/local_setup.bash ]; then
    source /doosan_ws/share/dsr_msgs2/local_setup.bash
else
    echo "경고: dsr_msgs2 local_setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "get_current_posx 조회 없이는 grasp가 base 좌표를 낼 수 없다." >&2
fi
exec "$@"
