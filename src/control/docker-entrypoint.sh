#!/usr/bin/env bash
# ROS2 환경을 소싱한 뒤 명령을 실행한다 (src/web/docker-entrypoint.sh와 같은 패턴).
#
# home_server.py가 dsr_msgs2/action/MovejH2r을 import한다 — src/grasp/docker-entrypoint.sh와
# 같은 이유로 doosan-robot2 드라이버 워크스페이스(기본 ~/cobot2_ws)를 이미지에 굽지 않고
# 마운트해 쓴다. **dsr_msgs2 패키지 하나만** 마운트한다 — cobot2_ws/install 전체를 마운트하면
# 그 워크스페이스에 있는 (이 프로젝트의 옛 위치였던) control/perception/sort_msgs 같은
# 동명 패키지까지 딸려 들어와 방금 빌드한 이미지 대신 그 낡은 빌드를 덮어써 버린다 —
# 실제로 겪은 문제다(홈 이동 로그가 새 코드를 반영하지 않아 추적한 끝에 발견).
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
if [ -f /doosan_ws/share/dsr_msgs2/local_setup.bash ]; then
    source /doosan_ws/share/dsr_msgs2/local_setup.bash
else
    echo "경고: dsr_msgs2 local_setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "dsr_msgs2를 못 찾아 home_server가 기동에 실패한다." >&2
fi
if [ -f /onrobot_ws/share/onrobot_rg_msgs/local_setup.bash ]; then
    source /onrobot_ws/share/onrobot_rg_msgs/local_setup.bash
else
    echo "경고: onrobot_rg_msgs local_setup.bash가 없다 — DOOSAN_WS_DIR 마운트를 확인하라." \
         "그리퍼 제어(dsr_motion.py)가 기동에 실패한다." >&2
fi
exec "$@"
