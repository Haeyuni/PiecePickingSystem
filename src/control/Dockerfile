# control: rclpy 스킬 실행기(pick/place_into/home) + robot_state/safety_events 발행.
#
# **실물 경로: home은 완전히, pick/place_into는 위치제어만 구현되어 있다** (2026-09-04).
# compliance.py(접촉감지·토크판정)와 visual_verification.py는 아직 빈 스텁이라 grasp_pose/
# bin_pose를 그대로 믿고 movel_h2r + 그리퍼(/onrobot/sendCommand, dsr_motion.py 참조
# — 표준 GripperCommand 액션은 실물에서 신뢰할 수 없어 안 쓴다)로만 움직인다. MoveIt2/
# ros2_control은 여전히 안 쓴다(dsr_msgs2 액션을 직접 부른다). 순응제어가 실제로
# 붙는 시점에 이 Dockerfile로 돌아와 그 스택을 추가한다.
#
# **빌드 컨텍스트는 저장소 루트다** (docker-compose.yml) — sort_msgs·perception_common이
# src/control 밖에 있어서 그것부터 이미지 안에서 빌드해야 한다(src/web/Dockerfile과 같은 이유).

# --- 1단계: sort_msgs 빌드 (인터페이스 패키지) ---
FROM ros:jazzy-ros-base AS msgs
WORKDIR /ros2_ws
COPY src/sort_msgs/ src/sort_msgs/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select sort_msgs

# --- 2단계: perception_common + control 빌드 + 런타임 ---
FROM ros:jazzy-ros-base
WORKDIR /ros2_ws

# python3-yaml: bins.yaml/skill_params.yaml. python3-numpy: perception_common.geometry의
# 쿼터니언 변환(dsr_motion.py). onrobot_rg_msgs는 apt에 없어(vendor 패키지) 이미지에
# 안 넣고 마운트한다(docker-compose.yml, docker-entrypoint.sh 참조).
# rmw_cyclonedds_cpp: 이 이미지의 기본 RMW(FastDDS)는 워크스테이션의 CycloneDDS 노드와
# 서로 안 보인다 — src/web/Dockerfile에 있는 것과 같은 이유(RMW는 시스템 전체가 같아야 한다).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-yaml python3-numpy ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY --from=msgs /ros2_ws/install /ros2_ws/install
COPY src/perception_common/ src/perception_common/
COPY src/control/ src/control/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select perception_common control

COPY src/control/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["ros2", "launch", "control", "control_launch.py"]
