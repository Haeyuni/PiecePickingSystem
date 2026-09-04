# control: rclpy 스킬 실행기(pick/place_into/home) + robot_state/safety_events 발행.
#
# **현재 코드는 FAKE_ROBOT=1 경로만 구현되어 있다** (control/main.py, robot_state_publisher.py
# 참조) — MoveIt2/doosan-robot2/ros2_control 어느 것도 import하지 않으므로 이 이미지에는
# 넣지 않는다. 실물 로봇 연동이 시작되면(시스템명세서 5.1절) 그 스택을 추가하러 돌아온다.
#
# **빌드 컨텍스트는 저장소 루트다** (docker-compose.yml) — sort_msgs가 src/control 밖에
# 있어서 그것부터 이미지 안에서 빌드해야 한다(src/web/Dockerfile과 같은 이유).

# --- 1단계: sort_msgs 빌드 (인터페이스 패키지) ---
FROM ros:jazzy-ros-base AS msgs
WORKDIR /ros2_ws
COPY src/sort_msgs/ src/sort_msgs/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select sort_msgs

# --- 2단계: control 빌드 + 런타임 ---
FROM ros:jazzy-ros-base
WORKDIR /ros2_ws

# place_server.py가 bins.yaml/skill_params.yaml을 PyYAML로 읽는다.
# rmw_cyclonedds_cpp: 이 이미지의 기본 RMW(FastDDS)는 워크스테이션의 CycloneDDS 노드와
# 서로 안 보인다 — src/web/Dockerfile에 있는 것과 같은 이유(RMW는 시스템 전체가 같아야 한다).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-yaml ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY --from=msgs /ros2_ws/install /ros2_ws/install
COPY src/control/ src/control/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select control

COPY src/control/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["ros2", "launch", "control", "control_launch.py"]
