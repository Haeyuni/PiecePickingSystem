# grasp: world_state_raw + instance_masks → 파지 후보를 채워 /world_state 최종 발행.
#
# GraspNet-baseline은 이 ROS 이미지에 PyTorch/CUDA를 섞지 않는다. host에서 실행한 grasp
# strategy가 준비된 `grasp/docker/graspnet_baseline` Docker runtime을 호출한다. 이 compose
# 컨테이너에는 Docker socket/CLI를 넣지 않으므로 heuristic_pca 용도로만 사용한다.
#
# perception_common(공용 패키지, 개발계획.md 2026-09-04)에 있는 dsr_msgs2 의존성은 이미지가
# 아니라 런타임에 호스트 워크스페이스를 마운트해 해결한다 — docker-entrypoint.sh 참조.
#
# **빌드 컨텍스트는 저장소 루트다** (docker-compose.yml) — sort_msgs/perception_common이
# src/grasp 밖에 있다(src/web/Dockerfile과 같은 이유).

# --- 1단계: sort_msgs 빌드 (인터페이스 패키지) ---
FROM ros:jazzy-ros-base AS msgs
WORKDIR /ros2_ws
COPY src/sort_msgs/ src/sort_msgs/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select sort_msgs

# --- 2단계: perception_common + grasp 빌드 + 런타임 ---
FROM ros:jazzy-ros-base
WORKDIR /ros2_ws

# numpy: pointcloud_utils.py/heuristic_pca.py. python3-yaml: grasp_params.yaml 로딩.
# message_filters: package.xml 의존이지만 이 이미지 베이스에는 기본으로 없다.
# rmw_cyclonedds_cpp: 워크스테이션이 CycloneDDS라 기본 RMW(FastDDS)와 서로 안 보인다
# (src/control/Dockerfile과 같은 이유 — 실제로 겪고 고친 문제라 매 ROS2 이미지에 반복해 넣는다).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml ros-jazzy-message-filters ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY --from=msgs /ros2_ws/install /ros2_ws/install
COPY src/perception_common/ src/perception_common/
COPY src/grasp/ src/grasp/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select perception_common grasp

COPY src/grasp/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["ros2", "launch", "grasp", "grasp_launch.py"]
