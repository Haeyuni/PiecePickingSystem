# grasp: world_state_raw + instance_masks → 파지 후보를 채워 /world_state 최종 발행.
#
# **지금은 heuristic_pca 전략만 뜬다.** strategies/contact_graspnet.py·graspnet_baseline.py는
# 파일만 있고 미구현이라(strategies/__init__.py 레지스트리에 미등록) Open3D/PyTorch/CUDA를
# 이 이미지에 아직 넣지 않는다 — 두 전략이 실제로 구현되는 시점에 이 Dockerfile로 돌아와
# 추가한다. 지금 필요한 건 numpy뿐이다(pointcloud_utils.py, strategies/heuristic_pca.py).
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
# opencv-python-headless: /grasp/debug_image 오버레이(node.py의 검출 박스·마스크·파지 후보
# 그리기)에 쓴다. web/requirements.txt와 같은 headless 빌드 선택 이유 — GUI 없는 컨테이너에
# libGL이 필요한 비headless판을 넣을 이유가 없다. --ignore-installed는 apt python3-numpy를
# pip이 RECORD 없이 못 지우는 문제를 web과 같은 방식으로 피한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml python3-pip \
        ros-jazzy-message-filters ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages --ignore-installed opencv-python-headless

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
