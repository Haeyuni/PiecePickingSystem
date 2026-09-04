# perception: RGB-D 손목 카메라 → YOLO11-seg 검출 → world_state_raw/instance_masks 발행.
# GPU 필요 (YOLO11-seg 추론, requirements.txt 참조).
#
# CUDA/PyTorch 조합은 이 워크스테이션에서 실측한 값을 그대로 쓴다(requirements.txt 주석,
# docs/environment.md 3절) — 드라이버·CUDA 버전이 다른 장비로 옮기면 다시 검증해야 한다.
# 이미지 자체에는 CUDA 툴킷을 넣지 않는다 — torch 휠이 필요한 CUDA 런타임을 pip 의존성으로
# 함께 받아 오고, GPU 접근은 호스트 드라이버 + nvidia-container-toolkit(docker-compose.yml의
# `deploy.resources.reservations.devices`)이 담당한다.
#
# **best.pt는 이미지에 굽지 않는다** — *.pt는 .gitignore 대상이고(D-2와 같은 원칙),
# docker-compose.yml이 ./models를 읽기 전용으로 마운트하고 PERCEPTION_MODEL로 가리킨다.
#
# **빌드 컨텍스트는 저장소 루트다** — sort_msgs/perception_common이 src/perception 밖에
# 있다(src/web/Dockerfile과 같은 이유).

# --- 1단계: sort_msgs 빌드 (인터페이스 패키지) ---
FROM ros:jazzy-ros-base AS msgs
WORKDIR /ros2_ws
COPY src/sort_msgs/ src/sort_msgs/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select sort_msgs

# --- 2단계: perception_common + perception 빌드 + 런타임 ---
FROM ros:jazzy-ros-base
WORKDIR /ros2_ws

# python3-pip: 이 이미지엔 기본으로 없다(rosdep/apt 기준 이미지라서, src/web/Dockerfile 주석 참조).
# libgl1/libglib2.0-0: opencv-python(비-headless, ultralytics 기본 의존)이 import 시점에 찾는다 —
# 없으면 "libGL.so.1을 열 수 없다"로 죽는다. rmw_cyclonedds_cpp: control/grasp와 같은 이유.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip libgl1 libglib2.0-0 ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY src/perception/requirements.txt .
# --ignore-installed: apt로 깔린 python3-numpy 등을 pip이 RECORD 파일 없이는 못 지운다
# ("installed by debian" 오류) — 지우려 들지 말고 덮어쓰게 한다.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

COPY --from=msgs /ros2_ws/install /ros2_ws/install
COPY src/perception_common/ src/perception_common/
COPY src/perception/ src/perception/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select perception_common perception

COPY src/perception/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["ros2", "launch", "perception", "perception_launch.py"]
