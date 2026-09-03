# web: FastAPI + 프론트엔드 정적 자산 + (MOCK_MODE=0일 때만) rclpy 브리지
#
# **런타임 베이스는 B4에서 python:3.12-slim → ros:jazzy-ros-base로 바뀌었다.**
# B2~B3의 mock 모드는 ROS2가 전혀 필요 없어 slim으로 충분했지만, MOCK_MODE=0에서는
# ros_bridge.py가 rclpy와 sort_msgs를 import한다. 두 모드가 같은 이미지를 쓰는 편이
# "어댑터만 교체한다"는 경계(executor.py)를 실제로 증명한다 — 모드별로 이미지가 갈리면
# 그 경계는 코드에만 있고 배포에는 없는 것이 된다.
#
# 이 교체가 앱 코드 수정 없이 가능한 이유는 main.py·routers/가 rclpy를 import하지 않기
# 때문이다. ROS2를 아는 코드는 web/ros_bridge.py 하나뿐이고 MOCK_MODE=0일 때만 로드된다.
#
# **빌드 컨텍스트는 저장소 루트다** (docker-compose.yml의 context: .). sort_msgs를
# 이미지 안에서 빌드해야 하는데 그 패키지가 web/ 밖에 있기 때문이다.

# --- 1단계: 프론트엔드 빌드 -------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /build
# 의존성 설치와 소스 복사를 분리해 소스만 고쳤을 때 npm 설치 레이어를 재사용한다
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm ci || npm install
COPY web/frontend/ ./
RUN npm run build

# --- 2단계: sort_msgs 빌드 --------------------------------------------------
# 인터페이스 패키지만 빌드한다. control/perception은 web과 같은 프로세스에 뜰 일이 없고,
# web이 필요한 것은 메시지·액션 타입뿐이다.
FROM ros:jazzy-ros-base AS msgs
WORKDIR /ros2_ws
COPY sort_msgs/ src/sort_msgs/
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select sort_msgs

# --- 3단계: 런타임 ----------------------------------------------------------
FROM ros:jazzy-ros-base
WORKDIR /app

# Ubuntu 24.04는 시스템 파이썬을 외부 관리로 표시한다(PEP 668). 컨테이너 안에서는
# 격리가 이미 이미지 단위로 이뤄지므로 가상환경을 한 겹 더 두지 않는다.
# ros:jazzy-ros-base에는 pip이 없다 (rosdep/apt로 파이썬 의존성을 받는 이미지라서).
#
# rmw_cyclonedds_cpp를 함께 넣는다. 이 이미지의 기본 RMW는 FastDDS인데 워크스테이션의
# control/perception 노드는 CycloneDDS로 뜬다. 서로 다른 RMW끼리는 **토픽 목록은 보이지만
# 데이터가 흐르지 않는다** — B4 검증에서 실제로 겪은 증상이다(`ros2 topic list`에
# /world_state가 뜨는데 구독 콜백이 한 번도 안 불림). RMW는 시스템 전체가 같아야 한다.
COPY web/requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY --from=msgs /ros2_ws/install /ros2_ws/install
COPY web/web/ ./web/
COPY --from=frontend /build/dist ./frontend_dist
COPY web/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    MOCK_DIR=/data/mock \
    FRONTEND_DIST=/app/frontend_dist \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
