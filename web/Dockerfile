# TODO: 베이스 이미지는 docs/environment.md 확정 후 반영
FROM ros:jazzy-ros-base

# GPU 불필요. FastAPI + rclpy + faster-whisper(2차)
# TODO: requirements 설치, frontend 빌드 단계 추가

WORKDIR /ros2_ws
COPY . /ros2_ws/src/web

# TODO: colcon build 및 entrypoint 구성 (uvicorn 실행)
