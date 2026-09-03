# TODO: 베이스 이미지·CUDA/PyTorch 버전은 docs/environment.md 확정 후 반영
FROM ros:jazzy-ros-base

# GPU 필요 (YOLO11-seg 추론)
# TODO: ultralytics, opencv-python, numpy 등 requirements 설치 단계 추가

WORKDIR /ros2_ws
COPY . /ros2_ws/src/perception

# TODO: colcon build 및 entrypoint 구성
