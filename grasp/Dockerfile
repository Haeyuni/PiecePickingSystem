# TODO: 베이스 이미지·CUDA/PyTorch 버전은 docs/environment.md 확정 후 반영
# perception과 CUDA 버전 분리 (시스템명세서 3.1절)
FROM ros:jazzy-ros-base

# GPU 필요 (Open3D, PyTorch, Contact-GraspNet, GraspNet-baseline)
# TODO: requirements 설치 단계 추가

WORKDIR /ros2_ws
COPY . /ros2_ws/src/grasp

# TODO: colcon build 및 entrypoint 구성
