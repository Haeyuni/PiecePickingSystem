# web: FastAPI + (MOCK_MODE=0일 때만) rclpy 브리지
#
# **B2 시점의 베이스는 python:3.12-slim이다.** mock 모드에서는 ROS2가 전혀 필요 없고,
# ros:jazzy 베이스는 sort_msgs colcon 빌드까지 딸려와 개발 반복을 느리게 만든다.
# B4(ROS2 E2E)에서 베이스를 ros:jazzy-ros-base로 바꾸고 sort_msgs를 빌드해 넣는다.
#
# 이 교체가 가능한 이유는 main.py·routers/가 rclpy를 import하지 않기 때문이다.
# ROS2를 아는 코드는 web/ros_bridge.py 하나뿐이고, 그 모듈은 MOCK_MODE=0일 때만 로드된다.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web/ ./web/

ENV PYTHONUNBUFFERED=1 \
    MOCK_DIR=/data/mock

EXPOSE 8000
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
