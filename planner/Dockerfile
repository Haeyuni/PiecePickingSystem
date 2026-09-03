# planner: GPU 불필요, ROS2와 무관한 순수 Python 서비스 (시스템명세서 3.1절)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# 마이그레이션(db/)과 물체 시드(perception/config/)는 빌드 컨텍스트(./planner) 밖이라
# 이미지에 굽지 않고 compose에서 읽기 전용 마운트한다 — 스키마·시드를 고치려고
# 이미지를 다시 빌드할 이유가 없다.
ENV PYTHONUNBUFFERED=1 \
    MIGRATIONS_DIR=/migrations \
    OBJECTS_YAML=/config/objects.yaml

EXPOSE 8100
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8100"]
