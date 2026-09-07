# planner: GPU 불필요, ROS2와 무관한 순수 Python 서비스 (시스템명세서 3.1절)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# 테스트는 stdlib unittest만 쓰므로 런타임 의존성이 늘지 않는다.
# 컨테이너 안에서 그대로 돌릴 수 있어야 "내 로컬에선 됐는데"를 피한다.
COPY tests/ ./tests/
# 평가셋과 채점 스크립트도 같은 이유로 이미지에 넣는다 (개발계획 B5). 컨테이너 안에는
# psycopg와 DATABASE_URL이 이미 있어 eval_runs 기록까지 한 번에 된다 — 호스트에 파이썬
# 의존성을 따로 깔게 만들면 "점수는 냈는데 기록이 안 남는" 상태가 조용히 생긴다.
COPY eval/ ./eval/

# 마이그레이션(db/)과 물체 시드(perception/config/)는 빌드 컨텍스트(./planner) 밖이라
# 이미지에 굽지 않고 compose에서 읽기 전용 마운트한다 — 스키마·시드를 고치려고
# 이미지를 다시 빌드할 이유가 없다.
ENV PYTHONUNBUFFERED=1 \
    MIGRATIONS_DIR=/migrations \
    OBJECTS_YAML=/config/objects.yaml \
    BINS_YAML=/config/bins.yaml \
    MOCK_DIR=/data/mock

EXPOSE 8100
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8100"]
