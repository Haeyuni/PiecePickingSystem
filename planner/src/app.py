"""planner 서비스 진입점 (FastAPI). ROS2와 무관 — web을 거쳐서만 control에 도달 (시스템명세서 3.1절).

B0 범위: 기동 시 마이그레이션 적용 + objects.yaml 시드 적재 + /health.
계획·검증 엔드포인트(/internal/plan)는 B1에서 붙인다.
"""
import logging
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI

from . import db, seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# 기동 시 수행한 작업 요약. /health가 그대로 노출한다.
_startup: dict = {"migrations": [], "seeded": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.connect() as conn:
        _startup["migrations"] = db.apply_migrations(conn)
        _startup["seeded"] = seed.seed_object_attributes(conn)
    logger.info("planner 기동 완료: %s", _startup)
    yield


app = FastAPI(title="piece-picking-system planner", lifespan=lifespan)


@app.get("/health")
def health():
    """DB까지 실제로 왕복해서 확인한다 — 프로세스만 살아있는 상태를 ok로 보고하지 않는다."""
    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM object_attributes")
            object_count = cur.fetchone()[0]
    except Exception as e:  # DB 장애 시 503 대신 상태를 실어 보낸다 (web이 판단)
        return {"schema_version": SCHEMA_VERSION, "status": "degraded", "db": str(e)}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "db": "ok",
        "object_attributes": object_count,
        "startup": _startup,
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
