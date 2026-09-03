"""planner 서비스 진입점 (FastAPI). ROS2와 무관 — web을 거쳐서만 control에 도달 (시스템명세서 3.1절).

B0: 마이그레이션 적용 + objects.yaml 시드 + /health
B1: /internal/plan (계획 생성 → 그라운딩 → 검증 → task_sequences 기록)
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import db, grounding, llm_client, seed, validator
from .schema import SCHEMA_VERSION, PlanRequest, PlanResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
        "llm_model": llm_client.model_name(),
        "prompt_version": llm_client.PROMPT_VERSION,
        "startup": _startup,
    }


def _record(req: PlanRequest, sequence_id: str, status: str, reason: str | None,
            steps: list) -> None:
    """생성된 시퀀스와 검증 결과를 남긴다 (task_sequences).

    기록 실패가 계획 응답 자체를 막지는 않는다 — 로봇 동작을 결정하는 것은 검증 결과이고,
    이력은 그 다음 문제다.
    """
    try:
        with psycopg.connect(db.dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO task_sequences (
                    sequence_id, trace_id, raw_command, world_state_snapshot,
                    generated_sequence, validation_status, validation_reason,
                    llm_model_version, prompt_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sequence_id, req.trace_id, req.command_text,
                    json.dumps(req.world_state, ensure_ascii=False),
                    json.dumps([s.model_dump() for s in steps], ensure_ascii=False),
                    status, reason,
                    llm_client.model_name(), llm_client.PROMPT_VERSION,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("task_sequences 기록 실패 (sequence_id=%s)", sequence_id)


@app.post("/internal/plan")
def internal_plan(req: PlanRequest):
    """자연어 명령 + 월드 상태 → 검증된 스킬 시퀀스 (웹_인터페이스_정의서 3절).

    rejected면 web은 ros_bridge를 호출하지 않는다 — 로봇은 움직이지 않는다.
    """
    sequence_id = str(uuid.uuid4())
    bins = grounding.load_bins()

    def rejected(reason: str, http_status: int = 200):
        _record(req, sequence_id, "rejected", reason, [])
        body = PlanResponse(
            trace_id=req.trace_id, sequence_id=sequence_id,
            validation_status="rejected", validation_reason=reason,
        )
        return JSONResponse(status_code=http_status, content=body.model_dump())

    # 1) LLM 계획 생성
    try:
        llm_plan = llm_client.plan(
            req.command_text,
            grounding.describe_for_prompt(req.world_state, bins),
            req.previous_failure.model_dump() if req.previous_failure else None,
        )
    except Exception as e:
        logger.exception("LLM 호출 실패")
        # 하위 서비스(LLM API) 장애는 503 (웹_인터페이스_정의서 6절)
        return rejected(f"계획 생성에 실패했습니다: {e}", http_status=503)

    # 2) LLM이 스스로 거부한 경우 — 그대로 전달한다
    if not llm_plan.steps:
        return rejected(llm_plan.refusal_reason or "지시를 수행할 수 없습니다")

    # 3) 검증 (그라운딩·작업공간·가반하중·전제조건·안전 게이트)
    try:
        steps = validator.validate(
            llm_plan.steps, req.world_state, bins, req.active_safety_events,
        )
    except validator.Rejected as e:
        # 검증기 거부는 422 (웹_인터페이스_정의서 6절)
        return rejected(str(e), http_status=422)

    _record(req, sequence_id, "approved", None, steps)
    return PlanResponse(
        trace_id=req.trace_id, sequence_id=sequence_id,
        validation_status="approved", steps=steps,
    )


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
