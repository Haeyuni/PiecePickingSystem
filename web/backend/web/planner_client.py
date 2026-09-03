"""planner(/internal/plan) 호출 (웹_인터페이스_정의서.md 3절).

`web`은 planner의 내부 구조를 모른다 — HTTP 경계 하나만 안다.
"""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

PLANNER_URL = os.environ.get("PLANNER_URL", "http://planner:8100")
# LLM 왕복이라 수 초가 걸린다. 실시간 제어 루프와는 분리된 경로다(NFR-04).
TIMEOUT_S = float(os.environ.get("PLANNER_TIMEOUT_S", "120"))


class PlannerUnavailable(Exception):
    """planner 자체에 도달하지 못한 경우 — 503으로 노출한다(6절)."""


async def plan(trace_id: str, command_text: str, world_state: dict,
               previous_failure: dict | None = None,
               active_safety_events: list[dict] | None = None) -> dict:
    payload = {
        "schema_version": "1.0.0",
        "trace_id": trace_id,
        "command_text": command_text,
        "world_state": world_state,
        "active_safety_events": active_safety_events or [],
    }
    if previous_failure:
        payload["previous_failure"] = previous_failure

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.post(f"{PLANNER_URL}/internal/plan", json=payload)
    except httpx.HTTPError as e:
        raise PlannerUnavailable(str(e)) from e

    # planner는 거부도 본문에 담아 돌려준다(422/503 포함) — 상태코드만 보고 버리지 않는다.
    try:
        return response.json()
    except ValueError as e:
        raise PlannerUnavailable(f"planner 응답을 해석할 수 없습니다: {response.text[:200]}") from e
