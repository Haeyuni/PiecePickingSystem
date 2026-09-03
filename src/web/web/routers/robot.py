"""로봇 직접 제어 (웹_인터페이스_정의서.md 2.6절).

자연어 해석·그라운딩이 필요 없는 결정적 동작이라 planner를 거치지 않고 executor로 곧장 간다.
`trace_id`/`sequence_id`를 발급하지 않지만, `execution_logs`에는 남겨 이력 화면에서
분류 작업과 시간순으로 같이 보이게 한다.
"""
import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/robot/stop")
async def stop(request: Request):
    """**mode와 무관하게 항상 허용된다** — busy일 때 가장 필요한 동작이므로

    2.1절의 차단 규칙이 적용되지 않는 유일한 엔드포인트다.
    """
    executor = request.app.state.executor
    cancelled = await executor.stop()
    logger.warning("정지 요청 (취소된 request_id=%s)", cancelled)

    store.insert_execution_log(
        skill_name="stop", result="success", request_id=cancelled,
    )
    return {"schema_version": "1.0.0", "stopped": True, "cancelled_request_id": cancelled}


async def _run_home(executor) -> None:
    """홈 이동을 끝까지 수행하고 이력을 남긴다. 백그라운드 태스크로 실행된다."""
    try:
        await executor.home()
        store.insert_execution_log(skill_name="home", result="success")
    except Exception as e:
        logger.exception("홈 이동 실패")
        store.insert_execution_log(skill_name="home", result="failure",
                                   failure_reason=str(e)[:200])


@router.post("/api/robot/home")
async def home(request: Request):
    """동작 중 홈이동은 위험하므로 idle일 때만 허용한다(2.1절과 같은 차단 규칙).

    **모션 완료를 기다리지 않고 202를 돌려준다.** 2.6절의 202 `moving_home`은 "이동을
    접수했다"는 뜻이지 "이동이 끝났다"가 아니다. `/api/commands`가 같은 모양이다(2.1절).

    완료를 기다리면 홈 이동이 도는 동안 로봇이 busy라는 사실을 아무도 관측할 수 없다 —
    응답이 돌아온 시점에는 이미 끝나 있기 때문이다. 그러면 2.1절 차단 규칙이 홈 이동에
    대해서는 실질적으로 없는 것이 되고, 화면도 진행 중 표시를 띄울 수 없다.
    실행 결과는 `execution_logs`와 `robot_state`로 확인한다.
    """
    executor = request.app.state.executor
    mode = executor.robot_state().get("mode", "idle")
    if mode != "idle":
        return JSONResponse(
            status_code=409,
            content={"schema_version": "1.0.0",
                     "error": {"code": "ROBOT_BUSY" if mode == "busy" else "ROBOT_NOT_IDLE",
                               "message": f"로봇이 {mode} 상태라 홈 이동할 수 없습니다"}},
        )

    asyncio.create_task(_run_home(executor))
    return JSONResponse(status_code=202,
                        content={"schema_version": "1.0.0", "status": "moving_home"})
