"""명령 1건의 수명주기: 계획 → 검증 결과 확인 → 순차 실행 → 실패 시 재계획.

검증을 통과한 시퀀스는 **사람 승인 없이 자동 실행된다**(웹_인터페이스_정의서.md 1절).
브라우저는 진행 상황을 볼 뿐 실행을 막는 승인 버튼이 없으므로, 여기서 실행 전 확인해야 할
것은 `validation_status`뿐이다 — rejected면 어떤 액션도 호출하지 않는다.
"""
import asyncio
import logging
import uuid

from . import planner_client, store
from .events import hub
from .executor import SkillGoal

logger = logging.getLogger(__name__)

# 재계획 횟수 상한. 같은 명령으로 무한히 LLM을 호출하지 않기 위한 것이다(FR-09와 같은 원칙).
MAX_REPLANS = 2

# trace_id → 스냅샷. WebSocket 재연결 시 GET /api/traces/{trace_id}로 돌려줄 현재 상태.
traces: dict[str, dict] = {}


def snapshot(trace_id: str) -> dict | None:
    return traces.get(trace_id)


def _new_trace(trace_id: str, command_text: str) -> dict:
    trace = {
        "schema_version": "1.0.0",
        "trace_id": trace_id,
        "command_text": command_text,
        "validation_status": None,
        "validation_reason": None,
        "sequence_id": None,
        "steps": [],
    }
    traces[trace_id] = trace
    return trace


async def _feedback(request_id: str, phase: str, trace: dict) -> None:
    for step in trace["steps"]:
        if step["request_id"] == request_id:
            step["phase"] = phase
            step["status"] = "in_progress"
            await hub.broadcast({
                "type": "execution_progress",
                "trace_id": trace["trace_id"],
                "request_id": request_id,
                "skill": step["skill"],
                "phase": phase,
            })
            return


async def _wait_for_fresh_observation(executor, timeout_s: float = 5.0) -> None:
    """스탬프가 바뀐 새 /world_state가 들어올 때까지 기다린다(최선을 다해서, 실패해도 넘어간다).

    home 이동 직후 곧바로 `get_latest_world_state()`를 부르면 팔이 아직 시야를 가리고
    있거나 이동 전에 찍힌 오래된 스냅샷을 돌려줄 수 있다 — 재계획이 그 상태를 근거로
    LLM에 다시 물으면 방금 건드린 물체가 아직도 "안 보이는" 것으로 나온다.
    """
    before = executor.get_latest_world_state()
    before_stamp = (before or {}).get("stamp")
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        current = executor.get_latest_world_state()
        stamp = (current or {}).get("stamp")
        if stamp and stamp != before_stamp:
            return
        await asyncio.sleep(0.2)


async def run_command(trace_id: str, command_text: str, executor) -> None:
    """명령 하나를 끝까지 처리한다. 백그라운드 태스크로 실행된다."""
    trace = _new_trace(trace_id, command_text)
    previous_failure = None

    for attempt in range(MAX_REPLANS + 1):
        # place_into가 실패했을 때는 건너뛴다 — 그리퍼가 이미 물체를 쥔 채라 validator가
        # 다음 스텝으로 pick을 허용하지 않으므로(같은 물체를 다시 집을 수 없다), 재계획해도
        # 결과는 어차피 같은 물체를 다시 place_into하는 것뿐이다. place_into의 목적지는
        # world_state가 아니라 bins.yaml에서 그라운딩되므로 다시 봐야 할 것이 없다 —
        # home 왕복만 로봇을 쥔 채로 더 움직이는 시간 낭비다.
        if attempt > 0 and (previous_failure or {}).get("skill") != "place_into":
            # 재계획 전에 home으로 돌아가 시야를 비우고 새로 스캔한다 — 실패한 스텝이
            # 파지 중 물체를 건드렸거나 팔이 카메라 시야에 그대로 남아있으면, 그 상태의
            # 관측으로 재계획해봤자 물체가 "안 보이는" 것으로 나와 매번 그라운딩이
            # 거부된다(2026-09-04 실물로 확인 — 네일 pick 실패 후 재계획이 항상
            # "무엇을 가리키는지 알 수 없습니다"로 거부됐다).
            logger.info("재계획 전 home 복귀 (trace=%s)", trace_id)
            try:
                await executor.home()
            except Exception:
                logger.exception("재계획 전 home 이동 실패 (trace=%s) — 그래도 재계획은 시도한다",
                                 trace_id)
            await _wait_for_fresh_observation(executor)

        world_state = executor.get_latest_world_state()
        if world_state is None:
            trace["validation_status"] = "rejected"
            trace["validation_reason"] = "월드 상태를 받지 못했습니다 (perception 미기동)"
            await hub.broadcast({"type": "execution_result", "trace_id": trace_id,
                                 "request_id": "", "success": False,
                                 "failure_reason": "no_world_state"})
            return

        try:
            result = await planner_client.plan(
                trace_id, command_text, world_state, previous_failure,
            )
        except planner_client.PlannerUnavailable as e:
            logger.error("planner 도달 실패: %s", e)
            trace["validation_status"] = "rejected"
            trace["validation_reason"] = f"계획 서비스에 연결할 수 없습니다: {e}"
            await hub.broadcast({"type": "execution_result", "trace_id": trace_id,
                                 "request_id": "", "success": False,
                                 "failure_reason": "planner_unavailable"})
            return

        trace["sequence_id"] = result.get("sequence_id")
        trace["validation_status"] = result.get("validation_status")
        trace["validation_reason"] = result.get("validation_reason")

        if result.get("validation_status") != "approved":
            # 검증 거부 — 로봇은 움직이지 않는다.
            logger.info("검증 거부 (trace=%s): %s", trace_id, result.get("validation_reason"))
            await hub.broadcast({
                "type": "execution_result", "trace_id": trace_id, "request_id": "",
                "success": False, "failure_reason": "rejected",
                "validation_reason": result.get("validation_reason"),
            })
            return

        steps = result.get("steps", [])
        trace["steps"] = [
            {
                "request_id": f"rq-{uuid.uuid4().hex[:8]}",
                "skill": s["skill"],
                "object_id": s["object_id"],
                "bin_id": s.get("bin_id"),
                "profile": s["profile"],
                "grasp_pose": s.get("grasp_pose"),
                "gripper_width_mm": s.get("gripper_width_mm"),
                "status": "pending",
                "phase": None,
            }
            for s in steps
        ]

        failure = await _execute_steps(trace, world_state, executor)
        if failure is None:
            return  # 전부 성공

        if attempt >= MAX_REPLANS:
            logger.warning("재계획 상한 도달 (trace=%s)", trace_id)
            return
        logger.info("재계획 시도 %d (trace=%s, 사유=%s)",
                    attempt + 1, trace_id, failure["failure_reason"])
        previous_failure = failure


async def _execute_steps(trace: dict, world_state: dict, executor) -> dict | None:
    """스텝을 순서대로 실행한다. 실패하면 previous_failure 형태로 반환."""
    class_map = store.object_class_map(world_state)

    for step in trace["steps"]:
        goal = SkillGoal(
            trace_id=trace["trace_id"],
            request_id=step["request_id"],
            object_id=step["object_id"],
            profile=step["profile"],
            grasp_pose=step.get("grasp_pose"),
            gripper_width_mm=step.get("gripper_width_mm"),
            bin_id=step.get("bin_id"),
        )

        async def on_feedback(request_id, phase, _trace=trace):
            await _feedback(request_id, phase, _trace)

        if step["skill"] == "pick":
            result = await executor.call_pick(goal, on_feedback)
        else:
            result = await executor.call_place_into(goal, on_feedback)

        step["status"] = "success" if result.success else "failure"
        step["phase"] = None

        store.insert_execution_log(
            sequence_id=trace["sequence_id"],
            trace_id=trace["trace_id"],
            request_id=step["request_id"],
            object_id=step["object_id"],
            class_name=class_map.get(step["object_id"]),
            skill_name=step["skill"],
            profile_used=step["profile"],
            bin_id=step.get("bin_id"),
            grasp_pose=step.get("grasp_pose"),
            torque_trace=result.torque_trace,
            visual_verification_passed=result.visual_verification_passed,
            result="success" if result.success else "failure",
            failure_reason=result.failure_reason,
            retry_count=result.retries_used,
            cycle_time_ms=result.cycle_time_ms,
        )

        await hub.broadcast({
            "type": "execution_result",
            "trace_id": trace["trace_id"],
            "request_id": step["request_id"],
            "success": result.success,
            "failure_reason": result.failure_reason,
        })

        if not result.success:
            return {
                "sequence_id": trace["sequence_id"],
                "request_id": step["request_id"],
                "object_id": step["object_id"],
                "failure_reason": result.failure_reason,
                "skill": step["skill"],
            }

    return None


def start_command(trace_id: str, command_text: str, executor) -> asyncio.Task:
    """명령 처리를 백그라운드로 띄운다 — HTTP 응답(202)은 즉시 돌려준다."""
    return asyncio.create_task(run_command(trace_id, command_text, executor))
