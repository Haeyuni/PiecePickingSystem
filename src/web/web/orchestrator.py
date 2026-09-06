"""명령 1건의 수명주기: 계획 → 검증 결과 확인 → 순차 실행 → 실패 시 재계획.

검증을 통과한 시퀀스는 **사람 승인 없이 자동 실행된다**(웹_인터페이스_정의서.md 1절).
브라우저는 진행 상황을 볼 뿐 실행을 막는 승인 버튼이 없으므로, 여기서 실행 전 확인해야 할
것은 `validation_status`뿐이다 — rejected면 어떤 액션도 호출하지 않는다.
"""
import asyncio
import dataclasses
import logging
import os
import time
import uuid

from . import planner_client, store
from .events import hub
from .executor import SkillGoal, is_mock

logger = logging.getLogger(__name__)

# 재계획 횟수 상한. 같은 명령으로 무한히 LLM을 호출하지 않기 위한 것이다(FR-09와 같은 원칙).
MAX_REPLANS = 2

# place_into가 실패했을 때 **같은 스텝을** 몇 번까지 다시 보낼지.
#
# 재계획(=planner에 다시 물어 pick부터 새로)과 구분해야 한다. place_into 실패 시점의
# 로봇은 물체를 이미 쥐고 있고, 목적지는 world_state가 아니라 bins.yaml에서 나오므로
# 새로 관측하거나 계획할 것이 없다 — 필요한 것은 "그 이동을 다시 해보는 것"뿐이다.
# 실제로 재계획으로 넘겼더니 planner가 새 pick을 지시했고, pick_server가 시작할 때
# 그리퍼를 여는 바람에 쥐고 있던 물체를 엉뚱한 자리에 떨어뜨렸다(2026-09-06 실물 확인).
MAX_PLACE_RETRIES = 2

# world_state가 이보다 오래되면 계획에 쓰지 않고 새 관측을 한 번 기다린다 — perception이
# 멈췄거나(카메라 문제) 팔이 시야를 가린 채로 오래된 스냅샷을 그대로 계획에 쓰는 사고를
# 막는다. 매직넘버로 박아두지 않고 환경변수로 둔다(2026-09-05, WorldState 최신성 점검).
MAX_WORLD_STATE_AGE_S = float(os.environ.get("MAX_WORLD_STATE_AGE_S", "5.0"))

# trace_id → 스냅샷. WebSocket 재연결 시 GET /api/traces/{trace_id}로 돌려줄 현재 상태.
traces: dict[str, dict] = {}

# trace_id → 실행 중인 asyncio.Task. Stop이 "지금 활성 ROS goal"뿐 아니라 이 태스크
# 자체도 취소해야 한다 — 안 그러면 두 액션 스텝 사이(재계획 대기·planner 호출 등,
# 활성 goal이 없는 구간)에 Stop을 눌러도 아무 효과가 없어서, 잠시 후 다음 Pick이
# 그대로 나가는 사고가 있었다(2026-09-05 실물 확인 — 이 세션의 최우선 수정 사항).
_running_tasks: dict[str, asyncio.Task] = {}


def snapshot(trace_id: str) -> dict | None:
    return traces.get(trace_id)


def cancel_running() -> list[str]:
    """현재 실행 중인 모든 명령을 취소한다. 취소된 trace_id 목록을 돌려준다(없으면 [])).

    `asyncio.Task.cancel()`은 해당 코루틴이 **다음으로 멈춰있던 await 지점**(ROS 결과
    대기든, `asyncio.sleep`이든, planner에 대한 HTTP 호출이든)에서 `CancelledError`를
    던진다 — 활성 ROS goal이 있을 때만 듣는 `executor.stop()`과 달리, 재계획 사이의
    대기 구간에서도 확실히 멈춘다. 두 메커니즘은 상호보완적이라 `routers/robot.py`가
    `executor.stop()`과 이 함수를 함께 부른다."""
    cancelled_ids = []
    for trace_id, task in list(_running_tasks.items()):
        if not task.done():
            task.cancel()
            cancelled_ids.append(trace_id)
    return cancelled_ids


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


def _world_state_age_s(world_state: dict | None) -> float | None:
    """world_state["stamp"]는 {"sec", "nanosec"}인 builtin_interfaces/Time 그대로다
    (ros_bridge.py의 _world_state_to_dict, mock 픽스처도 같은 모양) — float 초가 아니므로
    직접 뺄 수 없다. use_sim_time을 안 쓰므로(이 노드들은 실물 시계 기준) time.time()과
    같은 축의 값이다."""
    stamp = (world_state or {}).get("stamp")
    if not stamp:
        return None
    return time.time() - (stamp["sec"] + stamp["nanosec"] * 1e-9)


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
    """명령 하나를 끝까지 처리한다. 백그라운드 태스크로 실행된다.

    `cancel_running()`이 이 태스크를 취소하면 `CancelledError`가 현재 멈춰있던 await
    지점(아래 for 루프 어디든)에서 던져진다 — 여기서 잡아 취소 결과를 broadcast하고
    다시 raise한다. 다시 raise하지 않으면 asyncio 입장에서 이 태스크는 "정상 종료"가
    되어 `task.cancelled()`가 False를 반환하고, 취소 여부를 확인하는 다른 코드(테스트,
    로깅)가 오판할 수 있다."""
    try:
        await _run_command_body(trace_id, command_text, executor)
    except asyncio.CancelledError:
        logger.info("명령 취소됨 (trace=%s)", trace_id)
        await hub.broadcast({
            "type": "execution_result", "trace_id": trace_id, "request_id": "",
            "success": False, "failure_reason": "cancelled", "cancelled": True,
        })
        raise


async def _run_command_body(trace_id: str, command_text: str, executor) -> None:
    trace = _new_trace(trace_id, command_text)
    previous_failure = None

    for attempt in range(MAX_REPLANS + 1):
        # 여기 오는 재계획은 pick 계열 실패뿐이다 — place_into 실패는 아래에서 재시도로
        # 처리하고 재계획 없이 끝내기 때문이다(MAX_PLACE_RETRIES 주석 참조).
        if attempt > 0:
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

        # mock 픽스처는 시나리오 재현용 고정 스냅샷이라 stamp가 실제 "지금"을 반영하지
        # 않는다(예: data/mock/world_state_normal.json은 stamp가 고정값이다) — 실물이
        # 아니므로 신선도가 안전과 무관하고, 검사하면 모든 mock 테스트가 매번
        # stale_world_state로 거부된다. 실물(RosExecutor)에서만 의미 있는 검사다.
        age_s = _world_state_age_s(world_state)
        trace["world_state_age_s"] = age_s
        if not is_mock() and age_s is not None and age_s > MAX_WORLD_STATE_AGE_S:
            # 오래된 스냅샷으로 그대로 계획하지 않는다 — 한 번 더 새 관측을 기다려보고,
            # 그래도 안 오면(카메라/perception 문제) 사용자에게 명확히 알리고 멈춘다.
            # home 복귀와 별개다: 여기는 애초에 최초 관측이 오래된 경우도 잡는다.
            logger.warning("world_state가 오래됨 (%.1fs > %.1fs, trace=%s) — 새 관측 대기",
                            age_s, MAX_WORLD_STATE_AGE_S, trace_id)
            await _wait_for_fresh_observation(executor)
            world_state = executor.get_latest_world_state()
            age_s = _world_state_age_s(world_state)
            trace["world_state_age_s"] = age_s
            if world_state is None or age_s is None or age_s > MAX_WORLD_STATE_AGE_S:
                trace["validation_status"] = "rejected"
                trace["validation_reason"] = (
                    f"월드 상태가 오래되었습니다 ({age_s:.1f}s 전, perception 확인 필요)"
                    if age_s is not None else "월드 상태를 받지 못했습니다 (perception 미기동)"
                )
                await hub.broadcast({"type": "execution_result", "trace_id": trace_id,
                                     "request_id": "", "success": False,
                                     "failure_reason": "stale_world_state"})
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

        if failure.get("cancelled"):
            # 사용자가 Stop을 눌러 취소된 것 — "실패"가 아니라 "그만둠"이다. 재계획하면
            # 안 된다: 취소를 재계획 사유로 넘기면 곧바로 다음 Pick이 다시 나가버려서,
            # 이 세션에서 조사한 "Stop 눌러도 잠시 후 재실행" 사고가 그대로 재현된다.
            logger.info("실행 취소로 종료, 재계획하지 않음 (trace=%s)", trace_id)
            return

        if failure.get("skill") == "place_into":
            # 여기까지 왔다는 건 _execute_steps의 place_into 재시도까지 다 실패했다는 뜻이다.
            # 그래도 **재계획은 하지 않는다.** 이 시점의 그리퍼는 물체를 쥔 채이고, 재계획은
            # 필연적으로 새 pick을 부른다 — pick_server는 시작할 때 그리퍼를 열므로(이전
            # 사이클 잔여 상태 대응 로직) 쥐고 있던 물체를 팔이 멈춰 있던 자리에 그대로
            # 떨어뜨린 뒤 다른 물체를 잡으러 간다. 2026-09-06 실물에서 정확히 그렇게 됐다:
            # 명령 한 번에 pick → place 실패 → (물체 낙하) → pick → place → home 이 통째로
            # 다시 돌았다. 사람이 물체 상태를 확인해야 하는 상황이므로 여기서 멈춘다.
            logger.warning("place_into가 재시도 후에도 실패 — 그리퍼가 물체를 쥔 채일 수 있어 "
                           "재계획하지 않고 멈춘다 (trace=%s)", trace_id)
            await hub.broadcast({
                "type": "execution_result", "trace_id": trace_id,
                "request_id": failure.get("request_id", ""), "success": False,
                "failure_reason": "place_failed_holding_object",
            })
            return

        if failure.get("skill") == "pick" and failure.get("failure_reason") == "unreachable":
            # pick_server.py의 _LiftFailedError만 pick 실패를 REASON_UNREACHABLE로
            # 보고한다 — 그리퍼를 이미 닫은(물었을 수 있는) *뒤*에 들어올리기·개폭 측정
            # 등이 실패한 경우로, "확실히 못 집었다"(grasp_failed)와 다르다. 이 상태로
            # home 복귀 후 재계획하면 planner가 새 world_state를 보고 다른 물체를
            # 새로 집으라고 지시할 수 있는데, 실제로는 방금 문 물체를 그리퍼가 아직
            # 들고 있을 수 있어 다음 pick 시도가 충돌·낙하로 이어질 위험이 있다
            # (2026-09-05, "상승은 됐는데 실패 판정" 조사에서 함께 확인한 안전 요구사항).
            # control이 이미 store.set_error()로 로봇을 error 모드로 두므로(사람이
            # home/리셋해야 풀림) 여기서는 재계획 없이 그대로 멈춘다.
            logger.warning("Pick 들어올리기 이후 실패 — 그리퍼 상태 불확실, 재계획하지 않음 "
                           "(trace=%s)", trace_id)
            await hub.broadcast({
                "type": "execution_result", "trace_id": trace_id,
                "request_id": failure.get("request_id", ""), "success": False,
                "failure_reason": "gripper_state_uncertain",
            })
            return

        if executor.robot_state().get("mode") == "error":
            # pick의 _LiftFailedError만 error 모드로 두는 게 아니다 — place_server의
            # 일반 예외 처리(_place_real 도중 아무 실패)도 store.set_error()를 부른다
            # (control/place_server.py execute_callback). 위 pick 전용 분기는 그중
            # 한 경우만 잡아서, place_into가 그리퍼를 아직 쥔 채(release 전) 실패했을
            # 때는 이 일반 체크가 처음으로 걸러준다.
            #
            # 이 체크가 없으면 무슨 일이 일어나는지: place_into가 물체를 쥔 채로
            # 실패해도(예: release 전 자세 확인 실패) 여기 도달 전까지는 그냥 "실패한
            # place_into"로만 보여서 곧바로 재계획한다. planner가 world_state를 보고
            # 다른 물체를 pick하라고 시키면, pick_server는 그리퍼가 이미 뭔가를 쥔 채로
            # "그리퍼를 먼저 연다"(이전 사이클 잔여 상태 대응 로직) 때문에 **방금 집은
            # 물체를 팔이 멈춰 있던 엉뚱한 자리에 떨어뜨리고** 새 물체를 잡으러 간다
            # (2026-09-06 실물 확인 — place_into가 즉시 실패한 뒤 "이상한 위치에서
            # 재파지"로 나타났다). control이 이미 error 모드로 사람 개입을 요구하고
            # 있으므로, 여기서도 그 신호를 그대로 존중해 재계획을 멈춘다.
            logger.warning(
                "control이 error 모드 — 그리퍼 상태를 신뢰할 수 없어 재계획하지 않음 "
                "(trace=%s, skill=%s)", trace_id, failure.get("skill"))
            await hub.broadcast({
                "type": "execution_result", "trace_id": trace_id,
                "request_id": failure.get("request_id", ""), "success": False,
                "failure_reason": "robot_error_state",
            })
            return

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
            for retry in range(1, MAX_PLACE_RETRIES + 1):
                if result.success or result.cancelled:
                    break
                # 새 request_id로 보낸다 — control의 중복요청 캐시(RequestCache)가 같은
                # id를 "재실행 대신 이전 결과 반환"으로 처리하기 때문이다. trace의 스텝에도
                # 같이 반영해야 진행 피드백(_feedback)이 계속 이 스텝을 찾는다.
                step["request_id"] = f"rq-{uuid.uuid4().hex[:8]}"
                goal = dataclasses.replace(goal, request_id=step["request_id"])
                logger.warning("place_into 재시도 %d/%d (trace=%s, 사유=%s)",
                               retry, MAX_PLACE_RETRIES, trace["trace_id"],
                               result.failure_reason)
                # 직전 실패가 타임아웃이었다면 place_server가 이전 movel을 취소했지만
                # 그 취소가 실제로 끝났다는 보장이 없다(dsr_motion.call_action_blocking
                # 참조 — 취소 확인에 최대 25초까지 걸릴 수 있고, 그마저 실패하면 이전
                # 목표가 드라이버 쪽에 여전히 살아있을 수 있다). 그 상태로 곧바로 새
                # movel을 보내면 액션 서버에 두 목표가 겹쳐 그 뒤로 계속 "접수는 되는데
                # 진행이 없는" 상태가 이어지는 걸 실물로 확인했다(2026-09-06). 재시도
                # 전에 짧게 쉬어 이전 취소가 정리될 시간을 준다.
                await asyncio.sleep(3.0)
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
                "cancelled": result.cancelled,
            }

        if step["skill"] == "place_into":
            # place_into는 바구니 접근 높이까지만 물러난다(control/place_server.py
            # _place_real retreat). 그 상태로 다음 스텝을 기다리면 팔이 바구니 위에 남아
            # 카메라 시야를 가리므로, 성공한 place_into 뒤에는 매번 home으로 자동 복귀한다.
            logger.info("place_into 완료 후 home 복귀 (trace=%s)", trace["trace_id"])
            try:
                await executor.home()
            except Exception:
                logger.exception("place_into 후 home 이동 실패 (trace=%s) — 계속 진행한다",
                                 trace["trace_id"])

    return None


def start_command(trace_id: str, command_text: str, executor) -> asyncio.Task:
    """명령 처리를 백그라운드로 띄운다 — HTTP 응답(202)은 즉시 돌려준다.

    태스크를 `_running_tasks`에 등록해야 `cancel_running()`(Stop 처리)이 이걸 찾아
    취소할 수 있다 — 등록하지 않으면 반환값을 호출자가 그냥 버리므로(기존 코드가 그랬다),
    Stop을 눌러도 활성 ROS goal이 없는 구간(재계획 대기, planner 호출 등)에서는 이 태스크가
    계속 돌아 잠시 후 다음 Pick을 그대로 실행해버린다(2026-09-05 실물에서 확인한 사고)."""
    task = asyncio.create_task(run_command(trace_id, command_text, executor))
    _running_tasks[trace_id] = task
    task.add_done_callback(lambda t: _running_tasks.pop(trace_id, None))
    return task
