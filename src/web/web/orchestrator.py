"""명령 1건의 수명주기: 계획 → 검증 결과 확인 → 실행 전 승인 대기 → 순차 실행 → 실패 시 재계획.

**검증을 통과한 시퀀스도 실행 전에 브라우저의 승인을 한 번 받는다**(명령 1건당 1회,
`_await_approval` 참조). SAM+VLM 경로가 어휘 없이 물체 속성(파지력에 직결되는 grip_level
포함)까지 스스로 판단하게 되면서(vlm_detect.py 상단 참조), 검증기를 지났다는 것만으로
그대로 실행하기보다 사람이 한 번 보게 하기로 정책을 바꿨다. `validation_status`가
rejected면 승인 단계까지 가지 않고 그대로 끝난다 — 이 부분은 이전과 같다.
"""
import asyncio
import copy
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
#
# 온디맨드 전환(docs/on-demand-perception.md) 이후에도 이 값 자체는 그대로 둔다 — D-8
# 덕분에 발행되는 stamp는 항상 재촬영 직전 시각이라 age_s는 성공한 관측이라면 여전히
# 거의 0이다. 실제로 늘어난 것은 "새 관측이 오기까지" 기다리는 시간 쪽이고, 그건 아래
# OBSERVE_*_TIMEOUT_S가 담당한다.
MAX_WORLD_STATE_AGE_S = float(os.environ.get("MAX_WORLD_STATE_AGE_S", "5.0"))

# 온디맨드 관측 타임아웃. GPU 실측이 아직 없다(docs/on-demand-perception.md 6절 미실측
# 항목) — CPU 실측(vlm_sam_pipeline.md)은 MODE_FULL(everything+VLM 라벨링+D-8 재촬영)이
# 26~40초, MODE_REPROMPT(재투영 박스로 SAM 1패스)가 0.2초였다. GPU에서 다시 재면 낮춰도
# 된다. 관측 자체(observe 액션)와 grasp가 뒤이어 /world_state를 내는 시간은 서로 다른
# 지연이라 따로 둔다.
OBSERVE_FULL_TIMEOUT_S = float(os.environ.get("OBSERVE_FULL_TIMEOUT_S", "60.0"))
OBSERVE_REPROMPT_TIMEOUT_S = float(os.environ.get("OBSERVE_REPROMPT_TIMEOUT_S", "10.0"))
WORLD_STATE_RELAY_TIMEOUT_S = float(os.environ.get("WORLD_STATE_RELAY_TIMEOUT_S", "3.0"))

# trace_id → 스냅샷. WebSocket 재연결 시 GET /api/traces/{trace_id}로 돌려줄 현재 상태.
traces: dict[str, dict] = {}

# trace_id → 승인 대기 큐. approvals.py가 여기 넣고(POST), 아래 _await_approval이 여기서
# 받는다(GET). _running_tasks와 같은 자리에 두는 이유도 같다 — 진행 중인 명령 하나당
# 하나씩만 있어야 한다.
_pending_approvals: dict[str, asyncio.Queue] = {}

# trace_id → 실행 중인 asyncio.Task. Stop이 "지금 활성 ROS goal"뿐 아니라 이 태스크
# 자체도 취소해야 한다 — 안 그러면 두 액션 스텝 사이(재계획 대기·planner 호출 등,
# 활성 goal이 없는 구간)에 Stop을 눌러도 아무 효과가 없어서, 잠시 후 다음 Pick이
# 그대로 나가는 사고가 있었다(2026-09-05 실물 확인 — 이 세션의 최우선 수정 사항).
_running_tasks: dict[str, asyncio.Task] = {}


@dataclasses.dataclass(frozen=True)
class PickSnapshot:
    observation_id: str
    observation_stamp: tuple[int, int]
    object_id: str
    class_name: str
    selected_candidate_id: str
    selected_grasp_pose: tuple[float, ...]
    executed_tcp_posx: tuple[float, ...]
    footprint_base_mm: tuple[tuple[float, float], ...]
    object_bottom_z_mm: float

    @property
    def tcp_to_object_bottom_mm(self) -> float | None:
        if len(self.executed_tcp_posx) != 6:
            return None
        return self.executed_tcp_posx[2] - self.object_bottom_z_mm


def snapshot(trace_id: str) -> dict | None:
    return traces.get(trace_id)


def has_running_command() -> bool:
    return any(not task.done() for task in _running_tasks.values())


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


async def _wait_for_fresh_observation(executor, trace_id: str, mode: str = "full",
                                      timeout_s: float | None = None,
                                      domain: str = "general") -> bool:
    """관측을 **트리거하고**, 그 결과가 새 /world_state로 들어올 때까지 기다린다.

    온디맨드 전환 전에는 이 함수가 순수 폴링이었다 — perception이 주기 발행 중이니
    가만히 기다리기만 하면 됐다. 이제 perception은 명령을 받은 순간에만 돌므로
    (docs/on-demand-perception.md), 여기서 직접 `executor.observe()`를 불러야 애초에
    새 관측이 생긴다. 안 그러면 첫 명령부터 world_state가 영영 None이다.

    home 이동 직후 곧바로 `get_latest_world_state()`를 부르면 팔이 아직 시야를 가리고
    있거나 이동 전에 찍힌 오래된 스냅샷을 돌려줄 수 있다 — 재계획이 그 상태를 근거로
    LLM에 다시 물으면 방금 건드린 물체가 아직도 "안 보이는" 것으로 나온다.

    반환값(성공 여부)은 호출자가 원하면 보되, 강제하지 않는다 — 실패해도 그다음의
    `get_latest_world_state()`/신선도 검사가 알아서 걸러 사용자에게 알린다(기존 "최선을
    다해서, 실패해도 넘어간다" 원칙을 유지한다).
    """
    if timeout_s is None:
        timeout_s = OBSERVE_FULL_TIMEOUT_S if mode == "full" else OBSERVE_REPROMPT_TIMEOUT_S

    before = executor.get_latest_world_state()
    before_stamp = (before or {}).get("stamp")

    try:
        observed = await asyncio.wait_for(
            executor.observe(trace_id, mode, domain=domain), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("관측 트리거 시간 초과 (%.0fs, trace=%s, mode=%s)",
                       timeout_s, trace_id, mode)
        return False
    if observed is not None and not observed.get("success", True):
        logger.warning("관측 실패 (trace=%s, mode=%s): %s",
                       trace_id, mode, observed.get("failure_reason"))
        return False

    # 관측 액션이 성공했으면 grasp가 그 뒤를 이어 /world_state를 낸다 — perception→grasp
    # 처리 지연분만 더 기다리면 된다(관측 자체의 지연은 이미 위에서 다 기다렸다).
    deadline = asyncio.get_event_loop().time() + WORLD_STATE_RELAY_TIMEOUT_S
    while asyncio.get_event_loop().time() < deadline:
        current = executor.get_latest_world_state()
        stamp = (current or {}).get("stamp")
        expected_observation_id = (observed or {}).get("observation_id")
        observation_matches = (
            expected_observation_id
            and (current or {}).get("observation_id") == expected_observation_id
            and (current or {}).get("trace_id") == trace_id
        )
        if observation_matches or (not expected_observation_id and stamp and stamp != before_stamp):
            return True
        await asyncio.sleep(0.1)
    logger.warning("observe는 성공했지만 /world_state가 갱신되지 않았다 (trace=%s) — "
                   "grasp 상태를 확인한다", trace_id)
    return False


# "파지 후보가 없다"는 거부는 **그 순간의 관측 하나**로 정해진다. GraspNet은 프레임마다
# 자세가 조금씩 달라서 접근각 필터(grasp_params.yaml의 approach_angle_*)를 넘겼다 못
# 넘겼다 하고, 그래서 같은 물체가 한 프레임에서는 후보가 있고 다음 프레임에서는 없다.
# 한 번 보고 포기하면 사람 입장에서는 "될 때도 있고 안 될 때도 있는" 것으로 보인다.
# 새 관측을 몇 번 더 기다렸다 다시 물어본다 (관측 주기가 ~0.5초라 아래 값이면 약 1초).
_GRASP_RETRY_ATTEMPTS = 2
_GRASP_RETRY_REASON = "파지 후보가 없습니다"


async def _plan_with_grasp_retry(trace_id: str, command_text: str, world_state: dict,
                                  previous_failure, executor, domain: str = "general") -> tuple[dict, dict]:
    """planner에 계획을 묻되, 파지 후보가 없어서 거부되면 새 관측으로 몇 번 더 시도한다.

    다른 거부 사유(파지 불가 상태, 작업반경 초과, 물체가 목록에 없음 등)는 관측을 더
    받는다고 달라지지 않으므로 그대로 돌려준다 — 무의미한 LLM 호출을 반복하지 않는다.
    """
    result = await planner_client.plan(trace_id, command_text, world_state,
                                       previous_failure, domain=domain)
    for attempt in range(1, _GRASP_RETRY_ATTEMPTS + 1):
        if result.get("validation_status") == "approved":
            return result, world_state
        if _GRASP_RETRY_REASON not in (result.get("validation_reason") or ""):
            return result, world_state
        logger.info("파지 후보 없음으로 거부 — 새 관측 대기 후 재시도 %d/%d (trace=%s): %s",
                    attempt, _GRASP_RETRY_ATTEMPTS, trace_id, result.get("validation_reason"))
        # VLM은 다시 안 부른다 — 물체가 무엇인지는 이미 알고, GraspNet이 프레임마다
        # 자세가 조금씩 달라 후보를 놓쳤을 뿐이다(위 주석). reprompt로 depth만 새로 잰다.
        await _wait_for_fresh_observation(executor, trace_id, mode="reprompt", domain=domain)
        latest = executor.get_latest_world_state()
        if latest is None:
            return result, world_state
        world_state = latest
        result = await planner_client.plan(trace_id, command_text, latest,
                                           previous_failure, domain=domain)
    return result, world_state


async def run_command(trace_id: str, command_text: str, executor, domain: str = "general") -> None:
    """명령 하나를 끝까지 처리한다. 백그라운드 태스크로 실행된다.

    `cancel_running()`이 이 태스크를 취소하면 `CancelledError`가 현재 멈춰있던 await
    지점(아래 for 루프 어디든)에서 던져진다 — 여기서 잡아 취소 결과를 broadcast하고
    다시 raise한다. 다시 raise하지 않으면 asyncio 입장에서 이 태스크는 "정상 종료"가
    되어 `task.cancelled()`가 False를 반환하고, 취소 여부를 확인하는 다른 코드(테스트,
    로깅)가 오판할 수 있다."""
    try:
        await _run_command_body(trace_id, command_text, executor, domain)
    except asyncio.CancelledError:
        logger.info("명령 취소됨 (trace=%s)", trace_id)
        await hub.broadcast({
            "type": "execution_result", "trace_id": trace_id, "request_id": "",
            "success": False, "failure_reason": "cancelled", "cancelled": True,
        })
        raise


async def _run_command_body(trace_id: str, command_text: str, executor, domain: str = "general") -> None:
    trace = _new_trace(trace_id, command_text)
    previous_failure = None
    previous_targets = None
    # 처음 계획(시도 0)의 pick 대상 클래스. 재계획이 이 범위를 벗어나면 실행하지 않는다.
    allowed_classes = None

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

        # 온디맨드 전환: attempt==0(최초 시도)도 이 트리거가 없으면 world_state가 영영
        # None이다 — perception이 더는 알아서 발행하지 않는다(docs/on-demand-perception.md).
        # 매 시도마다 전체 스캔(full)을 다시 한다 — 재계획은 장면이 통째로 달라졌을 수
        # 있다고 보는 경로이므로 라벨을 재사용하는 reprompt로는 부족하다.
        await _wait_for_fresh_observation(executor, trace_id, mode="full", domain=domain)

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
            await _wait_for_fresh_observation(executor, trace_id, mode="full", domain=domain)
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
            result, world_state = await _plan_with_grasp_retry(
                trace_id, command_text, world_state, previous_failure, executor,
                domain=domain,
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

        trace["steps"] = _build_steps(result.get("steps", []), world_state)

        trace["objects"] = world_state.get("objects", [])
        decision = await _await_approval(trace, world_state, command_text,
                                         previous_failure, executor, domain)
        if decision is None:
            if trace["validation_status"] == "rejected":
                # correct_label로 재계획했는데 이번엔 검증을 못 지났다 — 사용자 거부가
                # 아니라 검증 거부다(위 306행과 같은 사유 표시).
                logger.info("재계획이 거부됨 (trace=%s): %s", trace_id, trace["validation_reason"])
                await hub.broadcast({
                    "type": "execution_result", "trace_id": trace_id, "request_id": "",
                    "success": False, "failure_reason": "rejected",
                    "validation_reason": trace["validation_reason"],
                })
            else:
                logger.info("사용자가 실행을 거부함 (trace=%s)", trace_id)
                await hub.broadcast({
                    "type": "execution_result", "trace_id": trace_id, "request_id": "",
                    "success": False, "failure_reason": "user_rejected",
                })
            return
        world_state, trace["steps"] = decision

        # **재계획마다 대상 물체가 바뀔 수 있다.** planner는 매 시도 새 world_state로
        # 명령을 다시 그라운딩하는데, object_id는 관측마다 새로 매겨지므로 같은 물체라도
        # id가 달라진다. 그래서 대상 비교는 id가 아니라 **클래스명**으로 한다.
        targets = [(s["skill"], s["object_id"], class_of(world_state, s["object_id"]))
                   for s in trace["steps"]]
        logger.info("실행 대상 (trace=%s, 시도 %d): %s", trace_id, attempt,
                    ", ".join(f"{skill}:{oid}({name or '?'})" for skill, oid, name in targets))
        if previous_targets is not None and previous_targets != targets:
            logger.warning(
                "재계획으로 대상이 바뀌었다 (trace=%s): %s → %s",
                trace_id,
                ", ".join(f"{o}({n or '?'})" for _, o, n in previous_targets),
                ", ".join(f"{o}({n or '?'})" for _, o, n in targets))
        previous_targets = targets

        # **처음 계획에 없던 물체를 재계획이 새로 데려오면 실행하지 않는다.**
        # 2026-09-08 실물: "물티슈 왼쪽으로 옮겨줘"가 파지 실패한 뒤, 재계획을 받은 LLM이
        #   refusal='물티슈는 파지 실패로 인해 건너뛰고, 접이 우산만 옮깁니다.'
        # 라며 **사용자가 말한 적 없는 우산**을 집으러 갔다. previous_failure를 알려주면
        # LLM이 "실패한 건 건너뛰고 대신 다른 걸 하자"로 읽는 것인데, 그건 명령의 범위를
        # 벗어나는 판단이다 — 사람이 물티슈를 옮기라고 했으면 못 옮기는 것이지 다른 물체를
        # 대신 옮겨도 되는 게 아니다.
        #
        # **id가 아니라 클래스로 본다**(id는 관측마다 바뀐다). 실패한 물체를 건너뛰고 원래
        # 계획의 나머지를 계속하는 것은 허용된다 — 부분집합이면 통과하기 때문이다.
        # 클래스를 모르는(None) 대상은 새로 들어온 것으로 본다: 판별할 수 없으면 멈추는
        # 쪽이 엉뚱한 물체를 집는 것보다 낫다.
        step_classes = {name for skill, _, name in targets if skill == "pick"}
        if allowed_classes is None:
            allowed_classes = step_classes
        else:
            intruders = {name for name in step_classes if name not in allowed_classes}
            if intruders:
                logger.error(
                    "재계획이 명령에 없던 물체를 대상으로 삼았다 — 실행하지 않고 멈춘다 "
                    "(trace=%s, 처음 대상=%s, 새 대상=%s)", trace_id,
                    sorted(n or "?" for n in allowed_classes),
                    sorted(n or "?" for n in intruders))
                await hub.broadcast({
                    "type": "execution_result", "trace_id": trace_id, "request_id": "",
                    "success": False, "failure_reason": "replan_changed_target",
                    "validation_reason": (
                        "재계획이 명령과 다른 물체("
                        + ", ".join(sorted(n or "?" for n in intruders))
                        + ")를 대상으로 삼아 중단했습니다."),
                })
                return

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


def _object_bottom_offset_mm(world_state: dict, steps: list, index: int) -> float | None:
    """place할 물체가 TCP보다 얼마나 아래로 내려와 있는지(mm). 못 구하면 None.

    **왜 필요한가.** bins.yaml의 bin_pose는 **빈 그리퍼로** 티칭한 높이다. 물체를 든 채
    그 높이까지 내려가면 물체가 바구니 바닥에 먼저 닿고, 로봇은 목표 z에 도달하지 못한
    채 계속 밀어붙이다 안전모드로 들어간다(2026-09-06 실물: 목표 213.4mm인데 217.9mm에서
    멈춘 뒤 알람 → heartbeat 유실 → 권한 회수). 예전에는 이걸 고정 여유
    (skill_params.yaml의 release_clearance_mm)로 짐작했는데, 물체마다 높이가 달라
    한 값으로는 맞출 수 없다 — 큰 물체엔 모자라고 작은 물체엔 과하다.

    **계산.** perception이 물체 높이를 depth로 실측해 발행한다(DetectedObject.height_mm =
    윗면 z - 지지면 z). 지지면 z = position_z - height_mm 이므로,
        offset = 파지 z - 지지면 z = grasp_pose.z - position_z + height_mm
    파지 z는 같은 물체의 **직전 pick 스텝**이 쓴 grasp_pose에서 가져온다 — planner가
    place 스텝에는 파지 자세를 싣지 않으므로(schema.PlanStep) 여기서 되짚는다.

    height_mm이 0(미상)이거나 pick 스텝을 못 찾으면 None을 돌려주고, control이 예전처럼
    고정 여유만 쓰게 둔다 — 잘못된 값으로 더 깊이 내려가는 것보다 낫다.
    """
    step = steps[index]
    object_id = step.get("object_id")
    grasp_pose = None
    for previous in reversed(steps[:index]):
        if previous.get("skill") == "pick" and previous.get("object_id") == object_id:
            grasp_pose = previous.get("grasp_pose")
            break
    if not grasp_pose:
        return None
    for obj in world_state.get("objects", []):
        if obj.get("object_id") != object_id:
            continue
        height_mm = float(obj.get("height_mm") or 0.0)
        position = obj.get("position_base_mm") or {}
        if height_mm <= 0.0 or "z" not in position:
            return None
        grasp_z = ((grasp_pose.get("position") or {}).get("z")
                   if isinstance(grasp_pose, dict) else None)
        if grasp_z is None:
            return None
        offset = float(grasp_z) - float(position["z"]) + height_mm
        # 음수는 물리적으로 말이 안 된다(물체 바닥이 TCP보다 위에 있다는 뜻).
        # 측정이 튄 경우이므로 쓰지 않는다.
        return offset if offset > 0.0 else None
    return None


def _build_steps(steps: list[dict], world_state: dict) -> list[dict]:
    """planner의 steps 응답 → trace가 들고 있을 스텝 목록. 최초 계획과 라벨 수정 후
    재계획이 같은 모양을 만들어야 하므로 함수로 뽑아 둔다.

    **world_state를 함께 받는 이유**: class_name·footprint·observation_id는 planner의
    step 응답에 없고 그 계획의 입력이 된 관측에만 있다. PickSnapshot이 이 넷을 전부
    요구하므로(_freeze_pick_snapshot) 여기서 안 실으면 place가 frozen snapshot 없이
    시작해 geometry 경로가 통째로 죽는다."""
    return [
        {
            "request_id": f"rq-{uuid.uuid4().hex[:8]}",
            "skill": s["skill"],
            "object_id": s["object_id"],
            "bin_id": s.get("bin_id"),
            "grip_level": s["grip_level"],
            "grasp_pose": s.get("grasp_pose"),
            "gripper_width_mm": s.get("gripper_width_mm"),
            # 후보 목록과 물체 정보는 화면에 내보내지 않고 control에 전달만 한다 —
            # control이 실행 가능한 후보를 고르는 데 쓴다(control/grasp_selection.py).
            "grasp_candidates": s.get("grasp_candidates") or [],
            "object_center_mm": s.get("object_center_mm"),
            "object_height_mm": s.get("object_height_mm"),
            "depth_valid_ratio": s.get("depth_valid_ratio"),
            "class_name": class_of(world_state, s["object_id"]),
            "object_footprint_base_mm": copy.deepcopy(
                (_object_of(world_state, s["object_id"]) or {}).get("footprint_base_mm") or []),
            "source_observation_id": world_state.get("observation_id", ""),
            "source_observation_stamp": copy.deepcopy(world_state.get("stamp")),
            "status": "pending",
            "phase": None,
        }
        for s in steps
    ]


async def _broadcast_approval_needed(trace: dict) -> None:
    await hub.broadcast({
        "type": "execution_approval_needed",
        "trace_id": trace["trace_id"],
        "sequence_id": trace["sequence_id"],
        "validation_status": trace["validation_status"],
        "validation_reason": trace["validation_reason"],
        "steps": trace["steps"],
        "objects": trace["objects"],
    })


def _apply_label_correction(world_state: dict, message: dict) -> None:
    """라벨 수정 요청을 world_state에 그대로 반영한다. 이 world_state가 재계획의 입력이 된다."""
    object_id = message.get("object_id")
    for obj in world_state.get("objects", []):
        if obj.get("object_id") != object_id:
            continue
        if message.get("class_name"):
            obj["class_name"] = message["class_name"]
        if message.get("name_ko"):
            obj["name_ko"] = message["name_ko"]
        return
    logger.warning("라벨 수정 대상 object_id를 world_state에서 못 찾음: %s", object_id)


async def _await_approval(trace: dict, world_state: dict, command_text: str,
                          previous_failure, executor, domain: str = "general") -> tuple[dict, list[dict]] | None:
    """계획된 시퀀스를 실행하기 전에 브라우저의 승인을 기다린다(명령 1건당 1회 원칙).

    라벨 수정(`correct_label`)이 오면 world_state를 고쳐 재계획하고, 그 결과를 다시
    승인 화면으로 내보낸다 — 승인은 한 번에 끝나지 않고 사용자가 만족할 때까지 반복될
    수 있다. 큐를 쓰는 이유는 `run_command`의 `CancelledError` 처리(위 참조)가
    `await queue.get()` 대기 중에도 그대로 적용되어, Stop이 승인 대기 구간도 끊을 수
    있게 하기 위해서다 — 별도 취소 처리를 만들 필요가 없다.
    """
    trace_id = trace["trace_id"]
    queue: asyncio.Queue = asyncio.Queue()
    _pending_approvals[trace_id] = queue
    try:
        while True:
            await _broadcast_approval_needed(trace)
            message = await queue.get()
            action = message.get("action")

            if action == "approve":
                return world_state, trace["steps"]
            if action == "reject":
                return None
            if action != "correct_label":
                logger.warning("알 수 없는 승인 액션 %r — 무시 (trace=%s)", action, trace_id)
                continue

            _apply_label_correction(world_state, message)
            result = await _plan_with_grasp_retry(
                trace_id, command_text, world_state, previous_failure, executor,
                domain=domain)
            trace["sequence_id"] = result.get("sequence_id")
            trace["validation_status"] = result.get("validation_status")
            trace["validation_reason"] = result.get("validation_reason")
            if result.get("validation_status") != "approved":
                # 승인 화면에 계속 보여줄 스텝이 없다 — 호출자가 validation_status로
                # "검증 거부"와 "사용자 거부"를 가른다(위 _run_command_body 참조).
                trace["steps"] = []
                return None
            trace["steps"] = _build_steps(result.get("steps", []), world_state)
            trace["objects"] = world_state.get("objects", [])
    finally:
        _pending_approvals.pop(trace_id, None)


def has_pending_approval() -> bool:
    """commands.py가 새 명령을 막을지 판단하는 데 쓴다. 승인 대기는 로봇의 실제 상태(ROS가
    보고하는 mode)와 무관하다 — 로봇은 가만히 있으므로 robot_state().mode를 흉내 내는 대신
    이 큐의 존재 여부로 직접 판단한다."""
    return bool(_pending_approvals)


def resolve_approval(trace_id: str, message: dict) -> bool:
    """approvals 라우터가 부른다. 대기 중인 큐가 없으면(이미 끝났거나 잘못된 trace_id) False."""
    queue = _pending_approvals.get(trace_id)
    if queue is None:
        return False
    queue.put_nowait(message)
    return True


def _record_selected_candidate(step: dict, result) -> None:
    """control이 **실제로 실행한** 후보를 스텝에 반영한다.

    planner가 채워 보낸 `grasp_pose`는 1순위 후보인데, control은 개폭·IK·관절·안전을 보고
    다른 후보를 고를 수 있다(control/grasp_selection.py). 그 경우 스텝에 1순위가 그대로
    남아 있으면 두 군데가 틀어진다:

    - **place가 놓는 높이** — `_object_bottom_offset_mm`이 직전 pick 스텝의 `grasp_pose`로
      "물체가 TCP보다 얼마나 내려와 있는지"를 계산한다. 실제로 문 자세가 아니면 그 값이
      틀리고, 물체를 바구니 바닥에 찍거나 너무 높은 데서 놓는다.
    - **실행 로그(DB)** — 나중에 파지 오차를 분석할 때 실행하지 않은 자세를 보게 된다.

    선택 결과를 못 받았거나(구 control) 후보 목록에서 못 찾으면 그대로 둔다.
    """
    selected_id = getattr(result, "selected_candidate_id", "") or ""
    if not selected_id:
        return
    for candidate in step.get("grasp_candidates") or []:
        if candidate.get("candidate_id") == selected_id:
            step["grasp_pose"] = candidate.get("pose")
            step["gripper_width_mm"] = candidate.get("gripper_width_mm")
            step["selected_candidate_id"] = selected_id
            return
    logger.warning("control이 고른 후보 %s를 계획 후보 목록에서 못 찾았다 — "
                   "grasp_pose를 1순위인 채로 둔다 (object=%s)",
                    selected_id, step.get("object_id"))


def _object_of(world_state: dict, object_id: str) -> dict | None:
    for obj in (world_state or {}).get("objects") or []:
        if obj.get("object_id") == object_id:
            return obj
    return None


def _pose_tuple(pose: dict | None) -> tuple[float, ...]:
    pose = pose or {}
    position = pose.get("position") or {}
    orientation = pose.get("orientation") or {}
    return tuple(float(v) for v in (
        position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0),
        orientation.get("x", 0.0), orientation.get("y", 0.0),
        orientation.get("z", 0.0), orientation.get("w", 1.0),
    ))


def _freeze_pick_snapshot(step: dict, result) -> PickSnapshot | None:
    executed = tuple(float(v) for v in (getattr(result, "executed_tcp_posx", None) or []))
    center = step.get("object_center_mm") or {}
    height = step.get("object_height_mm")
    if len(executed) != 6 or center.get("z") is None or height is None:
        return None
    footprint = tuple(
        (float(point["x"]), float(point["y"]))
        for point in (step.get("object_footprint_base_mm") or [])
    )
    stamp = step.get("source_observation_stamp") or {}
    observation_id = str(step.get("source_observation_id") or "")
    if not observation_id or "sec" not in stamp or "nanosec" not in stamp:
        return None
    return PickSnapshot(
        observation_id=observation_id,
        observation_stamp=(int(stamp.get("sec", 0)), int(stamp.get("nanosec", 0))),
        object_id=str(step["object_id"]),
        class_name=str(step.get("class_name") or ""),
        selected_candidate_id=str(getattr(result, "selected_candidate_id", "") or ""),
        selected_grasp_pose=_pose_tuple(step.get("grasp_pose")),
        executed_tcp_posx=executed,
        footprint_base_mm=footprint,
        object_bottom_z_mm=float(center["z"]) - float(height),
    )


def class_of(world_state: dict, object_id: str) -> str | None:
    """world_state에서 그 물체의 클래스명. 대상 추적 로그가 id만으로는 안 읽혀서 붙인다."""
    for obj in (world_state or {}).get("objects") or []:
        if obj.get("object_id") == object_id:
            return obj.get("class_name") or obj.get("class_ko")
    return None


async def _execute_steps(trace: dict, world_state: dict, executor) -> dict | None:
    """스텝을 순서대로 실행한다. 실패하면 previous_failure 형태로 반환."""
    class_map = store.object_class_map(world_state)
    held_snapshot: PickSnapshot | None = None

    for index, step in enumerate(trace["steps"]):
        bottom_offset = (_object_bottom_offset_mm(world_state, trace["steps"], index)
                         if step["skill"] == "place_into" else None)
        if step["skill"] == "place_into":
            logger.info("place_into 놓는 높이 계산 (trace=%s, object=%s): 물체 바닥 오프셋 %s",
                        trace["trace_id"], step["object_id"],
                        f"{bottom_offset:.1f}mm" if bottom_offset is not None
                        else "미상 — control의 고정 여유 사용")
        goal = SkillGoal(
            trace_id=trace["trace_id"],
            request_id=step["request_id"],
            object_id=step["object_id"],
            grip_level=step["grip_level"],
            source_observation_id=(held_snapshot.observation_id
                                   if step["skill"] == "place_into" and held_snapshot
                                   else step.get("source_observation_id", "")),
            source_observation_stamp=(
                {"sec": held_snapshot.observation_stamp[0],
                 "nanosec": held_snapshot.observation_stamp[1]}
                if step["skill"] == "place_into" and held_snapshot
                else step.get("source_observation_stamp")),
            class_name=step.get("class_name", ""),
            grasp_pose=step.get("grasp_pose"),
            gripper_width_mm=step.get("gripper_width_mm"),
            grasp_candidates=step.get("grasp_candidates") or [],
            object_center_mm=step.get("object_center_mm"),
            object_height_mm=step.get("object_height_mm"),
            depth_valid_ratio=step.get("depth_valid_ratio"),
            object_footprint_base_mm=(
                [{"x": x, "y": y, "z": 0.0} for x, y in held_snapshot.footprint_base_mm]
                if step["skill"] == "place_into" and held_snapshot
                else step.get("object_footprint_base_mm") or []),
            bin_id=step.get("bin_id"),
            object_bottom_offset_mm=bottom_offset,
            pickup_tcp_posx=(list(held_snapshot.executed_tcp_posx)
                              if step["skill"] == "place_into" and held_snapshot else None),
            tcp_to_object_bottom_mm=(held_snapshot.tcp_to_object_bottom_mm
                                     if step["skill"] == "place_into" and held_snapshot else None),
        )

        async def on_feedback(request_id, phase, _trace=trace):
            await _feedback(request_id, phase, _trace)

        if step["skill"] == "pick":
            result = await executor.call_pick(goal, on_feedback)
            _record_selected_candidate(step, result)
            if result.success:
                if (getattr(result, "source_observation_id", "")
                        != step.get("source_observation_id", "")):
                    logger.error("Pick result observation 불일치 (goal=%s, result=%s)",
                                 step.get("source_observation_id"),
                                 getattr(result, "source_observation_id", ""))
                    result.success = False
                    step["snapshot_error"] = "observation_mismatch"
                    # 물리 pick은 성공했으므로 물체를 들고 있을 수 있다. outer loop의
                    # post-lift uncertainty 경로로 보내 자동 replan/다음 pick을 막는다.
                    result.failure_reason = "unreachable"
                else:
                    held_snapshot = _freeze_pick_snapshot(step, result)
                    if held_snapshot is None:
                        logger.error("Pick 성공 결과에 실제 TCP/object geometry가 없어 중단한다")
                        result.success = False
                        step["snapshot_error"] = "missing_pick_snapshot"
                        result.failure_reason = "unreachable"
                    else:
                        step["pick_snapshot"] = dataclasses.asdict(held_snapshot)
                        logger.info(
                            "[PickSnapshot] obs=%s object=%s candidate=%s tcp=%s bottom=%.1f",
                            held_snapshot.observation_id, held_snapshot.object_id,
                            held_snapshot.selected_candidate_id,
                            [round(v, 2) for v in held_snapshot.executed_tcp_posx],
                            held_snapshot.object_bottom_z_mm)
        else:
            if held_snapshot is None or held_snapshot.object_id != step["object_id"]:
                logger.error("place_into와 대응하는 frozen PickSnapshot이 없다 (object=%s)",
                             step["object_id"])
                return {"sequence_id": trace["sequence_id"],
                        "request_id": step["request_id"], "object_id": step["object_id"],
                        "failure_reason": "missing_pick_snapshot", "skill": "place_into",
                        "cancelled": False}
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
            grip_level_used=step["grip_level"],
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

            # 온디맨드 관측(D-1): VLM은 다시 안 부른다 — 방금 옮긴 물체를 지웠다는 것과
            # 아직 안 옮긴 물체들이 잘 있다는 것을 재투영 박스로만 확인한다.
            #
            # **의도적으로 이 결과로 위의 `world_state`/`class_map`을 바꾸지 않는다.**
            # 처음엔 여기서 world_state를 갈아끼우면 "뒤 스텝이 옛 좌표로 돈다"는 문제가
            # 풀릴 줄 알았는데(계획 초안), 실제로는 그렇지 않다 — 이 함수의 각 스텝이 쓰는
            # grasp_pose는 이미 planner 응답에 다 정해져서 trace["steps"]에 박혀 있고
            # (`_object_bottom_offset_mm`도 world_state가 아니라 trace["steps"]의 직전
            # pick 기록에서 파지 z를 가져온다), 이 루프 안에서 world_state를 다시 읽어도
            # 그 좌표가 바뀌지 않는다. 오히려 방금 집은 물체는 이제 테이블 위에 없으므로
            # world_state를 갈아끼우면 `_object_bottom_offset_mm`이 그 물체의 height_mm/
            # position을 못 찾아 지금보다 더 자주 None(고정 여유로 후퇴)이 된다 — 있던
            # 정보를 잃는 방향이라 갈아끼우지 않는다. 여기서 트리거하는 이유는 순전히
            # 화면의 관측 이미지(D-5)와 executor.get_latest_world_state()를 최신으로
            # 유지하기 위해서다. 다음 명령이 쓸 진짜 새 전체 스캔은 `_run_command_body`
            # 진입 시(위 246행)의 mode="full" 트리거가 담당한다.
            await _wait_for_fresh_observation(
                executor, trace["trace_id"], mode="reprompt",
                timeout_s=OBSERVE_REPROMPT_TIMEOUT_S)

    return None


def start_command(trace_id: str, command_text: str, executor, domain: str = "general") -> asyncio.Task:
    """명령 처리를 백그라운드로 띄운다 — HTTP 응답(202)은 즉시 돌려준다.

    태스크를 `_running_tasks`에 등록해야 `cancel_running()`(Stop 처리)이 이걸 찾아
    취소할 수 있다 — 등록하지 않으면 반환값을 호출자가 그냥 버리므로(기존 코드가 그랬다),
    Stop을 눌러도 활성 ROS goal이 없는 구간(재계획 대기, planner 호출 등)에서는 이 태스크가
    계속 돌아 잠시 후 다음 Pick을 그대로 실행해버린다(2026-09-05 실물에서 확인한 사고)."""
    task = asyncio.create_task(run_command(trace_id, command_text, executor, domain))
    _running_tasks[trace_id] = task
    task.add_done_callback(lambda t: _running_tasks.pop(trace_id, None))
    return task
