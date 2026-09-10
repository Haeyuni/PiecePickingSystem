"""검증기: object_id·bin_id 그라운딩 + SafetyEvent 게이트 체크 (FR-11, FR-12).

검증 항목: 스키마 유효성, 작업공간 경계, 가반하중, 스킬 전제조건, 활성 SafetyEvent 유무.

**파지 후보는 여기서 하나로 좁히지 않는다.** 작업반경 안에 있는 것을 점수 순으로 전부
넘기고, 실행할 하나는 control이 고른다(control/grasp_selection.py) — 개폭 유효성·IK·
관절 한계는 로봇에 붙어 있어야만 답할 수 있고 planner는 ROS2를 모르기 때문이다.

검증을 통과한 시퀀스는 사람의 승인 없이 자동 실행되므로(웹_인터페이스_정의서 1절),
여기서 막지 못한 것은 로봇이 그대로 수행한다. 판정은 전부 코드로 하고 LLM 출력을 신뢰하지 않는다.
"""
import math

from . import grounding
from .schema import GraspCandidateOut, PlanStep, Pose

# M0609 사양 (BR 6.1절). 작업반경 900mm, 가반하중 6kg.
WORKSPACE_RADIUS_MM = 900.0
# 그리퍼(SG2/RG2) 자중을 뺀 실효 가반하중. 실측 전까지 보수적으로 잡는다.
MAX_PAYLOAD_G = 5000.0

# 파지력 단계 값을 못 읽었을 때의 기본 (FR-05, NFR-03a) — resolve_grip_level 참조.
# 5 = 가장 약하게(20N). 여기서 아는 것이 없을 때는 조심스러운 쪽이어야 한다.
FALLBACK_GRIP_LEVEL = 5


GRIP_LEVELS = (1, 2, 3, 4, 5)


class Rejected(Exception):
    """검증 거부. message가 그대로 validation_reason이 된다."""


def _reach_mm(position: dict) -> float:
    return math.sqrt(position["x"] ** 2 + position["y"] ** 2 + position["z"] ** 2)


def check_safety_gate(active_safety_events: list[dict]) -> None:
    """활성 안전 이벤트가 있으면 어떤 시퀀스도 통과시키지 않는다 (FR-12).

    severity=critical만 게이트로 쓴다. warning까지 막으면 사소한 경고로 시스템이 멈춘다.
    """
    for event in active_safety_events:
        if event.get("severity") == "critical":
            raise Rejected(
                f"활성 안전 이벤트로 실행이 차단되었습니다: "
                f"{event.get('event_type')} ({event.get('description') or '설명 없음'})"
            )


def resolve_grip_level(obj: dict) -> int:
    """물체에 적용할 파지력 단계(1~5)를 결정한다.

    **예전에는 미확인(needs_confirmation) 물체를 무조건 fallback(fragile)로 강제했다.**
    등록 클래스 표(objects.yaml)로 아는 물체만 제 프로파일을 쓰고 나머지는 전부 조심스럽게
    다루자는 규칙이었다(NFR-03a, FR-05b).

    SAM+VLM 경로가 어휘 없이 돌기 시작하면서 그 규칙이 성립하지 않는다 — 그 경로의 물체는
    **전부** 미확인이므로 강제가 걸리면 치약이든 우산이든 5단계(20N·최저속도)가 되어
    grip_level이라는 구분 자체가 사라진다. 그래서 지금은 **값이 유효하면 그대로 쓴다**.
    그 값이 사진을 본 모델의 판단이라는 뜻이고, 파지력을 조심스러운 쪽으로 미는 책임은
    인지 단계로 옮겼다 (`vlm_detect.SYSTEM_PROMPT_MARKS`의 [grip_level],
    `_normalize_marks`의 5 강제).

    값이 없거나 모르는 값이면 여전히 5(가장 약하게)다 — 여기서 아는 것이 없을 때의 기본은
    조심스러운 쪽이어야 한다.
    """
    level = obj.get("grip_level")
    try:
        level = int(level)
    except (TypeError, ValueError):
        level = 0
    if level not in GRIP_LEVELS:
        return FALLBACK_GRIP_LEVEL
    return level


def _reachable_candidates(obj: dict) -> list[dict]:
    """작업반경 안에 있는 파지 후보를 점수 내림차순으로 돌려준다. 하나도 없으면 Rejected.

    **후보 하나가 아니라 목록을 넘긴다.** 예전에는 점수 최고 후보 하나만 골라 보냈고,
    그 하나가 실행 불가능하면(개폭 미상, IK 안 풀림) pick 전체가 실패했다 — 나머지
    후보는 멀쩡한데도 쓰이지 못했다. 실행 가능성 판정은 로봇에 붙은 control이 하므로
    (control/grasp_selection.py) 여기서는 **로봇 없이도 확실히 아는 것**만 거른다:
    팔이 물리적으로 닿지 않는 거리에 있는 후보.

    그래서 작업반경 검사도 "1순위가 벗어나면 거부"가 아니라 "전부 벗어나면 거부"다.
    """
    candidates = obj.get("grasp_candidates") or []
    if not candidates:
        raise Rejected(
            f"'{obj['object_id']}'({obj.get('name_ko') or obj.get('class_name')})에 "
            f"유효한 파지 후보가 없습니다"
        )
    ordered = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
    in_reach = [c for c in ordered
                if _reach_mm(c["pose"]["position"]) <= WORKSPACE_RADIUS_MM]
    if not in_reach:
        raise Rejected(
            f"'{obj['object_id']}'의 파지 후보 {len(ordered)}개가 모두 "
            f"작업반경 {WORKSPACE_RADIUS_MM:g}mm를 벗어납니다"
        )
    return in_reach


def _candidate_out(candidate: dict) -> GraspCandidateOut:
    return GraspCandidateOut(
        pose=Pose(**candidate["pose"]),
        score=float(candidate.get("score") or 0.0),
        gripper_width_mm=float(candidate.get("gripper_width_mm") or 0.0),
        candidate_id=candidate.get("candidate_id") or "",
        grasp_depth_mm=float(candidate.get("grasp_depth_mm") or 0.0),
        strategy=candidate.get("strategy") or "",
        point_cloud_path=candidate.get("point_cloud_path") or "",
    )


def validate(llm_steps: list, world_state: dict, bins: dict,
             active_safety_events: list[dict] | None = None) -> list[PlanStep]:
    """LLM 스텝을 검증하고 실행 가능한 PlanStep으로 변환한다.

    거부 사유가 있으면 Rejected를 던진다.
    """
    check_safety_gate(active_safety_events or [])

    if not llm_steps:
        raise Rejected("실행할 스킬이 없습니다")

    reason = grounding.check_grounding(llm_steps, world_state, bins)
    if reason:
        raise Rejected(reason)

    objects = grounding.world_objects(world_state)
    # pick으로 집어 든 물체만 place_into할 수 있다 (스킬 전제조건)
    held: str | None = None
    plan: list[PlanStep] = []

    for i, step in enumerate(llm_steps):
        where = f"{i + 1}번째 스텝({step.skill})"
        obj = objects[step.object_id]
        grip_level = resolve_grip_level(obj)

        if step.skill == "pick":
            if held is not None:
                raise Rejected(f"{where}: '{held}'를 든 상태에서 다시 pick할 수 없습니다")
            if not obj.get("graspable", True):
                raise Rejected(
                    f"{where}: '{step.object_id}'는 파지 불가 상태입니다"
                    f"({obj.get('not_graspable_reason') or '사유 없음'})"
                )
            mass = obj.get("mass_g") or 0.0
            if mass > MAX_PAYLOAD_G:
                raise Rejected(
                    f"{where}: '{step.object_id}'의 무게 {mass:g}g가 "
                    f"가반하중 {MAX_PAYLOAD_G:g}g를 초과합니다"
                )
            candidates = _reachable_candidates(obj)
            best = candidates[0]
            center = obj.get("position_base_mm") or None
            held = step.object_id
            plan.append(PlanStep(
                skill="pick",
                object_id=step.object_id,
                grip_level=grip_level,
                grasp_pose=Pose(**best["pose"]),
                gripper_width_mm=best.get("gripper_width_mm") or None,
                grasp_candidates=[_candidate_out(c) for c in candidates],
                object_center_mm=dict(center) if center else None,
                object_height_mm=obj.get("height_mm") or None,
                depth_valid_ratio=obj.get("depth_valid_ratio") or None,
            ))

        else:  # place_into
            if held != step.object_id:
                holding = f"'{held}'를 들고 있습니다" if held else "든 물체가 없습니다"
                raise Rejected(
                    f"{where}: '{step.object_id}'를 집지 않은 채 배치할 수 없습니다 — {holding}"
                )
            bin_pose = (bins[step.bin_id] or {}).get("pose") or {}
            if bin_pose and _reach_mm(bin_pose) > WORKSPACE_RADIUS_MM:
                raise Rejected(
                    f"{where}: 목적지 '{step.bin_id}'가 작업반경을 벗어납니다"
                )
            held = None
            plan.append(PlanStep(
                skill="place_into",
                object_id=step.object_id,
                grip_level=grip_level,
                bin_id=step.bin_id,
            ))

    if held is not None:
        raise Rejected(f"'{held}'를 집은 뒤 배치하는 스텝이 없습니다")

    return plan
