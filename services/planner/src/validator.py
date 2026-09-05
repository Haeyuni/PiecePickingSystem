"""검증기: object_id·bin_id 그라운딩 + SafetyEvent 게이트 체크 (FR-11, FR-12).

검증 항목: 스키마 유효성, 작업공간 경계, 가반하중, 스킬 전제조건, 활성 SafetyEvent 유무.

검증을 통과한 시퀀스는 사람의 승인 없이 자동 실행되므로(웹_인터페이스_정의서 1절),
여기서 막지 못한 것은 로봇이 그대로 수행한다. 판정은 전부 코드로 하고 LLM 출력을 신뢰하지 않는다.
"""
import math

from . import grounding
from .schema import PlanStep, Pose

# M0609 사양 (BR 6.1절). 작업반경 900mm, 가반하중 6kg.
WORKSPACE_RADIUS_MM = 900.0
# 그리퍼(SG2/RG2) 자중을 뺀 실효 가반하중. 실측 전까지 보수적으로 잡는다.
MAX_PAYLOAD_G = 5000.0

# 신규 클래스(미확인)에 강제하는 프로파일 (FR-05, NFR-03a)
FALLBACK_PROFILE = "fragile"


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


def resolve_profile(obj: dict) -> str:
    """물체에 적용할 제어 프로파일을 결정한다.

    미확인 신규 클래스는 VLM이 무엇을 제안했든 fallback(fragile)로 강제한다 —
    사진 기반 추정치가 파지력·속도에 직접 반영되어서는 안 된다(NFR-03a, FR-05b).
    """
    if obj.get("needs_confirmation"):
        return FALLBACK_PROFILE
    profile = obj.get("profile")
    if profile not in ("normal", "fragile", "deformable"):
        return FALLBACK_PROFILE
    return profile


def _best_grasp(obj: dict) -> dict:
    candidates = obj.get("grasp_candidates") or []
    if not candidates:
        raise Rejected(
            f"'{obj['object_id']}'({obj.get('name_ko') or obj.get('class_name')})에 "
            f"유효한 파지 후보가 없습니다"
        )
    return max(candidates, key=lambda c: c.get("score", 0.0))


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
        profile = resolve_profile(obj)

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
            grasp = _best_grasp(obj)
            position = grasp["pose"]["position"]
            if _reach_mm(position) > WORKSPACE_RADIUS_MM:
                raise Rejected(
                    f"{where}: 파지 위치가 작업반경 {WORKSPACE_RADIUS_MM:g}mm를 벗어납니다"
                )
            held = step.object_id
            plan.append(PlanStep(
                skill="pick",
                object_id=step.object_id,
                profile=profile,
                grasp_pose=Pose(**grasp["pose"]),
                grasp_width_mm=grasp.get("width_mm") or None,
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
                profile=profile,
                bin_id=step.bin_id,
            ))

    if held is not None:
        raise Rejected(f"'{held}'를 집은 뒤 배치하는 스텝이 없습니다")

    return plan
