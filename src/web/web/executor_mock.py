"""mock 실행 어댑터 — 로봇·GPU·perception 없이 web 전체를 돌린다 (개발계획.md 4.2절 B2).

`data/mock/world_state_*.json`을 월드 상태로 쓰고, pick/place_into는 실제 액션 대신
phase를 순서대로 흘려보낸 뒤 성공을 반환한다. **타이밍과 상태 전이는 흉내 내되,
인터페이스는 실물과 동일하다** — 그래야 여기서 통과한 코드가 B4에서도 그대로 돈다.

실패 주입: MOCK_FAIL_OBJECT에 object_id를 넣으면 그 물체의 pick이 grasp_failed로
실패한다. 재계획 경로(FR-16)를 로봇 없이 확인하기 위한 스위치다.
"""
import asyncio
import json
import logging
import os
import pathlib
import time
from typing import Awaitable, Callable

from .executor import SkillGoal, SkillResult

logger = logging.getLogger(__name__)

MOCK_DIR = pathlib.Path(os.environ.get("MOCK_DIR", "/data/mock"))
DEFAULT_FIXTURE = os.environ.get("MOCK_FIXTURE", "world_state_normal")

# 실제 액션의 phase 순서 (Pick/PlaceInto.action의 Feedback 상수)
PICK_PHASES = ["approaching", "contact_detected", "lifting", "verifying"]
PLACE_PHASES = ["moving", "inserting", "releasing", "verifying"]

# 각 phase 사이 지연. 진행률 UI가 눈에 보이도록 하되 테스트가 느려지지 않을 정도.
PHASE_DELAY_S = float(os.environ.get("MOCK_PHASE_DELAY_S", "0.4"))


class MockExecutor:
    def __init__(self):
        self._fixture = DEFAULT_FIXTURE
        self._mode = "idle"
        self._current_skill = "none"
        self._current_request_id: str | None = None
        self._gripper_width_mm = 0.0
        self._cancelled: set[str] = set()
        self._on_event: Callable[[dict], Awaitable[None]] | None = None
        # observe()가 찍는다. 픽스처 파일 자체는 고정 스냅샷이라 stamp가 안 바뀌므로(예:
        # world_state_normal.json), 이게 없으면 orchestrator._wait_for_fresh_observation의
        # "stamp가 바뀔 때까지 기다린다"가 매 명령마다 타임아웃까지 통째로 날아간다.
        self._stamp_override: dict | None = None
        self._observation_id: str | None = None
        self._trace_id: str | None = None

    # --- 수명주기 -----------------------------------------------------------

    async def start(self) -> None:
        logger.info("mock 실행기 시작 (픽스처=%s, 경로=%s)", self._fixture, MOCK_DIR)

    async def close(self) -> None:
        pass

    # --- 상태 ---------------------------------------------------------------

    def get_latest_world_state(self) -> dict | None:
        path = MOCK_DIR / f"{self._fixture}.json"
        if not path.exists():
            logger.error("픽스처를 찾을 수 없습니다: %s", path)
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if self._stamp_override is not None:
            state["stamp"] = self._stamp_override
        if self._observation_id is not None:
            state["observation_id"] = self._observation_id
        if self._trace_id is not None:
            state["trace_id"] = self._trace_id
        return state

    def use_fixture(self, name: str) -> None:
        """개발 중 시나리오 전환용 (mock 전용 — 실물 어댑터에는 없는 기능)."""
        self._fixture = name

    def robot_state(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "mode": self._mode,
            "current_skill": self._current_skill,
            "gripper_width_mm": (self._gripper_width_mm
                                  if self._current_skill == "none" else 42.0),
        }

    def subscribe_state(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        self._on_event = on_event

    def latest_raw_color_jpeg(self) -> bytes | None:
        """mock에는 카메라가 없다 — 화면은 스트림 없음으로 처리한다."""
        return None

    def latest_color_jpeg(self) -> bytes | None:
        return None

    def latest_depth_jpeg(self) -> bytes | None:
        return None

    async def observe(self, trace_id: str, mode: str = "full") -> dict | None:
        """SAM·VLM 없이 신선도만 흉내 낸다 — 픽스처 내용(물체 목록)은 그대로 두고
        stamp만 지금 시각으로 찍는다. orchestrator는 "새 관측이 왔는가"만 보므로 이걸로
        충분하고, 실물처럼 초 단위로 기다리게 하면 mock의 존재 이유(로봇·GPU 없이 빠르게
        확인)가 없어진다."""
        now = time.time()
        self._stamp_override = {"sec": int(now), "nanosec": int((now % 1) * 1e9)}
        self._observation_id = f"mock-{trace_id}-{time.monotonic_ns()}"
        self._trace_id = trace_id
        world = self.get_latest_world_state()
        object_count = len((world or {}).get("objects", []))
        logger.info("mock 관측 트리거 (mode=%s, trace=%s) — 물체 %d개", mode, trace_id, object_count)
        return {"success": True, "failure_reason": "none", "object_count": object_count,
                "observation_id": self._observation_id, "stamp": self._stamp_override,
                "cycle_time_ms": 0.0, "cancelled": False}

    async def _emit_state(self) -> None:
        if self._on_event:
            await self._on_event({"type": "robot_state", **self.robot_state()})

    # --- 스킬 실행 ----------------------------------------------------------

    async def _run(self, goal: SkillGoal, phases: list[str], skill: str,
                   on_feedback) -> SkillResult:
        started = time.monotonic()
        self._mode = "busy"
        self._current_skill = skill
        self._current_request_id = goal.request_id
        await self._emit_state()
        try:
            for phase in phases:
                if goal.request_id in self._cancelled:
                    self._cancelled.discard(goal.request_id)
                    return SkillResult(success=False, failure_reason="no_contact",
                                       cycle_time_ms=(time.monotonic() - started) * 1000,
                                       cancelled=True)
                await on_feedback(goal.request_id, phase)
                await asyncio.sleep(PHASE_DELAY_S)

            fail_target = os.environ.get("MOCK_FAIL_OBJECT")
            if skill == "pick" and fail_target and goal.object_id == fail_target:
                return SkillResult(success=False, failure_reason="grasp_failed",
                                   visual_verification_passed=False,
                                   cycle_time_ms=(time.monotonic() - started) * 1000)

            pose = goal.grasp_pose or {}
            position = pose.get("position") or {}
            return SkillResult(
                success=True,
                visual_verification_passed=True if skill == "pick" else None,
                cycle_time_ms=(time.monotonic() - started) * 1000,
                torque_trace=[0.4, 1.9, 2.6, 2.4] if skill == "pick" else [],
                selected_candidate_id=(goal.grasp_candidates[0].get("candidate_id", "")
                                       if skill == "pick" and goal.grasp_candidates else ""),
                source_observation_id=(goal.source_observation_id if skill == "pick" else ""),
                executed_tcp_posx=(
                    [float(position.get("x", 0.0)), float(position.get("y", 0.0)),
                     float(position.get("z", 0.0)), 0.0, 180.0, 0.0]
                    if skill == "pick" else None),
            )
        finally:
            self._mode = "idle"
            self._current_skill = "none"
            self._current_request_id = None
            await self._emit_state()

    async def call_pick(self, goal: SkillGoal, on_feedback) -> SkillResult:
        return await self._run(goal, PICK_PHASES, "pick", on_feedback)

    async def call_place_into(self, goal: SkillGoal, on_feedback) -> SkillResult:
        return await self._run(goal, PLACE_PHASES, "place_into", on_feedback)

    # --- 직접 제어 ----------------------------------------------------------

    async def stop(self) -> str | None:
        cancelled = self._current_request_id
        if cancelled:
            self._cancelled.add(cancelled)
        logger.info("mock 정지 요청 (취소 대상=%s)", cancelled)
        return cancelled

    async def home(self, open_gripper: bool = False) -> None:
        self._mode = "busy"
        self._current_skill = "home"
        await self._emit_state()
        await asyncio.sleep(PHASE_DELAY_S * 2)
        if open_gripper:
            await asyncio.sleep(PHASE_DELAY_S)
            self._gripper_width_mm = 110.0
        self._mode = "idle"
        self._current_skill = "none"
        await self._emit_state()
