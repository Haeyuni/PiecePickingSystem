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
        self._cancelled: set[str] = set()
        self._on_event: Callable[[dict], Awaitable[None]] | None = None

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
        return json.loads(path.read_text(encoding="utf-8"))

    def use_fixture(self, name: str) -> None:
        """개발 중 시나리오 전환용 (mock 전용 — 실물 어댑터에는 없는 기능)."""
        self._fixture = name

    def robot_state(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "mode": self._mode,
            "current_skill": self._current_skill,
            "gripper_width_mm": 0.0 if self._current_skill == "none" else 42.0,
        }

    def subscribe_state(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        self._on_event = on_event

    def latest_color_jpeg(self) -> bytes | None:
        """mock에는 카메라가 없다 — 화면은 스트림 없음으로 처리한다."""
        return None

    def latest_depth_jpeg(self) -> bytes | None:
        return None

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

            return SkillResult(
                success=True,
                visual_verification_passed=True if skill == "pick" else None,
                cycle_time_ms=(time.monotonic() - started) * 1000,
                torque_trace=[0.4, 1.9, 2.6, 2.4] if skill == "pick" else [],
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

    async def home(self) -> None:
        self._mode = "busy"
        self._current_skill = "home"
        await self._emit_state()
        await asyncio.sleep(PHASE_DELAY_S * 2)
        self._mode = "idle"
        self._current_skill = "none"
        await self._emit_state()
