"""실행 전 승인 게이트(명령 1건당 1회) — orchestrator._await_approval.

검증을 통과한 시퀀스도 사람이 승인해야 실행된다는 정책(orchestrator.py 모듈 docstring)이
승인/거부/라벨수정 세 경로에서 실제로 지켜지는지 확인한다. planner·DB·ROS는 전부
가짜로 대체한다 — 여기서 보는 것은 큐 기반 상태 머신이다.
"""
import asyncio
import unittest
from types import SimpleNamespace as NS

from web import orchestrator


def trace(trace_id="tr-1"):
    return {
        "trace_id": trace_id,
        "sequence_id": "seq-1",
        "validation_status": "approved",
        "validation_reason": None,
        "steps": [{"skill": "pick", "object_id": "obj_001", "grip_level": 3}],
        "objects": [{"object_id": "obj_001", "class_name": "toothpaste"}],
    }


def world_state():
    return {"objects": [{"object_id": "obj_001", "class_name": "toothpaste", "name_ko": "치약"}]}


class ApprovalFlowTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._broadcasts = []

        async def fake_broadcast(event):
            self._broadcasts.append(event)

        self._real_broadcast = orchestrator.hub.broadcast
        orchestrator.hub.broadcast = fake_broadcast

    def tearDown(self):
        orchestrator.hub.broadcast = self._real_broadcast
        orchestrator._pending_approvals.clear()

    async def test_approve_returns_world_state_and_steps_unchanged(self):
        t = trace()
        task = asyncio.create_task(
            orchestrator._await_approval(t, world_state(), "치약 왼쪽으로", None, NS()))
        await asyncio.sleep(0)  # _await_approval이 큐에 등록하고 broadcast할 시간을 준다
        self.assertTrue(orchestrator.has_pending_approval())

        orchestrator.resolve_approval(t["trace_id"], {"action": "approve"})
        result = await task

        self.assertIsNotNone(result)
        ws, steps = result
        self.assertEqual(ws["objects"][0]["object_id"], "obj_001")
        self.assertEqual(steps, t["steps"])
        self.assertFalse(orchestrator.has_pending_approval())
        self.assertEqual(self._broadcasts[0]["type"], "execution_approval_needed")

    async def test_reject_returns_none_without_touching_validation_status(self):
        t = trace()
        task = asyncio.create_task(
            orchestrator._await_approval(t, world_state(), "치약 왼쪽으로", None, NS()))
        await asyncio.sleep(0)

        orchestrator.resolve_approval(t["trace_id"], {"action": "reject"})
        result = await task

        self.assertIsNone(result)
        self.assertEqual(t["validation_status"], "approved")  # 사용자 거부, 검증 거부 아님

    async def test_correct_label_replans_and_asks_again(self):
        t = trace()
        ws = world_state()

        async def fake_replan(trace_id, command_text, world_state_arg, previous_failure, executor):
            self.assertEqual(world_state_arg["objects"][0]["class_name"], "wet_wipes")
            return {
                "sequence_id": "seq-2", "validation_status": "approved",
                "validation_reason": None,
                "steps": [{"skill": "pick", "object_id": "obj_001", "grip_level": 4}],
            }

        orchestrator._plan_with_grasp_retry = fake_replan
        task = asyncio.create_task(
            orchestrator._await_approval(t, ws, "물티슈 왼쪽으로", None, NS()))
        await asyncio.sleep(0)

        orchestrator.resolve_approval(t["trace_id"], {
            "action": "correct_label", "object_id": "obj_001",
            "class_name": "wet_wipes", "name_ko": "물티슈",
        })
        await asyncio.sleep(0)  # 재계획 + 두 번째 broadcast까지 처리될 시간을 준다

        self.assertTrue(orchestrator.has_pending_approval())  # 다시 승인 대기 중
        self.assertEqual(t["sequence_id"], "seq-2")
        self.assertEqual(t["steps"][0]["grip_level"], 4)
        self.assertEqual(
            [e["type"] for e in self._broadcasts],
            ["execution_approval_needed", "execution_approval_needed"],
        )

        orchestrator.resolve_approval(t["trace_id"], {"action": "approve"})
        result = await task
        self.assertIsNotNone(result)

    async def test_correct_label_rejected_by_replan_is_not_user_rejection(self):
        t = trace()

        async def fake_replan(*args, **kwargs):
            return {"sequence_id": "seq-2", "validation_status": "rejected",
                    "validation_reason": "가반하중 초과", "steps": []}

        orchestrator._plan_with_grasp_retry = fake_replan
        task = asyncio.create_task(
            orchestrator._await_approval(t, world_state(), "치약 왼쪽으로", None, NS()))
        await asyncio.sleep(0)

        orchestrator.resolve_approval(t["trace_id"], {
            "action": "correct_label", "object_id": "obj_001", "class_name": "heavy_box",
        })
        result = await task

        self.assertIsNone(result)
        self.assertEqual(t["validation_status"], "rejected")
        self.assertEqual(t["validation_reason"], "가반하중 초과")

    def test_resolve_approval_on_unknown_trace_returns_false(self):
        self.assertFalse(orchestrator.resolve_approval("no-such-trace", {"action": "approve"}))


if __name__ == "__main__":
    unittest.main()
