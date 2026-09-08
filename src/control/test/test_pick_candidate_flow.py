"""pick_server가 후보 선택 결과를 어떻게 다루는지 — ROS2 없이 `execute_callback`만 떼어 검증한다.

여기서 지키려는 계약은 두 개다.

1. **후보가 전부 탈락하면 로봇을 한 번도 움직이지 않는다.** `_pick_real`도 fake phase도
   지나가지 않고 `no_feasible_grasp`로 끝나야 한다. 모션 도중 실패와 구분되어야
   orchestrator가 그리퍼 상태를 의심하지 않는다.
2. **고른 후보가 그대로 실행에 넘어간다.** 1순위가 아닌 후보가 뽑혔는데 실행이 1순위를
   쓰면 IK를 통과시킨 근거가 사라진다.

`test_grasp_selection.py`가 "무엇을 고르는가"를, 이 파일이 "고른 것을 어떻게 쓰는가"를 본다.
"""
import ast
import pathlib
import sys
import time
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))                    # src/control
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "perception_common"))

from control import grasp_selection as gs  # noqa: E402

SOURCE = pathlib.Path(__file__).parents[1] / "control" / "pick_server.py"


def _method(name):
    """pick_server.py에서 메서드 하나만 떼어 컴파일한다 — rclpy/dsr_msgs2를 import하지 않는다."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PickServer")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec")


class _Result:
    """Pick.Result 대역 — 어떤 필드가 채워졌는지 보려고 그대로 들고 있는다."""

    def __init__(self):
        self.success = False
        self.failure_reason = ""
        self.selected_candidate_id = ""
        self.retries_used = 0
        self.cycle_time_ms = 0.0
        self.torque_trace_summary = []
        self.visual_verification_passed = False
        self.visual_verification_note = ""


PICK = NS(
    Result=NS(REASON_NONE="none", REASON_NO_CONTACT="no_contact",
              REASON_GRASP_FAILED="grasp_failed", REASON_UNREACHABLE="unreachable",
              REASON_NO_FEASIBLE_GRASP="no_feasible_grasp"),
    Feedback=NS(PHASE_APPROACHING="approaching", PHASE_CONTACT_DETECTED="contact_detected",
                PHASE_LIFTING="lifting", PHASE_VERIFYING="verifying"),
)


def candidate(rank, width=40.0, score=0.9):
    return gs.Candidate(candidate_id=f"obj_001#{rank}", rank=rank,
                        pose=NS(position=NS(x=400.0, y=0.0, z=300.0),
                                orientation=NS(x=0.0, y=1.0, z=0.0, w=1.0)),
                        score=score, gripper_width_mm=width)


def evaluation(candidate_obj, status):
    geometry = gs.PickGeometry(
        target_posx=[400.0, 0.0, 300.0, 0.0, 180.0, 0.0],
        approach_posx=[400.0, 0.0, 380.0, 0.0, 180.0, 0.0],
        approach_axis=[0.0, 0.0, -1.0], pad_reference_mm=[400.0, 0.0, 295.0])
    return gs.Evaluation(candidate=candidate_obj, status=status, geometry=geometry,
                         rejection_reason="" if status == gs.STATUS_SELECTED else "테스트 사유")


class Harness:
    """`execute_callback`을 돌릴 최소 환경. 어떤 부수효과가 일어났는지 기록한다."""

    def __init__(self, selection, fake=False):
        self.selection = selection            # (selected, evaluations) 또는 None
        self.fake = fake
        self.moved = []                       # _pick_real / fake phase 호출 기록
        self.states = []
        self.logs = []
        self.published = []
        self.cache = {}
        self.aborted = False
        self.succeeded = False

    # --- PickServer가 부르는 것들 --------------------------------------------
    def _select_candidate(self, goal, goal_handle):
        selected, evaluations = self.selection
        self.published.append(selected.candidate.candidate_id if selected else "")
        return selected, evaluations, None

    def _pick_real(self, goal_handle, goal, selected, wrist_pose):
        self.moved.append(("real", selected.candidate.candidate_id))
        evidence = {
            "pre": {"grip": False}, "post": {"grip": True}, "lift": {"grip": True},
            "lift_width_mm": 33.0,
        }
        return 33.0, 30.0, [400.0, 0.0, 300.0, 0.0, 180.0, 0.0], evidence

    def _publish_phase(self, goal_handle, phase):
        self.moved.append(("phase", phase))

    def _grip_detected(self):
        return True

    def _injected_failure(self, object_id):
        return False

    def _selected_id(self, selected):
        return selected.candidate.candidate_id if selected is not None else ""

    def _result(self, success, reason, started, visual_passed=False, torque=None,
                candidate_id="", source_observation_id="", executed_tcp=None):
        result = _Result()
        result.success = success
        result.failure_reason = reason
        result.selected_candidate_id = candidate_id
        result.visual_verification_passed = visual_passed
        result.torque_trace_summary = torque or []
        result.source_observation_id = source_observation_id
        result.has_executed_tcp_posx = executed_tcp is not None
        result.executed_tcp_posx = executed_tcp or []
        return result

    def get_logger(self):
        return NS(info=self.logs.append, warning=self.logs.append, error=self.logs.append)

    _min_grip_width_mm = 5.0


def run_execute(harness, candidate_count=3):
    store = NS(set_busy=lambda skill: harness.states.append(f"busy:{skill}"),
               set_idle=lambda: harness.states.append("idle"),
               set_error=lambda: harness.states.append("error"),
               set_gripper=lambda **kw: harness.states.append("gripper"),
               snapshot=lambda: {"mode": "idle"})
    goal_handle = NS(request_id="rq-1", is_cancel_requested=False,
                     succeed=lambda: setattr(harness, "succeeded", True),
                     abort=lambda: setattr(harness, "aborted", True),
                     canceled=lambda: None)
    goal_handle.request = NS(request_id="rq-1", trace_id="tr-1",
                              object_id="obj_001", grip_level=3,
                             source_observation_id="obs-1",
                             grasp_candidates=[None] * candidate_count)
    harness._cache = NS(get=lambda rid: harness.cache.get(rid),
                        put=lambda rid, result: harness.cache.__setitem__(rid, result))
    namespace = dict(Pick=PICK, store=store, time=time,
                     is_fake_robot=lambda: harness.fake,
                     grasp_selection=gs,
                     PHASES=[PICK.Feedback.PHASE_APPROACHING, PICK.Feedback.PHASE_CONTACT_DETECTED,
                             PICK.Feedback.PHASE_LIFTING, PICK.Feedback.PHASE_VERIFYING],
                     FAKE_PHASE_DURATION_S=0.0,
                     _LiftFailedError=type("_LiftFailedError", (RuntimeError,), {}))
    exec(_method("execute_callback"), namespace)
    return namespace["execute_callback"](harness, goal_handle)


class TestCase5NoFeasibleCandidate(unittest.TestCase):
    """Case 5 — 전부 탈락하면 로봇을 움직이지 않고 명확한 실패를 돌려준다."""

    def setUp(self):
        self.evaluations = [evaluation(candidate(0, width=0.0), gs.STATUS_WIDTH_INVALID),
                            evaluation(candidate(1), gs.STATUS_IK_FAILED),
                            evaluation(candidate(2), gs.STATUS_JOINT_LIMIT)]

    def test_returns_no_feasible_grasp(self):
        harness = Harness((None, self.evaluations))
        result = run_execute(harness)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_feasible_grasp")
        self.assertEqual(result.selected_candidate_id, "")

    def test_robot_never_moves(self):
        harness = Harness((None, self.evaluations))
        run_execute(harness)
        self.assertEqual(harness.moved, [], "후보가 없으면 모션도 그리퍼도 건드리지 않는다")

    def test_does_not_enter_error_mode(self):
        """팔도 그리퍼도 그대로다 — 사람이 리셋해야 하는 error 상태로 두면 안 된다."""
        harness = Harness((None, self.evaluations))
        run_execute(harness)
        self.assertNotIn("error", harness.states)
        self.assertIn("idle", harness.states)
        self.assertTrue(harness.succeeded)
        self.assertFalse(harness.aborted)

    def test_fake_mode_also_refuses_without_moving(self):
        """fake 모드에서도 같은 계약이어야 실물로 바꿀 때 새로 검증할 것이 없다."""
        harness = Harness((None, self.evaluations), fake=True)
        result = run_execute(harness)
        self.assertEqual(result.failure_reason, "no_feasible_grasp")
        self.assertEqual(harness.moved, [])

    def test_result_is_cached_like_any_other(self):
        """중복 request_id 재전송이 다시 검사를 돌리지 않게 캐시에 들어가야 한다."""
        harness = Harness((None, self.evaluations))
        run_execute(harness)
        self.assertIn("rq-1", harness.cache)

    def test_reasons_are_logged_per_candidate(self):
        harness = Harness((None, self.evaluations))
        run_execute(harness)
        joined = " ".join(harness.logs)
        for status in ("width_invalid", "ik_failed", "joint_limit"):
            self.assertIn(status, joined)


class TestCase6ExecutesTheSelectedCandidate(unittest.TestCase):
    """Case 6 — 고른 후보가 그대로 실행되고 결과에 실린다(기존 정상 pick 경로)."""

    def test_non_top_candidate_is_the_one_executed(self):
        chosen = evaluation(candidate(2), gs.STATUS_SELECTED)
        harness = Harness((chosen, [evaluation(candidate(0), gs.STATUS_WIDTH_INVALID),
                                    evaluation(candidate(1), gs.STATUS_IK_FAILED), chosen]))
        result = run_execute(harness)
        self.assertEqual(harness.moved, [("real", "obj_001#2")])
        self.assertTrue(result.success)
        self.assertEqual(result.selected_candidate_id, "obj_001#2")

    def test_top_candidate_path_is_unchanged(self):
        chosen = evaluation(candidate(0), gs.STATUS_SELECTED)
        harness = Harness((chosen, [chosen]))
        result = run_execute(harness)
        self.assertEqual(harness.moved, [("real", "obj_001#0")])
        self.assertTrue(result.success)

    def test_fake_mode_still_runs_every_phase(self):
        chosen = evaluation(candidate(0), gs.STATUS_SELECTED)
        harness = Harness((chosen, [chosen]), fake=True)
        result = run_execute(harness)
        self.assertEqual([entry[1] for entry in harness.moved],
                         ["approaching", "contact_detected", "lifting", "verifying"])
        self.assertTrue(result.success)


class TestGoalAdapters(unittest.TestCase):
    """goal → 후보/물체 컨텍스트 변환. 예전 goal(후보 목록 없음)도 그대로 지나가야 한다."""

    def _exec_namespace(self):
        """두 정적 메서드를 떼어 컴파일한다. 반환 타입 주석이 def 시점에 평가되므로
        namespace에 grasp_selection이 먼저 있어야 한다."""
        namespace = dict(grasp_selection=gs)
        exec(_method("_candidates_from_goal"), namespace)
        exec(_method("_object_context"), namespace)
        return namespace

    def _convert(self, goal, logs=None):
        """`_candidates_from_goal`을 로거만 갖춘 가짜 self로 부른다."""
        namespace = self._exec_namespace()
        server = NS(get_logger=lambda: NS(info=(logs if logs is not None else []).append,
                                          warning=(logs if logs is not None else []).append,
                                          error=(logs if logs is not None else []).append))
        return namespace["_candidates_from_goal"](server, goal)

    def test_candidate_list_is_converted_in_order(self):
        raw = [NS(candidate_id="obj_001#0", pose="p0", score=0.9, gripper_width_mm=30.0,
                  strategy="graspnet_baseline"),
               NS(candidate_id="obj_001#1", pose="p1", score=0.5, gripper_width_mm=40.0,
                  strategy="graspnet_baseline")]
        goal = NS(object_id="obj_001", grasp_candidates=raw, grasp_pose="fallback",
                  gripper_width_mm=0.0)
        converted = self._convert(goal)
        self.assertEqual([c.candidate_id for c in converted], ["obj_001#0", "obj_001#1"])
        self.assertEqual([c.rank for c in converted], [0, 1])

    def test_empty_list_falls_back_to_single_grasp_pose(self):
        goal = NS(object_id="obj_001", grasp_candidates=[], grasp_pose="only",
                  gripper_width_mm=25.0)
        converted = self._convert(goal)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].pose, "only")
        self.assertEqual(converted[0].candidate_id, "")
        self.assertEqual(converted[0].gripper_width_mm, 25.0)

    # --- 명령 대상 물체와의 결속 (2026-09-08) --------------------------------
    # candidate_id는 "<object_id>#<순위>"라 출처가 값 안에 있다. 다른 물체의 후보가
    # 섞여 들어오면 점수가 높다는 이유로 명령과 무관한 물체를 집을 수 있다.

    def test_foreign_candidates_are_dropped(self):
        raw = [NS(candidate_id="obj_009#0", pose="p0", score=0.99, gripper_width_mm=30.0,
                  strategy="graspnet_baseline"),          # 다른 물체 — 게다가 최고점
               NS(candidate_id="obj_001#0", pose="p1", score=0.40, gripper_width_mm=40.0,
                  strategy="graspnet_baseline")]
        goal = NS(object_id="obj_001", grasp_candidates=raw, grasp_pose="fallback",
                  gripper_width_mm=0.0)
        logs = []
        converted = self._convert(goal, logs)
        self.assertEqual([c.candidate_id for c in converted], ["obj_001#0"])
        self.assertEqual(converted[0].rank, 0, "남은 후보의 순위는 다시 매겨져야 한다")
        self.assertTrue(any("대상불일치" in line for line in logs),
                        "조용히 버리면 안 된다 — 섞인 사실이 로그에 남아야 한다")

    def test_blank_candidate_id_is_kept(self):
        """출처를 모르는 후보(예전 planner 형식)는 버리지 않는다 — goal 자체가 그 물체 것이다."""
        raw = [NS(candidate_id="", pose="p0", score=0.9, gripper_width_mm=30.0,
                  strategy="")]
        goal = NS(object_id="obj_001", grasp_candidates=raw, grasp_pose="fallback",
                  gripper_width_mm=0.0)
        self.assertEqual(len(self._convert(goal)), 1)

    def test_all_foreign_means_no_candidates(self):
        """전부 남의 것이면 빈 목록 — 그러면 execute_callback이 no_feasible_grasp로 끝낸다."""
        raw = [NS(candidate_id="obj_009#0", pose="p0", score=0.99, gripper_width_mm=30.0,
                  strategy="")]
        goal = NS(object_id="obj_001", grasp_candidates=raw, grasp_pose="fallback",
                  gripper_width_mm=0.0)
        self.assertEqual(self._convert(goal), [])

    def test_zero_center_means_not_supplied(self):
        namespace = self._exec_namespace()
        goal = NS(object_id="obj_001", object_center_mm=NS(x=0.0, y=0.0, z=0.0),
                  object_height_mm=0.0, depth_valid_ratio=0.0)
        context = namespace["_object_context"].__func__(goal)
        self.assertIsNone(context.center_mm)
        self.assertIsNone(context.support_z_mm)

    def test_real_center_is_carried_through(self):
        namespace = self._exec_namespace()
        goal = NS(object_id="obj_001", object_center_mm=NS(x=400.0, y=-20.0, z=310.0),
                  object_height_mm=30.0, depth_valid_ratio=0.85)
        context = namespace["_object_context"].__func__(goal)
        self.assertEqual(context.center_mm, (400.0, -20.0, 310.0))
        self.assertAlmostEqual(context.support_z_mm, 280.0)


if __name__ == "__main__":
    unittest.main()
