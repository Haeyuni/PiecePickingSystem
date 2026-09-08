"""Offline diagnostics regression: extract the method without importing ROS or starting a node.

2026-09-07: `_pick_real`은 좌표를 스스로 계산하지 않고 `_select_candidate`가 고른
후보의 `PickGeometry`를 그대로 실행한다(검사한 좌표와 실행 좌표가 갈라지면 안 되므로).
여기서는 예전 스텁이 만들던 것과 **같은 값**을 그 형태로 넣어, 로그·하강 계산이
바뀌지 않았음을 그대로 확인한다.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class DiagnosticsComplete(Exception):
    pass


class PickDepthDiagnosticsTest(unittest.TestCase):
    def test_depth_logs_and_failed_descent_exit(self):
        source = Path(__file__).parents[1] / "control" / "pick_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PickServer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pick_real")
        code = compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec")

        for ok, measured, feedback, canceled in (
            (True, True, True, False),
            (False, True, True, False),
            (False, False, True, False),
            (False, False, False, False),
            (False, False, True, True),
        ):
            with self.subTest(ok=ok, measured=measured, feedback=feedback, canceled=canceled):
                logs, commands, moves = [], [], []
                handle = NS(is_cancel_requested=False)
                goal = NS(request_id="req-depth", object_id="obj-depth", gripper_width_mm=20.0,
                          grasp_pose=NS(position=NS(x=10.0, y=20.0, z=100.0),
                                        orientation=NS(x=math.sqrt(0.9), y=0.0, z=0.0,
                                                       w=math.sqrt(0.1))))
                # A tilted rotation with nonzero tool X/Y. Tool Y contributes to base Z.
                shift = [2.0, -5.58, -2.44]  # R @ [2, 3, 5.3]
                orientation = [90.0, math.degrees(math.acos(-0.8)), -90.0]
                # 예전 스텁이 계산하던 값 그대로:
                #   tcp_posx_for_grasp = [8.0, 25.58, 102.44]
                #   target  = tcp + 4.0(하강) * axis  = [8.0, 23.18, 99.24]
                #   approach = target - 80.0 * axis   = [8.0, 71.18, 163.24]
                approach_axis = [0.0, -0.6, -0.8]
                target_posx = [8.0, 23.18, 99.24, *orientation]
                approach_posx = [8.0, 71.18, 163.24, *orientation]
                selected = NS(
                    candidate=NS(pose=goal.grasp_pose, gripper_width_mm=20.0,
                                 candidate_id="obj-depth#0"),
                    geometry=NS(target_posx=target_posx, approach_posx=approach_posx,
                                approach_axis=approach_axis,
                                pad_reference_mm=[target_posx[i] + shift[i] for i in range(3)]))
                wrist_pose = [0.0, 0.0, 0.0, *orientation]

                def current(*args, **kwargs):
                    if len(moves) < 2:
                        return [0.0, 0.0, 0.0, *orientation]
                    return [10.0, 20.0, 100.24, *orientation] if measured else None

                def move(client, pos, *args, **kwargs):
                    moves.append(pos)
                    if len(moves) == 1:
                        return True, pos
                    self.assertEqual(len(moves), 2, "no lift after failed descent")
                    handle.is_cancel_requested = canceled
                    return ok, ([10.0, 20.0, 100.24, *orientation] if feedback else None)

                def phase(*args):
                    if len(moves) == 2:
                        raise DiagnosticsComplete

                motion = NS(
                    get_current_posx=current,
                    grasp_center_from_posx=lambda p, o: [p[i] + shift[i] for i in range(3)],
                    rotation_diff_deg=lambda *a: 0.0,
                    move_linear=move,
                    gripper_width_command=lambda w, f: w,
                    send_gripper_command=lambda c, command: commands.append(command) or True,
                    wait_gripper_settled=lambda *a: 0.0,
                )
                server = NS(
                    _posx_client=None, _movel_client=None, _gripper_cmd_client=None,
                    # 도달 불가 알람 감시자(dsr_motion.MotionErrorMonitor). 이 테스트는
                    # 알람 없는 정상 경로만 보므로 move_linear에 그대로 넘겨지기만 하면 된다.
                    _motion_errors=None,
                    _grasp_center_offset_mm=[2.0, 3.0, 5.3], _pick_depth_extra_mm=4.0,
                    _approach_height_mm=80.0, _linear_vel_mm_s=30.0, _linear_acc_mm_s2=30.0,
                    _rot_vel_deg_s=20.0, _rot_acc_deg_s2=20.0,
                    _gripper_open_m=0.110, _gripper_open_force_n=40.0,
                    _gripper_width_margin_mm=30.0, _gripper_joint_angle=0.0,
                    _open_width_m=lambda w: 0.050, _publish_phase=phase,
                    _min_grip_width_mm=5.0, _grip_close_ratio=0.8, _profile_force_n={},
                    get_logger=lambda: NS(info=logs.append, warning=logs.append),
                )
                namespace = dict(math=math, dsr_motion=motion, _Canceled=type("Canceled", (Exception,), {}),
                                 Pick=NS(Feedback=NS(PHASE_APPROACHING=1, PHASE_CONTACT_DETECTED=2)))
                exec(code, namespace)
                if canceled:
                    self.assertIsNone(
                        namespace["_pick_real"](server, handle, goal, selected, wrist_pose))
                else:
                    with self.assertRaises(DiagnosticsComplete if ok else RuntimeError):
                        namespace["_pick_real"](server, handle, goal, selected, wrist_pose)
                self.assertEqual(commands, [0.110, 0.050], "only opening commands, never close on failure")
                equation = next(line for line in logs if "pick depth equation" in line)
                self.assertIn("99.240000 = 100.000000 - (-2.440000) + 4.000000 * (-0.800000)", equation)
                self.assertIn("predicted_modeled_reference_z=96.800000", equation)
                self.assertIn("fitted_pose_quaternion_xyzw=", logs[0])
                self.assertIn("fitted_pose_xyz_mm=(10.000, 20.000, 100.000)", logs[0])
                actual = next(line for line in logs if "actual_tcp_z=" in line)
                self.assertIn("final_commanded_tcp_z=99.240000", actual)
                if measured or feedback:
                    self.assertIn("actual_tcp_z=100.240000 dz=+1.000000", actual)
                    self.assertIn("get_current_posx" if measured else "feedback", actual)
                    self.assertTrue(any("reconstructed_pad_reference_xyz_mm=" in line and
                                        "not actual physical finger" in line for line in logs))
                else:
                    self.assertIn("actual_tcp_z=unavailable dz=unavailable", actual)
                for line in logs:
                    if "pick " in line or "actual_tcp_z=" in line or "reference residual" in line:
                        self.assertIn("request_id=req-depth object=obj-depth", line)


if __name__ == "__main__":
    unittest.main()
