"""M0609/RG2 실물 실행 어댑터.

로봇 PC에서 실제로 떠 있는 액션/서비스에 직접 붙는다(`ros2 action/service list`로 확인,
2026-09-04):

    /dsr01/motion/movej_h2r  [dsr_msgs2/action/MovejH2r]  Home 관절 이동
    /dsr01/motion/movel_h2r  [dsr_msgs2/action/MovelH2r]  접근/하강/들어올리기 직선 이동
    /onrobot/sendCommand     [onrobot_rg_msgs/srv/SetCommand]  RG2 open/close
    /onrobot/pose            [onrobot_rg_msgs/srv/GripperPose] 관절각 → 개폭(mm) 변환
    /onrobot_joint_states    [sensor_msgs/msg/JointState]      그리퍼 관절각 구독(정착 대기용)

control 패키지의 /pick·/home(sort_msgs)도 같은 드라이버 액션을 감싸지만 grasp_test는 그걸
쓰지 않고 드라이버에 바로 붙는다:
1) `Pick.Result`에는 실제 개폭이 없다. grasp_test는 결과표의 `rg2_grip_state`를 채우려면
   그리퍼가 실제로 얼마나 닫혔는지가 필요한데 sort_msgs/Pick.action은 그 값을 내보내지 않는다.
2) `control/control/pick_server.py`가 그리퍼를 열지 않는다(닫기만 한다) — grasp_test는
   trial마다 물체를 다시 쥐어야 하므로 open 어댑터가 어차피 따로 필요하다.
control 코드 자체는 건드리지 않는다(README 참조) — 같은 드라이버를 병행 사용할 뿐이다.

**그리퍼는 `control_msgs/action/GripperCommand`(`/rg6_controller`)를 쓰지 않는다.** 같은
로봇 PC에서 `control` 패키지를 실물로 검증하던 중(2026-09-04, `control/control/dsr_motion.py`
참조) 그 액션의 `execute_callback`(onrobot_rg_control 패키지)에 실물로 재현되는 문제가 둘
있는 것으로 확인됐다 — (1) `position`(미터, 개폭)을 관절각(rad)으로 변환 없이 그대로 넘기고,
(2) 첫 목표 처리 시 내부 상태가 아직 `None`인 채로 비교 연산을 해 예외로 죽는다. 대신 같은
드라이버가 제공하는 `/onrobot/sendCommand`(문자 명령)를 쓰고, 이 서비스는 완료를 기다리지
않는 fire-and-forget이라 `/onrobot_joint_states`를 직접 폴링해 정착을 기다린다. control의
`dsr_motion.py`를 그대로 재사용하지 않고 여기 자체 구현이 있는 이유는 이 파일 자체의
설계(모듈 docstring 상단 1),2) 참조)와 같다 — 대신 같은 서비스/토픽 계약을 그대로 따른다.

파지 자세의 방향(orientation)은 `grasp/grasp/strategies/heuristic_pca.py`의
`_pose_from_closing_axis` 기본값과 같다: 접근은 base -Z(수직 하강), 닫는 방향(폐합축)은
base +X. grasp_test가 비교하는 academic 모델(ggcnn/graspnet_baseline/contact_graspnet/
pca_normal)은 x_m,y_m,z_m 위치만 내고 자세를 만들지 않으므로, 위치만으로 비교하려면 방향은
고정해야 한다 — 새로 정하지 않고 이미 이 저장소에서 검증된 관례를 그대로 쓴다.
"""
import time
from pathlib import Path

import numpy as np
import rclpy
from dsr_msgs2.action import MovejH2r, MovelH2r
from onrobot_rg_msgs.srv import GripperPose, SetCommand
from perception_common import geometry
from perception_common.robot_pose import RobotPoseClient
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

from . import action_client
from .motion_precheck import MotionPrecheck

MOVEJ_ACTION = '/dsr01/motion/movej_h2r'
MOVEL_ACTION = '/dsr01/motion/movel_h2r'
GRIPPER_COMMAND_SERVICE = '/onrobot/sendCommand'
GRIPPER_POSE_SERVICE = '/onrobot/pose'
GRIPPER_JOINT_STATES_TOPIC = '/onrobot_joint_states'

# heuristic_pca._pose_from_closing_axis의 기본값과 동일: 접근 base -Z, 닫는 방향 base +X
# (닫는 방향을 정할 근거가 없을 때 그 함수가 쓰는 폴백과 같다).
_TOPDOWN_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


class RobotExecutor:
    def __init__(self, node, config):
        self._node, self._config = node, config
        self._movej_client = ActionClient(node, MovejH2r, MOVEJ_ACTION)
        self._movel_client = ActionClient(node, MovelH2r, MOVEL_ACTION)
        self._gripper_cmd_client = node.create_client(SetCommand, GRIPPER_COMMAND_SERVICE)
        self._gripper_pose_client = node.create_client(GripperPose, GRIPPER_POSE_SERVICE)
        self._gripper_joint_angle = None
        node.create_subscription(JointState, GRIPPER_JOINT_STATES_TOPIC, self._on_gripper_state, 5)
        self._pose_client = RobotPoseClient(node, service_name=config['pose_service'])
        self._precheck = MotionPrecheck(node, config)
        self.last_grip = None

    def _on_gripper_state(self, msg):
        if msg.position:
            self._gripper_joint_angle = msg.position[0]

    def preflight(self, execute):
        missing = []
        if not Path(self._config['handeye_path']).is_file():
            missing.append('HANDEYE_MISSING')
        if execute:
            # MoveL controller rejection is not an IK/collision *pre-check*. The benchmark must
            # not discover reachability or a collision by first commanding the physical robot.
            # motion_precheck.MotionPrecheck now provides a real one (Doosan Ikin + a
            # hand-rolled sphere-vs-AABB collision check, not MoveIt — see that module's
            # docstring for why and for its known coverage limits). Collision geometry
            # (obstacles/gripper_radius_m) is optional here by explicit operator decision
            # (2026-09-04, direct human supervision at the robot) — if unset, MotionPrecheck
            # still runs IK-only and logs a WARN every call; it does not silently claim safety.
            if not self._precheck.available():
                missing.append('IK_COLLISION_VALIDATION_INTERFACE_UNAVAILABLE')
            for name, client in (('MOVEJ', self._movej_client), ('MOVEL', self._movel_client)):
                if not client.wait_for_server(timeout_sec=3.0):
                    missing.append(f'{name}_ACTION_UNAVAILABLE')
            for name, client in (('GRIPPER_COMMAND', self._gripper_cmd_client),
                                  ('GRIPPER_POSE', self._gripper_pose_client)):
                if not client.wait_for_service(timeout_sec=3.0):
                    missing.append(f'{name}_SERVICE_UNAVAILABLE')
            if self._current_posx() is None:
                missing.append('ROBOT_POSE_UNAVAILABLE')
        if execute and missing:
            return False, 'DRY_RUN_ONLY:' + ','.join(missing)
        return not missing, ','.join(missing)

    def _current_posx(self):
        # RobotPoseClient는 내부 타이머로 주기 요청한다 — 첫 응답이 올 때까지 잠깐 spin해야 한다.
        # 2.0s는 이 노드의 첫 트라이얼(카메라 구독 생성 직후, YOLO 로딩으로 한동안 spin이 없다가
        # 처음 spin되는 시점)에 한 번 ROBOT_POSE_UNAVAILABLE로 걸린 적이 있어 5.0s로 늘렸다 —
        # posx() 자체의 max_age_s=1.0(신선도 기준)은 그대로라 오래된 값을 쓰게 되는 건 아니고,
        # 신선한 값을 얻을 때까지 재시도할 여유만 늘린 것.
        end = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < end:
            posx = self._pose_client.posx(max_age_s=1.0)
            if posx is not None:
                return posx
            rclpy.spin_once(self._node, timeout_sec=0.1)
        return None

    def transform_and_validate(self, candidate):
        # 모델 결과는 카메라 좌표계 미터. 로봇 자세로 변환하려면 최신 TCP 자세가 반드시 있어야 한다.
        if any(candidate.get(key) is None for key in ('x_m', 'y_m', 'z_m')):
            return None, False, 'CANDIDATE_POSITION_MISSING'
        if not Path(self._config['handeye_path']).is_file():
            return None, False, 'HANDEYE_MISSING'
        posx = self._current_posx()
        if posx is None:
            return None, False, 'ROBOT_POSE_UNAVAILABLE'

        gripper2camera = geometry.load_handeye(Path(self._config['handeye_path']))
        base2gripper = geometry.posx_to_matrix(posx)
        point_cam_mm = (candidate['x_m'] * 1000.0, candidate['y_m'] * 1000.0, candidate['z_m'] * 1000.0)
        x_mm, y_mm, z_mm = geometry.base_from_camera(point_cam_mm, base2gripper, gripper2camera)

        workspace = self._config['workspace_mm']
        in_bounds = (workspace['x'][0] <= x_mm <= workspace['x'][1]
                     and workspace['y'][0] <= y_mm <= workspace['y'][1]
                     and workspace['z'][0] <= z_mm <= workspace['z'][1])
        if not in_bounds:
            return None, False, 'WORKSPACE_OUT_OF_BOUNDS'

        rx, ry, rz = geometry.matrix_to_zyz_deg(_TOPDOWN_ROTATION)
        return {'target_pos': [x_mm, y_mm, z_mm, rx, ry, rz]}, True, ''

    def validate_width(self, model_result):
        """Only execute candidates whose RG2 width was actually validated by the model."""
        if model_result.get('width_validation') != 'VALID':
            return False, 'RG2_WIDTH_UNAVAILABLE'
        width_m = model_result.get('width_m')
        if width_m is None:
            return False, 'RG2_WIDTH_UNAVAILABLE'
        width_mm = float(width_m) * 1000.0
        if not self._config['min_grip_width_mm'] <= width_mm <= self._config['max_grip_width_mm']:
            return False, 'RG2_WIDTH_OUT_OF_RANGE'
        return True, ''

    def execute_pick(self, robot_pose):
        self.last_grip = None
        target = list(robot_pose['target_pos'])
        approach = list(target)
        approach[2] += float(self._config['approach_height_mm'])

        # Pre-plan approach -> grasp -> lift (lift reuses the approach pose, same as the
        # real motion below) BEFORE commanding anything real. Any failure here blocks motion
        # entirely — nothing past this point runs unless all three legs check out.
        current = self._current_posx()
        if current is None:
            return False, 'ROBOT_POSE_UNAVAILABLE'
        ok, stage, code = self._precheck.check_pick_path(current, approach, target, approach)
        if not ok:
            return False, f'{code}:{stage}' if stage else code

        vel = (self._config['linear_vel_mm_s'], self._config['rot_vel_deg_s'])
        acc = (self._config['linear_acc_mm_s2'], self._config['rot_acc_deg_s2'])

        if not self._gripper_open():
            return False, 'RG2_OPEN_FAILED'
        ok, reason = self._move_linear(approach, vel, acc)
        if not ok:
            return False, f'MOVE_APPROACH_FAILED:{reason}'
        ok, reason = self._move_linear(target, vel, acc)
        if not ok:
            return False, f'MOVE_DESCEND_FAILED:{reason}'
        grip, code = self._gripper_close()
        self.last_grip = grip
        if grip is None:
            return False, code
        ok, reason = self._move_linear(approach, vel, acc)
        if not ok:
            return False, f'MOVE_LIFT_FAILED:{reason}'
        return True, ''

    def return_home(self):
        """쥔 물체를 작업영역에서 떨어뜨리지 않고 Home으로 복귀한다.

        물체를 여는 일은 사람이 Home 위치에서 안전하게 꺼낸 뒤 다음 trial을 준비하는
        `reset_mode=manual`의 책임이다. 실패한 이동 뒤에도 호출자가 이 메서드를 시도한다.
        """
        return self._move_home()

    def _move_linear(self, target_pos, vel, acc):
        """반환: (ok, reason). reason은 실패했을 때만 의미 있다 — 'ACTION_GOAL_TIMEOUT' 등
        action_client.call_blocking의 error_code, 또는 액션은 완료됐지만
        result.success=False였다는 뜻의 'CONTROLLER_REPORTED_FAILURE'."""
        goal = MovelH2r.Goal()
        goal.target_pos = [float(v) for v in target_pos]
        goal.target_vel = [float(vel[0]), float(vel[1])]
        goal.target_acc = [float(acc[0]), float(acc[1])]
        ok, result, error_code = action_client.call_blocking(self._node, self._movel_client, goal)
        if not ok:
            return False, error_code
        return bool(getattr(result, 'success', False)), 'CONTROLLER_REPORTED_FAILURE'

    def _move_home(self):
        home_deg = [float(v) for v in self._config['home_pose_deg']]
        goal = MovejH2r.Goal()
        goal.target_pos = home_deg
        goal.target_vel = [float(self._config['home_vel_deg_s'])] * 6
        goal.target_acc = [float(self._config['home_acc_deg_s2'])] * 6
        ok, result, _ = action_client.call_blocking(self._node, self._movej_client, goal)
        return ok and bool(getattr(result, 'success', False))

    def _gripper_open(self):
        # 특정 개폭(max_grip_width_mm)까지 여는 것이므로 정수 문자열(0.1mm 단위) 명령을 쓴다
        # (control/place_server.py가 gripper_open_m에 쓰는 것과 같은 경로).
        command = str(int(round(self._config['max_grip_width_mm'] * 10.0)))
        if not self._send_gripper_command(command):
            return False
        return self._wait_gripper_settled() is not None

    def _gripper_close(self):
        # 힘 제한을 이 경로로 정확히 지정할 수 없어(모듈 docstring 참조) 서버 기본값(보수적인
        # 축)으로 완전히 닫는 'c' 단축 명령을 쓴다 — control/pick_server.py와 같은 이유·같은 값.
        if not self._send_gripper_command('c'):
            return None, 'RG2_CLOSE_COMMAND_FAILED'
        final_angle = self._wait_gripper_settled()
        if final_angle is None:
            return None, 'RG2_CLOSE_TIMEOUT'
        width_mm = self._gripper_width_mm(final_angle)
        gripped = width_mm > self._config['min_grip_width_mm']
        return {'width_mm': width_mm, 'grip_state': 'gripped' if gripped else 'empty'}, ''

    def _send_gripper_command(self, command):
        request = SetCommand.Request()
        request.command = command
        ok, _, _ = action_client.call_service_blocking(self._node, self._gripper_cmd_client, request, timeout_s=3.0)
        return ok

    def _wait_gripper_settled(self, timeout_s=8.0, poll_interval_s=0.2, stable_polls=3):
        # /onrobot/sendCommand는 완료 신호가 없는 fire-and-forget이라 관절각이 더 안 바뀔
        # 때까지 /onrobot_joint_states를 직접 폴링한다 (control/dsr_motion.py의
        # wait_gripper_settled와 같은 방식 — spin_once로 구독 콜백을 처리해야 하는 점만 다르다).
        deadline = time.monotonic() + timeout_s
        last, stable = None, 0
        while rclpy.ok() and time.monotonic() < deadline:
            current = self._gripper_joint_angle
            if current is not None and last is not None and abs(current - last) < 1e-3:
                stable += 1
                if stable >= stable_polls:
                    return current
            else:
                stable = 0
            last = current
            rclpy.spin_once(self._node, timeout_sec=poll_interval_s)
        return None

    def _gripper_width_mm(self, joint_angle, timeout_s=1.5):
        # 관절각(rad) → 개폭(mm)의 기하 상수를 여기서 하드코딩하지 않는다 — /onrobot/pose가
        # 서버 자신의 그리퍼 타입(RG2/RG6)에 맞는 변환을 해 준다.
        request = GripperPose.Request()
        request.known.theta = float(joint_angle)
        ok, result, _ = action_client.call_service_blocking(self._node, self._gripper_pose_client, request, timeout_s=timeout_s)
        return float(result.pose.x) * 1000.0 if ok and result is not None else 0.0
