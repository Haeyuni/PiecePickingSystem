"""IK/충돌 사전검증 (MoveIt2 미사용, Doosan API 직접 호출).

grasp_test가 실제 M0609/RG2 실행 인터페이스(MovelH2r/MovejH2r/RG2)를 부르기 전,
approach -> grasp -> lift 세 구간을 사전 검증한다:

  - IK 실패: 각 목적지(approach/grasp/lift)를 `/dsr01/dsr_controller2/motion/ikin`
    (dsr_msgs2/srv/Ikin)으로 조회해 도달 가능성을 확인한다. 이 서비스는 순수 계산이라
    로봇을 움직이지 않는다. **`Ikin.success`만 믿지 않는다** — 2026-09-04 실측 확인 결과
    `success=True`인데도 `conv_posj`가 관절 한계를 훨씬 벗어난 값(수백~수천 도)을 반환하는
    경우가 있었다(원점 근처 등 특이/퇴화 pose에서 재현). 그래서 응답을 받으면 M0609 실제
    관절 한계(cobot2_ws의 dsr_moveit_config_m0609/config/joint_limits.yaml, 라디안 →
    JOINT_LIMITS_DEG로 변환해 둠)로 직접 재검증한다. 하나라도 벗어나면 실패로 취급한다.
  - 충돌 예상: Doosan API에는 기하학적 충돌예측 서비스가 없다 — 있는 건
    ChangeCollisionSensitivity(실제 동작 중 토크 기반 반응형 정지)뿐이라 사전 예측이
    아니다. 그래서 이 파일이 직접 구현한다: TCP를 반지름 gripper_radius_m인 구로
    근사해, 각 구간(현재->approach, approach->grasp, grasp->lift)을 path_check_step_m
    간격으로 선형보간하며 등록된 obstacle(table/box/jig, robot base frame, meter, AABB)과
    교차하는지 검사한다. 교차하면 COLLISION_EXPECTED.

한계 (거짓 안전 신호를 주지 않기 위해 그대로 남긴다):
  - 팔/팔꿈치/포어암 등 몸체 링크는 검사하지 않는다. 오직 TCP 근방(반지름
    gripper_radius_m)만 본다 — URDF 기반 전신 충돌검사가 아니다.
  - RG2+카메라 브래킷의 실제 형상을 반영하지 않는다. gripper_radius_m는 사람이 실측해
    보수적으로 넣어야 한다(기본값 없음).
  - obstacles/gripper_radius_m이 비어있으면 충돌 검사는 **실행되지 않는다** — IK만 검사한다.
    2026-09-04, 로봇 옆에서 직접 감독하겠다는 운영자 확인을 받고 `ready()`가 충돌 형상 없이도
    True를 반환하도록 완화했다(이전엔 `geometry_configured()`도 요구해 무조건 막았음). 이
    상태에서 `check_pick_path()`를 호출하면 매번 ROS 로거에 WARN으로 남긴다 — 콘솔에 안 보일
    수는 있어도 조용히 넘어가지는 않는다.
"""
import math

from dsr_msgs2.srv import GetCurrentSolutionSpace, Ikin

from . import action_client

IKIN_SERVICE = '/dsr01/dsr_controller2/motion/ikin'
SOLUTION_SPACE_SERVICE = '/dsr01/dsr_controller2/aux_control/get_current_solution_space'

# Real M0609 joint limits, converted from radians to degrees (Ikin's conv_posj is in degrees).
# Source: ~/cobot2_ws/install/dsr_moveit_config_m0609/share/dsr_moveit_config_m0609/config/
# joint_limits.yaml, verified 2026-09-04 (min_position/max_position per joint_1..joint_6).
_JOINT_LIMITS_RAD = ((-3.14, 3.14), (-1.6581, 1.6581), (-2.1817, 2.1817),
                     (-3.14, 3.14), (-2.3562, 2.3562), (-3.14, 3.14))
JOINT_LIMITS_DEG = tuple((math.degrees(lo), math.degrees(hi)) for lo, hi in _JOINT_LIMITS_RAD)


def _within_joint_limits(posj_deg):
    if len(posj_deg) != len(JOINT_LIMITS_DEG):
        return False
    return all(lo <= value <= hi for value, (lo, hi) in zip(posj_deg, JOINT_LIMITS_DEG))


def _aabb_from_config(obstacle):
    cx, cy, cz = (float(v) for v in obstacle['center_m'])
    length, width, height = (float(v) for v in obstacle['size_m'])
    return {
        'name': str(obstacle.get('name', '?')),
        'min': (cx - length / 2.0, cy - width / 2.0, cz - height / 2.0),
        'max': (cx + length / 2.0, cy + width / 2.0, cz + height / 2.0),
    }


def _sphere_hits_aabb(point_m, radius_m, box):
    clamped = tuple(min(max(point_m[i], box['min'][i]), box['max'][i]) for i in range(3))
    dist_sq = sum((point_m[i] - clamped[i]) ** 2 for i in range(3))
    return dist_sq <= radius_m ** 2


class MotionPrecheck:
    def __init__(self, node, config):
        self._node = node
        self._ikin_client = node.create_client(Ikin, IKIN_SERVICE)
        self._solspace_client = node.create_client(GetCurrentSolutionSpace, SOLUTION_SPACE_SERVICE)
        self._obstacles = [_aabb_from_config(o) for o in (config.get('obstacles') or [])]
        radius = config.get('gripper_radius_m')
        self._gripper_radius_m = float(radius) if radius is not None else None
        self._step_m = float(config.get('path_check_step_m', 0.01))

    def available(self):
        """Ikin 서비스가 지금 응답 가능한지 (인터페이스 자체 유무)."""
        return self._ikin_client.service_is_ready()

    def geometry_configured(self):
        """table/box/jig + gripper 반경이 실측으로 채워졌는지 (MANUAL_REQUIRED 항목)."""
        return bool(self._obstacles) and self._gripper_radius_m is not None

    def ready(self):
        # Collision geometry is no longer required to unblock execute:=true (operator-confirmed
        # 2026-09-04: IK-only, with direct human supervision + E-stop at the robot). See module
        # docstring. IK availability is still mandatory — that check never gets waived.
        return self.available()

    def _current_sol_space(self):
        ok, result, _ = action_client.call_service_blocking(
            self._node, self._solspace_client, GetCurrentSolutionSpace.Request(), timeout_s=3.0)
        return result.sol_space if ok and result is not None and result.success else 0

    def _ik_ok(self, pos_mm_zyz):
        request = Ikin.Request()
        request.pos = [float(v) for v in pos_mm_zyz[:6]]
        request.sol_space = self._current_sol_space()
        request.ref = 0  # DR_BASE
        ok, result, _ = action_client.call_service_blocking(self._node, self._ikin_client, request, timeout_s=3.0)
        if not (ok and result is not None and bool(result.success)):
            return False
        # Ikin.success alone is not trustworthy (see module docstring) — re-check against the
        # robot's real joint limits before trusting the solution.
        return _within_joint_limits(list(result.conv_posj))

    def _segment_collision(self, start_mm, end_mm):
        """반환: (hit, obstacle_name). obstacles/반경 미설정이면 (False, None)로 조용히
        통과시키지 않고, 호출자(check_pick_path)가 geometry_configured()로 먼저 걸러낸다."""
        start_m = tuple(v / 1000.0 for v in start_mm[:3])
        end_m = tuple(v / 1000.0 for v in end_mm[:3])
        length_m = sum((end_m[i] - start_m[i]) ** 2 for i in range(3)) ** 0.5
        steps = max(1, int(length_m / self._step_m))
        for step in range(steps + 1):
            t = step / steps
            point = tuple(start_m[i] + (end_m[i] - start_m[i]) * t for i in range(3))
            for box in self._obstacles:
                if _sphere_hits_aabb(point, self._gripper_radius_m, box):
                    return True, box['name']
        return False, None

    def check_pick_path(self, current_pos_mm, approach_pos_mm, grasp_pos_mm, lift_pos_mm):
        """approach -> grasp -> lift 세 구간을 사전 plan한다.

        반환: (ok, stage, code). ok=False면 code는 'IK_FAILED' 또는 'COLLISION_EXPECTED'
        (COLLISION_EXPECTED일 때 stage에 '<단계>:<obstacle 이름>'을 담는다).
        geometry_configured()가 False면 충돌 검사 자체를 건너뛴다(IK만 본다) — 매번 WARN
        로그를 남겨 "검사 안 함"이 "충돌 없음"으로 오인되지 않게 한다.
        """
        collision_check_enabled = self.geometry_configured()
        if not collision_check_enabled:
            self._node.get_logger().warning(
                'MotionPrecheck: obstacles/gripper_radius_m 미설정 — IK만 검사하고 충돌 사전검증은'
                ' 건너뜁니다. 사람이 직접 감독해야 합니다.')
        stages = (
            ('approach', current_pos_mm, approach_pos_mm),
            ('grasp', approach_pos_mm, grasp_pos_mm),
            ('lift', grasp_pos_mm, lift_pos_mm),
        )
        for name, start, end in stages:
            if not self._ik_ok(end):
                return False, name, 'IK_FAILED'
            if collision_check_enabled:
                hit, obstacle_name = self._segment_collision(start, end)
                if hit:
                    return False, f'{name}:{obstacle_name}', 'COLLISION_EXPECTED'
        return True, '', ''
