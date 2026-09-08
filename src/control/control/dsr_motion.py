"""dsr_msgs2 액션 호출 공용 헬퍼: 블로킹 대기 + 취소 전파, pose(mm, 쿼터니언) → posx 변환,
OnRobot RG2 그리퍼 제어.

home_server.py가 MovejH2r을 부르며 만든 패턴(ActionClient를 threading.Event로 동기식
대기하되, 콜백은 MultiThreadedExecutor의 다른 스레드가 처리)을 pick_server.py/
place_server.py의 MovelH2r 호출도 그대로 필요로 해서 여기로 뺐다. 세 서버 모두
ReentrantCallbackGroup을 액션 서버와 이 모듈이 만드는 ActionClient에 공유해야
데드락 없이 동작한다 — 호출부에서 그 그룹을 넘겨준다.

**그리퍼는 표준 액션(`/rg6_controller`, control_msgs/GripperCommand)을 쓰지 않는다.**
실물로 확인해 보니 이 액션의 `execute_callback`(onrobot_rg_control 패키지)이 두 가지
문제가 있다 — (1) `position`(미터, 개폭)을 관절각(rad)으로 변환 없이 그대로 넘긴다
(같은 파일의 서비스 핸들러 `sendCommandCallback`은 `widthToJointValue()`로 제대로
변환한다), (2) 첫 목표 처리 시 내부 상태가 아직 `None`인 채로 비교 연산을 해 예외로
죽고 rclpy가 빈 결과로 ABORTED 처리한다. 두 번 다 실물로 재현했다. 대신 같은 드라이버가
제공하는 `/onrobot/sendCommand`(문자 명령, 그 서버가 자기 그리퍼 타입에 맞는 변환을
알아서 한다)를 쓴다 — 단, 이 서비스는 명령만 보내고 완료를 기다리지 않으므로
`/onrobot_joint_states`를 직접 폴링해 안정될 때까지 기다린다.

**연속된 movel은 회전을 매번 다시 계산하지 않고 방금 도달한 실제 자세에서 이어간다.**
그리퍼가 위에서 수직으로 접근할 때 `ry`가 180° 근처(ZYZ 오일러각의 특이점)라, 우리가
계산한 (rx,rz)와 컨트롤러가 실제로 도달한 (rx,rz)가 **물리적으로는 같은 방향인데
파라미터값이 다를 수 있다**(실물로 확인: `rx=-39.8`을 보냈는데 로봇은 `rx=35.2,
rz=-91.1`로 도달했다 — 둘 다 유효한 표현이다). 이 상태에서 원래 계산값(rx=-39.8)을
그대로 다시 보내면 컨트롤러가 그걸 큰 회전으로 오인해 응답 없이 멈추는 것을 실물로
재현했다. 그래서 pick_server.py/place_server.py는 첫 이동(접근)에만 계산된 회전을
쓰고, 그다음 이동들(하강·들어올리기/삽입·물러나기)은 `get_current_posx`로 방금
도달한 회전을 읽어 그대로 유지한 채 위치만 바꾼다.
"""
import math
import threading
import time

from perception_common.geometry import matrix_to_zyz_deg, posx_to_matrix, quaternion_to_matrix


def _rotation_angle_diff_deg(a_zyz: list[float], b_zyz: list[float]) -> float:
    """ZYZ 오일러각(도) 두 세트가 물리적으로 실제 몇 도 떨어져 있는지.

    ry가 180도 근처면 서로 다른 (rx,rz) 조합이 같은 방향을 나타낼 수 있다(모듈 docstring
    참조) — 그래서 성분별로 `abs(rx_a-rx_b)`를 더하는 식으로는 비교할 수 없다(같은 방향인데
    rx만 180도 넘게 차이 나는 걸 "회전 많이 남음"으로 오판한다). 회전행렬로 바꿔 실제
    각도차(내적 기반)를 잰다 — `posx_to_matrix`가 이미 이 ZYZ 관례로 행렬을 만든다."""
    ra = posx_to_matrix([0.0, 0.0, 0.0, *a_zyz])[:3, :3]
    rb = posx_to_matrix([0.0, 0.0, 0.0, *b_zyz])[:3, :3]
    trace = float((ra.T @ rb).trace())
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos_angle))

def rotation_diff_deg(a_zyz, b_zyz) -> float:
    """두 ZYZ 자세가 실제로 몇 도 떨어져 있는지. 로그용 공개 이름."""
    return _rotation_angle_diff_deg(list(a_zyz), list(b_zyz))


MOVEJ_ACTION = "/dsr01/motion/movej_h2r"
MOVEL_ACTION = "/dsr01/motion/movel_h2r"
GET_CURRENT_POSX_SERVICE = "/dsr01/dsr_controller2/aux_control/get_current_posx"
GRIPPER_COMMAND_SERVICE = "/onrobot/sendCommand"
GRIPPER_JOINT_STATES_TOPIC = "/onrobot_joint_states"
# RG2 컨트롤러 상태 비트필드(gsta 포함). onrobot_rg_control 드라이버가 발행한다 —
# 2026-09-07에 추가한 토픽이라, 드라이버를 그 이후로 빌드/재기동하지 않았으면 없다.
GRIPPER_STATUS_TOPIC = "/onrobot/status"
# 컨트롤러 알람(OnLogAlarm)이 그대로 실려 나오는 토픽. dsr_controller2.cpp의 OnLogAlarm이
# LOG_ALARM을 RobotError로 옮겨 발행한다 — `code`가 알람 index고 `msg1`이 알람 본문이다.
MOTION_ERROR_TOPIC = "/dsr01/error"
IKIN_SERVICE = "/dsr01/dsr_controller2/motion/ikin"
FKIN_SERVICE = "/dsr01/dsr_controller2/motion/fkin"
GET_CURRENT_SOLUTION_SPACE_SERVICE = (
    "/dsr01/dsr_controller2/aux_control/get_current_solution_space")
# safety_monitor.py / compliance.py가 쓴다. 이 저장소엔 dsr_msgs2가 마운트로만 존재해(위
# GET_CURRENT_POSX_SERVICE 등과 같은 사정) 서비스 이름·필드는 doosan-robot2 GitHub
# 소스(dsr_controller2.cpp의 svc_prefix_ 등록부, jazzy 브랜치, 2026-09-08 확인)로만
# 검증했다 — 실물로는 확인하지 못했다.
GET_ROBOT_STATE_SERVICE = "/dsr01/dsr_controller2/system/get_robot_state"
GET_EXTERNAL_TORQUE_SERVICE = "/dsr01/dsr_controller2/aux_control/get_external_torque"
GET_TOOL_FORCE_SERVICE = "/dsr01/dsr_controller2/aux_control/get_tool_force"

# M0609 관절 한계(도). dsr_description2/urdf/m0609.urdf의 <limit lower/upper>를 도로 옮긴 값
# (±6.2832 rad = ±360도, J3만 ±2.618 rad = ±150도). 컨트롤러의 소프트 리밋이 더 좁게
# 설정돼 있을 수 있으므로 skill_params.yaml에서 좁힐 수 있게 해 둔다.
JOINT_LIMITS_DEG = (360.0, 360.0, 150.0, 360.0, 360.0, 360.0)

# ikin이 돌려준 관절해를 fkin으로 되돌렸을 때 원래 목표와 이만큼 안쪽이면 "해가 맞다"로 본다.
# 실측(2026-09-07): 정상 해의 왕복 오차는 0.005mm / 0.001도 수준이고, 해가 없을 때는
# 수백 mm 단위로 벌어진다 — 두 경우 사이에 넓은 간격이 있어 경계값 선택이 민감하지 않다.
IK_ROUND_TRIP_TOL_MM = 1.0
IK_ROUND_TRIP_TOL_DEG = 1.0

IK_OK = "ok"                 # 해가 있고 관절 한계 안
IK_UNREACHABLE = "unreachable"   # 해가 없다 (왕복 검증 실패)
IK_JOINT_LIMIT = "joint_limit"   # 해는 있는데 관절 한계를 넘는다
IK_UNKNOWN = "unknown"           # 서비스 무응답 — 판단하지 않았다


class IkVerdict:
    """`verify_ik`의 결과. `status`는 IK_* 중 하나.

    **판단 불가(IK_UNKNOWN)를 "가능"으로 뭉개면 안 된다.** 호출부가 그 경우를 구분해야
    "확인하고 통과시켰다"와 "확인을 못 했다"를 로그에서 나눌 수 있다.
    """

    __slots__ = ("status", "posj", "joint_margin_deg", "round_trip_mm", "round_trip_deg")

    def __init__(self, status, posj=None, joint_margin_deg=None,
                 round_trip_mm=None, round_trip_deg=None):
        self.status = status
        self.posj = posj
        self.joint_margin_deg = joint_margin_deg
        self.round_trip_mm = round_trip_mm
        self.round_trip_deg = round_trip_deg

    @property
    def ok(self) -> bool:
        return self.status == IK_OK

    @property
    def known(self) -> bool:
        return self.status != IK_UNKNOWN

    def __repr__(self) -> str:
        margin = ("%.1f" % self.joint_margin_deg) if self.joint_margin_deg is not None else "?"
        return f"IkVerdict({self.status}, margin={margin}deg)"


def _call_service(client, request, timeout_s: float):
    """서비스를 한 번 부르고 결과를 돌려준다. 무응답이면 None.

    `rclpy.spin_until_future_complete`를 쓰지 않는 이유는 이 모듈의 다른 호출들과 같다 —
    노드는 MultiThreadedExecutor가 이미 돌리고 있어서, 여기서는 완료 이벤트만 기다린다.
    """
    if not client.service_is_ready():
        return None
    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _f: done.set())
    if not done.wait(timeout=timeout_s):
        return None
    return future.result()


def joint_margin_deg(posj, limits=JOINT_LIMITS_DEG) -> float:
    """관절해가 한계까지 남긴 여유(도) 중 **가장 적은 것**. 음수면 이미 한계를 넘었다."""
    return min(float(limit) - abs(float(q)) for q, limit in zip(posj, limits))


def current_solution_space(client, default: int = 2, timeout_s: float = 2.0) -> int:
    """지금 로봇이 있는 solution space. 못 읽으면 `default`.

    ikin은 solution space마다 다른 해를 낸다. movel은 현재 space를 유지하므로,
    "이 자세로 **지금 상태에서** 갈 수 있는가"를 물으려면 현재 space로 물어야 한다.
    """
    from dsr_msgs2.srv import GetCurrentSolutionSpace

    result = _call_service(client, GetCurrentSolutionSpace.Request(), timeout_s)
    if result is None or not result.success:
        return int(default)
    return int(result.sol_space)


# GetRobotState.srv의 robot_state 값(dsr_msgs2, jazzy 브랜치) 중 safety_monitor.py가
# 쓰는 두 값만 옮긴다. 나머지(SAFE_OFF류 등)는 문서 버전마다 번호가 어긋나 있어(예:
# msg 주석은 STATE_SAFE_STOP2=10, srv 주석은 9) 여기서는 두 문서가 일치하는 값만 쓴다.
ROBOT_STATE_SAFE_STOP = 5
ROBOT_STATE_EMERGENCY_STOP = 6


def get_robot_state(client, timeout_s: float = 1.0) -> int | None:
    """`system/get_robot_state`로 컨트롤러의 robot_state 코드를 읽는다. 무응답/실패면 None."""
    from dsr_msgs2.srv import GetRobotState

    result = _call_service(client, GetRobotState.Request(), timeout_s)
    if result is None or not result.success:
        return None
    return int(result.robot_state)


def get_external_torque(client, timeout_s: float = 1.0) -> list[float] | None:
    """`aux_control/get_external_torque`로 외부 관절토크(Nm, 6축)를 읽는다. 무응답/실패면 None.

    실제 로봇에 진짜 힘이 걸렸는지(순응제어)가 아니라, 지금 흐르는 값을 그대로 옮긴다 —
    임계값 판정은 compliance.py가 한다.
    """
    from dsr_msgs2.srv import GetExternalTorque

    result = _call_service(client, GetExternalTorque.Request(), timeout_s)
    if result is None or not result.success:
        return None
    return [float(v) for v in result.ext_torque[:6]]


def get_tool_force(client, ref: int = 0, timeout_s: float = 1.0) -> list[float] | None:
    """`aux_control/get_tool_force`로 툴에 걸린 힘·모멘트(N, Nm, 6축 = [Fx,Fy,Fz,Mx,My,Mz])를
    읽는다. `ref`: DR_BASE(0, 기본)/DR_TOOL(1)/DR_WORLD(2). 무응답/실패면 None.

    place_server.py의 순응 하강(compliance.py 참조)이 여기 앞 3축(힘, N)을 쓴다 —
    skill_params.yaml의 profiles.<profile>.contact_threshold_n이 같은 단위(N)다.
    """
    from dsr_msgs2.srv import GetToolForce

    request = GetToolForce.Request()
    request.ref = int(ref)
    result = _call_service(client, request, timeout_s)
    if result is None or not result.success:
        return None
    return [float(v) for v in result.tool_force[:6]]

ARM_JOINT_STATES_TOPIC = "/dsr01/joint_states"
# `/dsr01/joint_states`가 쓰는 이름. 순서를 가정하지 않고 이 이름으로 골라낸다.
ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


def posx_via_fkin(fkin_client, joint_deg, timeout_s: float = 2.0) -> list[float] | None:
    """관절각(도) → TCP posx. `get_current_posx`가 무응답일 때의 **대체 경로**다.

    **왜 이게 필요한가.** movel 직후 aux_control(`get_current_posx`)이 10~20초 무응답인
    구간이 있는데(get_current_posx 주석), pick은 그 구간에서 시작한다. 자세를 못 읽으면
    `tcp_posx_for_grasp`의 뒤집기 보정이 통째로 꺼져서(current_zyz_deg=None) 같은 파지를
    172도 헛돌아 가는 일이 다시 생긴다 — 2026-09-07에 고쳤던 그 문제가 **재시도 때만**
    되살아나는 셈이다.

    `motion/fkin`은 aux_control과 다른 네임스페이스라 그 무응답 구간에도 답한다
    (후보 검사의 verify_ik가 바로 그때 ikin/fkin을 정상적으로 쓰고 있다).
    2026-09-08 실측: 같은 순간 fkin과 get_current_posx의 차이가 0.0006mm / 0.0006도였다 —
    TCP 설정도 그대로 반영된다.

    **solution space는 안 돌려준다.** posx의 7번째 값이 필요한 곳은 이 폴백을 쓰지 않는다
    (필요하면 current_solution_space를 따로 부른다).
    """
    from dsr_msgs2.srv import Fkin

    if joint_deg is None or len(joint_deg) < 6:
        return None
    request = Fkin.Request()
    request.pos = [float(v) for v in joint_deg[:6]]
    request.ref = 0  # DR_BASE
    result = _call_service(fkin_client, request, timeout_s)
    if result is None or not result.success:
        return None
    posx = [float(v) for v in result.conv_posx[:6]]
    return posx if all(math.isfinite(v) for v in posx) else None


def verify_ik(ikin_client, fkin_client, posx, sol_space: int = 2,
              limits=JOINT_LIMITS_DEG, timeout_s: float = 2.0) -> IkVerdict:
    """`posx`에 실제로 갈 수 있는지 ikin으로 확인한다.

    **`ikin`의 `success`는 아무 의미가 없다.** dsr_controller2.cpp의 `ikin_cb`는 해가
    있든 없든 마지막 줄에서 `res->success = true`를 넣는다(소스로 확인). 2026-09-07 실측:
    도달 불가능한 목표(x=2000)에 `success=True`, `conv_posj=[-7203, -2186, -7732, ...]`도.
    그래서 두 가지를 직접 본다.

    1. **왕복 검증** — fkin(ikin(posx))이 원래 posx로 돌아오는가. 해가 없을 때 나오는
       쓰레기 관절값은 전혀 다른 자세로 되돌아온다(위 예: (80, -3.9, -138)). 회전은
       ZYZ 파라미터가 아니라 회전행렬 각도차로 비교한다 — ry가 180도 근처면 같은 방향이
       다른 (rx,rz)로 표현될 수 있어서다(이 모듈 docstring 참조).
    2. **관절 한계** — 왕복이 맞아도 한계를 넘는 해일 수 있다.

    서비스가 준비 안 됐거나 무응답이면 IK_UNKNOWN이다. 호출부는 이것을 "불가"로 다루면
    안 된다 — ikin이 죽어 있다는 이유로 모든 후보를 버리면 pick이 통째로 멈춘다.
    """
    from dsr_msgs2.srv import Fkin, Ikin

    request = Ikin.Request()
    request.pos = [float(v) for v in posx[:6]]
    request.sol_space = int(sol_space)
    request.ref = 0  # DR_BASE
    result = _call_service(ikin_client, request, timeout_s)
    if result is None:
        return IkVerdict(IK_UNKNOWN)
    posj = [float(q) for q in result.conv_posj]
    if not all(math.isfinite(q) for q in posj):
        return IkVerdict(IK_UNREACHABLE, posj=posj)

    back = Fkin.Request()
    back.pos = posj
    back.ref = 0
    forward = _call_service(fkin_client, back, timeout_s)
    if forward is None:
        # ikin은 답했는데 fkin이 무응답 — 왕복 검증을 못 했다. 관절 한계만으로 판정하면
        # "쓰레기 해가 우연히 한계 안"인 경우를 놓치므로 모른다고 답한다.
        return IkVerdict(IK_UNKNOWN, posj=posj, joint_margin_deg=joint_margin_deg(posj, limits))
    reached = [float(v) for v in forward.conv_posx]
    error_mm = math.sqrt(sum((reached[i] - float(posx[i])) ** 2 for i in range(3)))
    error_deg = rotation_diff_deg(reached[3:6], [float(v) for v in posx[3:6]])
    margin = joint_margin_deg(posj, limits)
    if error_mm > IK_ROUND_TRIP_TOL_MM or error_deg > IK_ROUND_TRIP_TOL_DEG:
        return IkVerdict(IK_UNREACHABLE, posj=posj, joint_margin_deg=margin,
                         round_trip_mm=error_mm, round_trip_deg=error_deg)
    if margin < 0.0:
        return IkVerdict(IK_JOINT_LIMIT, posj=posj, joint_margin_deg=margin,
                         round_trip_mm=error_mm, round_trip_deg=error_deg)
    return IkVerdict(IK_OK, posj=posj, joint_margin_deg=margin,
                     round_trip_mm=error_mm, round_trip_deg=error_deg)




def get_current_posx(client, goal_handle, timeout_s: float = 5.0, retries: int = 6,
                     retry_delay_s: float = 3.0) -> list[float] | None:
    """get_current_posx를 호출한다 — [x,y,z,rx,ry,rz](mm, deg). 실패하면 잠깐 쉬었다
    재시도한다(기본값 기준 최대 약 `retries * (timeout_s + retry_delay_s)` ≈ 48초).

    **실물로 확인**: movel 직후 이 서비스(aux_control)가 10~20초 이상 무응답인 구간이
    있다 — 컨트롤러가 방금 끝난 모션을 정리하는 동안으로 보인다. 한 번 실패로 바로
    포기하면 정상적인 지연을 오류로 오판한다. `goal_handle`이 취소되면 재시도를
    멈추고 None을 돌려준다(오류가 아니라 취소로 처리하도록 호출부가 구분해야 한다).

    perception_common.robot_pose.RobotPoseClient(주기 캐싱, 검출 프레임마다 필요)와
    달리 여기서는 movel 사이사이 딱 한 번씩만 필요해 별도 캐시 없이 그때그때 묻는다.
    """
    from dsr_msgs2.srv import GetCurrentPosx

    for attempt in range(retries):
        if goal_handle.is_cancel_requested:
            return None
        if client.wait_for_service(timeout_sec=2.0):
            request = GetCurrentPosx.Request()
            request.ref = 0  # DR_BASE
            done = threading.Event()
            future = client.call_async(request)
            future.add_done_callback(lambda _f: done.set())
            done.wait(timeout=timeout_s)
            result = future.result()
            if result is not None and result.success and result.task_pos_info:
                data = list(result.task_pos_info[0].data)
                if len(data) >= 6:
                    return [float(v) for v in data[:6]]
        if attempt < retries - 1:
            time.sleep(retry_delay_s)
    return None


def bin_pose_to_posx(pose: dict) -> list[float]:
    """bins.yaml의 pose 딕셔너리({x,y,z,qx,qy,qz,qw}, mm) → [x,y,z,rx,ry,rz]."""
    matrix = quaternion_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    rx, ry, rz = matrix_to_zyz_deg(matrix)
    return [pose["x"], pose["y"], pose["z"], rx, ry, rz]


def pose_mm_to_posx(pose) -> list[float]:
    """geometry_msgs/Pose(위치 mm, 회전 쿼터니언) → dsr_msgs2가 쓰는 [x,y,z,rx,ry,rz]
    (mm, ZYZ 오일러 도). `perception_common.geometry.posx_to_matrix`의 역변환이라
    get_current_posx·캘리브레이션과 같은 해석을 유지한다."""
    matrix = quaternion_to_matrix(pose.orientation.x, pose.orientation.y,
                                  pose.orientation.z, pose.orientation.w)
    rx, ry, rz = matrix_to_zyz_deg(matrix)
    return [pose.position.x, pose.position.y, pose.position.z, rx, ry, rz]


# --- TCP 원점 ↔ 실제 파지점 -------------------------------------------------
#
# **로봇이 명령·보고하는 TCP 원점은 그리퍼가 실제로 무는 지점이 아니다.** 두 점의 차이는
# 툴 좌표계에서 상수(그리퍼 기구가 고정이므로)지만, base 좌표계에서는 **자세에 따라
# 방향이 돈다**. 이 상수를 어디에도 넣지 않으면 파이프라인이 "여기를 물어라"로 계산한
# 점에 TCP 원점이 가고, 손가락은 그만큼 빗나간 곳에서 닫힌다.
#
# 실측(2026-09-07, TCP posx=[266.86, 26.13, 314.11, 117.68, -179.84, 117.64]에서 손끝을
# 물체에 대고 잰 값): 툴 좌표계로 (+9.4, +24.2, **-18.3**)mm. z의 -18.3이 이 오프셋이다 —
# 손가락 접촉면이 TCP 원점보다 접근 방향으로 18.3mm **앞**에 있다는 뜻이라, TCP를 파지점에
# 보내면 손가락은 18.3mm 못 미친 허공에서 닫힌다.
#
# 이 값이 왜 예전 튜닝값들의 정체인지: 수직 파지에서는 순수한 base -Z 부족분이라
# `pick_depth_extra_mm`(한때 40)이나 `camera_frame_offset_mm.z`(한때 45)로 덮으면
# 그 자세에서는 맞아떨어졌다. 기울어진 파지에서는 그 두 축이 툴 z축과 갈라져 옆으로 샌다.
# 여기서 툴 좌표계 상수로 다루면 모든 자세에서 한 번에 맞는다.
#
# **티치펜던트의 툴 오프셋을 고쳐 없애면 안 된다.** hand-eye 캘리브레이션이
# `get_current_posx()`가 보고하는 바로 이 TCP 정의로 풀려 있고(perception_common/
# geometry.py), bins.yaml의 목적지도 이 정의로 티칭돼 있다. 펜던트에서 TCP를 옮기면
# 그 둘이 같은 양만큼 조용히 틀어진다 — 소프트웨어에서 다루는 편이 안전하고 되돌리기 쉽다.
DEFAULT_GRASP_CENTER_OFFSET_MM = (0.0, 0.0, 0.0)


def grasp_center_offset_mm(params: dict) -> list[float]:
    """skill_params.yaml의 `tool.grasp_center_offset_mm` → 툴 좌표계 [x, y, z] (mm).

    없으면 (0,0,0) — 예전과 완전히 같은 동작이라 설정을 안 만든 환경이 조용히 달라지지 않는다.
    """
    tool = (params or {}).get("tool") or {}
    offset = tool.get("grasp_center_offset_mm")
    if isinstance(offset, dict):
        return [float(offset.get("x", 0.0)), float(offset.get("y", 0.0)),
                float(offset.get("z", 0.0))]
    if isinstance(offset, (list, tuple)) and len(offset) == 3:
        return [float(v) for v in offset]
    return list(DEFAULT_GRASP_CENTER_OFFSET_MM)


def _rotation_from_pose(pose):
    q = pose.orientation
    return quaternion_to_matrix(q.x, q.y, q.z, q.w)


# 접근축(툴 Z) 둘레 180도 회전. 평행 그리퍼는 손가락 두 개가 대칭이라 닫힘축의 부호가
# 뒤집혀도 **물리적으로 완전히 같은 파지**다 — 무는 지점도, 접근 방향도, 개폭도 같다.
_FLIP_ABOUT_APPROACH = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))


def nearest_equivalent_grasp_rotation(rotation, current_zyz_deg):
    """같은 파지를 나타내는 두 자세(R, R·Rz180) 중 **현재 손목 자세에 가까운 쪽**을 고른다.

    **왜 필요한가.** 전략이 내는 닫힘축 부호는 임의다. 먼 쪽 표현을 그대로 명령하면 로봇이
    같은 파지를 위해 손목을 180도 가까이 헛돌린다 — 2026-09-07 실물: 회전오차 172.4도로
    시작해 접근에만 14초가 걸렸고, 뒤집은 표현을 썼다면 7.6도면 될 일이었다.
    회전이 크면 시간만 드는 게 아니라 경로가 크게 휘어 주변과 부딪힐 위험도 커진다.

    `current_zyz_deg`가 없으면(자세를 못 읽음) 원래 자세를 그대로 돌려준다 — 모르면
    바꾸지 않는 쪽이 안전하다.
    """
    import numpy as np

    if current_zyz_deg is None:
        return rotation
    current = posx_to_matrix([0.0, 0.0, 0.0, *current_zyz_deg[:3]])[:3, :3]
    flipped = np.asarray(rotation) @ np.asarray(_FLIP_ABOUT_APPROACH)

    def separation(candidate):
        trace = float((current.T @ np.asarray(candidate)).trace())
        return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))

    return flipped if separation(flipped) < separation(rotation) else rotation


def tcp_posx_for_grasp(pose, offset_mm, current_zyz_deg=None) -> list[float]:
    """**파지점**(손가락이 만나야 하는 지점) Pose → 거기에 손가락이 오도록 명령할 TCP posx.

    `current_zyz_deg`를 주면 같은 파지를 나타내는 두 자세 중 현재 손목에 가까운 쪽을
    고른다(nearest_equivalent_grasp_rotation) — 헛도는 180도 회전을 막는다.

    파지점 P와 자세 R이 주어졌을 때 실제 파지점은 `TCP + R @ offset`이므로, 그것이 P가
    되려면 `TCP = P - R @ offset`을 명령해야 한다. R을 함께 곱하기 때문에 수직 파지든
    기울어진 파지든 같은 식 하나로 맞는다 — base 축에 상수를 더하는 방식과 다른 점이 이것이다.
    """
    rotation = _rotation_from_pose(pose)
    if current_zyz_deg is not None:
        rotation = nearest_equivalent_grasp_rotation(rotation, current_zyz_deg)
    rx, ry, rz = matrix_to_zyz_deg(rotation)
    shift = rotation @ [float(v) for v in offset_mm]
    return [float(pose.position.x - shift[0]),
            float(pose.position.y - shift[1]),
            float(pose.position.z - shift[2]), rx, ry, rz]


def grasp_center_from_posx(posx, offset_mm) -> list[float]:
    """TCP posx → 그 자세에서 손가락이 실제로 만나는 base 좌표(mm). `tcp_posx_for_grasp`의 역방향.

    진단용이다 — "명령한 TCP"가 아니라 "손끝이 실제로 간 곳"을 로그에 남겨야 파지 오차를
    물체 좌표와 직접 견줄 수 있다.
    """
    rotation = posx_to_matrix(posx)[:3, :3]
    shift = rotation @ [float(v) for v in offset_mm]
    return [float(posx[i] + shift[i]) for i in range(3)]


def _release_remote_goal(remote_handle, finished, logger, reason: str,
                         timeout_s: float = 1.5) -> None:
    """원격 액션 goal을 확실히 끝내고 온다(취소를 보내고 종료를 기다린다).

    **결과를 안 기다리기로 했으면 반드시 여기를 지나야 한다.** dsr_controller2의
    movej_h2r/movel_h2r 액션은 goal 하나마다 detached 스레드를 띄우는데, 그 스레드는
    *취소되거나 목표에 도달할 때만* 빠져나온다. 우리가 `on_timeout_verify`로 도착을
    직접 확인하고 취소 없이 그냥 다음 이동으로 넘어가면 그 goal은 둘 중 어느 것도
    되지 않아 **스레드가 드라이버 프로세스가 죽을 때까지 100Hz로 남는다**(2026-09-06
    실물 확인: 버려진 movel goal의 스레드가 11분 뒤에도 돌고 있었다).

    남은 스레드 하나는 초당 200번씩 드라이버의 단일 DRFL 뮤텍스를 잡는다 — 그 뮤텍스를
    100Hz RT 루프(read)와 get_current_posx가 함께 쓰기 때문에, 이동을 한 번 할 때마다
    부하가 한 겹씩 영구히 쌓인다. 이게 "드라이버 재시작 후 사이클마다 규칙적으로
    느려짐"(pick 14.9→22.6→31.3초), "get_current_posx 10~20초 무응답", 컨트롤러의
    "heartbeat packet was not received for 5 seconds" 알람의 공통 원인이었다.
    더구나 남은 스레드는 자기 **옛** 목표와 현재 위치를 계속 비교하다가, 팔이 나중에
    그 근처를 지나가면 진행 중인 다른 모션 한가운데서 quick stop을 쏜다 — 단계와
    무관하게 무작위로 movel이 멈칫하던 현상이 이것이다.

    (드라이버 쪽에도 2026-09-06에 같은 문제를 막는 장치를 넣었다 — 새 H2R goal이
    들어오면 이전 goal의 스레드를 선점해 정리한다. 여기 취소는 그와 별개로, 다음
    goal을 보내기 **전에** 이미 정리해 두기 위한 것이다.)

    **여기서 오래 기다리면 안 된다.** 이 함수는 "도착은 확인됐고 다음 이동으로 넘어가는"
    정상 경로에서 매 이동마다 불린다 — 대기 시간이 그대로 사이클 시간에 더해진다.
    중요한 것은 취소를 *보내는* 것이지 확인을 *받는* 것이 아니다. 확인이 안 와도
    드라이버가 다음 goal에서 이전 goal을 선점해 정리하므로(위 참조) 안전하고, 새 모션이
    이전 스레드가 빠져나간 뒤에야 시작되는 것도 드라이버가 보장한다. 그래서 짧게만
    기다리고(`timeout_s`) 확인이 없으면 그냥 진행한다 — 실패로 돌리지 않는다.
    """
    if remote_handle is None:
        return
    try:
        remote_handle.cancel_goal_async()
    except Exception as exc:  # 드라이버가 이미 goal을 끝냈으면 여기서 예외가 날 수 있다
        if logger:
            logger.debug(f"_release_remote_goal: 취소 요청 실패(무시) — {exc}")
        return
    if finished.wait(timeout=timeout_s):
        return
    if logger:
        logger.debug(
            f"_release_remote_goal: {reason} — 취소 확인이 {timeout_s:.1f}초 안에 오지 "
            "않았다. 드라이버가 다음 goal에서 정리하므로 그대로 진행한다")


def approach_axis_from_pose(pose) -> list[float]:
    """파지 자세(geometry_msgs/Pose)의 **접근축**을 base 좌표 단위벡터로 낸다.

    이 프로젝트의 TCP 규약에서 접근축은 **Z축**이다 — grasp의
    `heuristic_pca._pose_from_closing_axis`가 `column_stack([x_axis, y_axis, z_axis])`로
    자세를 만들면서 `z_axis = (0, 0, -1)`(아래 방향)을 접근축으로 넣는다. 따라서 회전행렬의
    3번째 열이 base 기준 접근 방향이고, 값은 "그리퍼가 물체로 다가가는 쪽"을 가리킨다.

    **왜 필요한가.** 접근 여유·하강 깊이 같은 오프셋을 base Z에 그냥 더하면, 기울어진
    파지에서 그 오프셋이 접근축을 벗어나 수평 성분을 만든다(pick_server._pick_real의
    주석에 실측값 있음). 수직 파지에서는 이 함수가 (0, 0, -1)을 돌려주므로 예전 식과
    같은 결과가 되어, 기존 동작을 바꾸지 않는다.
    """
    from perception_common.geometry import quaternion_to_matrix

    q = pose.orientation
    rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    axis = rotation[:, 2]
    norm = float((axis ** 2).sum() ** 0.5)
    if norm < 1e-9:
        return [0.0, 0.0, -1.0]     # 회전이 깨진 경우: 예전 동작(수직 하강)으로 대체
    return [float(v / norm) for v in axis]


def plan_pick_posx(pose, offset_mm, approach_height_mm: float, depth_extra_mm: float,
                   current_zyz_deg=None):
    """파지 후보 Pose → pick이 실제로 명령할 (하강 posx, 접근 posx, 접근축).

    `pick_server._pick_real`이 쓰던 계산을 그대로 함수로 뺐다. **후보를 고르기 전 IK를
    검사하려면 실행할 좌표와 완전히 같은 좌표로 물어야 한다** — 검사와 실행이 각자
    계산하면 둘이 갈라져 "검사는 통과했는데 실행은 못 가는" 후보가 생긴다.

    두 오프셋(하강 깊이, 접근 여유)은 base Z가 아니라 **접근축**을 따라 준다. 기울어진
    파지에서 base Z에 더하면 오프셋이 수평 성분을 만들어 그리퍼가 물체를 옆으로 밀어낸다
    (2026-09-07 실측: 37도 후보에서 접근 시작점이 수평으로 48.2mm 어긋났다).
    수직 파지에서는 접근축이 (0,0,-1)이라 예전 식과 정확히 같은 값이 된다.
    """
    target_posx = tcp_posx_for_grasp(pose, offset_mm, current_zyz_deg=current_zyz_deg)
    axis = approach_axis_from_pose(pose)
    target_xyz = [target_posx[i] + depth_extra_mm * axis[i] for i in range(3)]
    approach_xyz = [target_xyz[i] - approach_height_mm * axis[i] for i in range(3)]
    return ([*target_xyz, *target_posx[3:]], [*approach_xyz, *target_posx[3:]], axis)


# --- 컨트롤러가 "이 이동은 시작조차 못 한다"고 알리는 알람 -----------------------------
# RobotError.group. dsr_msgs2/msg/RobotError.msg: SYSTEM=1, MOTION=2, TP=3, INVERTER=4,
# SAFETY_CONTROLLER=5.
ERROR_GROUP_MOTION = 2
# "[ERR] Pose(...) is NOT REACHABLE". **movel은 이 알람을 내고도 goal을 accept한 채
# 아무것도 하지 않는다** — 실패 응답을 주지 않으므로, 이 알람을 안 보면 호출부는
# overall_timeout_s(60초)를 다 채우고서야 실패로 끝난다. 2026-09-07 21:08 실물:
# place_into가 바구니 상공(334.7, -479.2, 321.6)으로 가려다 이 알람을 세 번 내고
# 매번 60초씩 기다렸다(총 3분, place_failed).
ALARM_NOT_REACHABLE = 1206
BLOCKING_MOTION_ALARMS = frozenset({ALARM_NOT_REACHABLE})

# safety_monitor.py가 쓰는 나머지 RobotError.group 값. MOTION(위)은 "이 자세로 못 간다"류
# 계획 거부라 안전 이벤트가 아니다 — SAFETY_CONTROLLER만 컨트롤러의 안전 기능(충돌·안전
# 입력 등)이 낸 알람이다.
ERROR_GROUP_SYSTEM = 1
ERROR_GROUP_SAFETY_CONTROLLER = 5
# RobotError.level (dsr_msgs2/msg/RobotError.msg).
ERROR_LEVEL_INFO = 1
ERROR_LEVEL_WARN = 2
ERROR_LEVEL_ERROR = 3


class MotionErrorMonitor:
    """`/dsr01/error`를 구독해 "이 이동은 못 한다"는 컨트롤러 알람을 기억한다.

    `call_action_blocking`에 넘기면 goal을 보낸 **뒤에** 온 알람만 보고 즉시 실패로
    끊는다. goal 전송 시각을 기준으로 거르는 이유: 이 토픽은 지난 알람도 그대로
    흘러다니므로, 시각을 안 보면 예전 알람 하나로 이후 모든 이동이 실패한다.

    `ikin`으로는 이 판정을 대신할 수 없다. 2026-09-08 실측: 실제로 거부된 자세
    (334.68, -479.25, 321.62, 22.82, -179.93, 23.16)를 ikin에 물으면
    `success=True, conv_posj=[-55.6, 32.4, 63.8, 0.05, 83.9, -55.3]`으로
    **관절 한계 안의 멀쩡한 해**를 돌려준다. 그래서 place_server._reachable()의
    사전 검사가 통과시켰고, 실패는 실제로 보내봐야만 드러난다.
    """

    def __init__(self, node, callback_group=None, codes=BLOCKING_MOTION_ALARMS):
        from dsr_msgs2.msg import RobotError

        self._codes = frozenset(codes)
        self._lock = threading.Lock()
        self._last: tuple[float, int, str] | None = None
        node.create_subscription(RobotError, MOTION_ERROR_TOPIC, self._on_error, 10,
                                 callback_group=callback_group)

    def _on_error(self, msg) -> None:
        if int(msg.group) != ERROR_GROUP_MOTION or int(msg.code) not in self._codes:
            return
        # 메시지에 stamp가 없어 수신 시각을 쓴다. 알람은 goal 수락 직후(수십 ms)에 오므로
        # 이 정도 해상도로 충분하다.
        with self._lock:
            self._last = (time.monotonic(), int(msg.code), (msg.msg1 or "").strip())

    def since(self, t0: float) -> str | None:
        """`t0` 이후에 받은 차단성 알람의 설명. 없으면 None."""
        with self._lock:
            last = self._last
        if last is None or last[0] < t0:
            return None
        return f"컨트롤러 알람 code={last[1]} {last[2]}"


def call_action_blocking(client, goal, goal_handle, send_timeout_s: float = 10.0,
                         cancel_timeout_s: float = 5.0, overall_timeout_s: float = 60.0,
                         on_timeout_verify=None, feedback_callback=None,
                         verify_poll_s: float = 1.0, logger=None,
                         error_monitor=None):
    """액션을 보내고 결과를 기다린다(현재 스레드를 막는다). goal_handle이 취소 요청을
    받으면(웹의 정지 버튼) 원격 목표도 함께 취소한다 — 안 그러면 화면엔 "취소됨"으로
    보이는데 로봇은 계속 움직이는 상태가 된다.

    **`overall_timeout_s`가 없으면 아무도 취소하지 않는 한 영원히 기다린다** — 실물로
    겪은 사고: movel_h2r 하나가 응답 없이 19분간 멈췄는데, 취소 감지만 있고 자체
    상한이 없어 사람이 수동으로 취소하기 전까지 그대로 대기했다. 넘으면 우리가
    먼저 취소를 보내고 실패로 처리한다 — pick/place 전체가 무한정 멈추는 대신
    유한 시간 안에 실패로 끝나야 재계획(FR-16)이 돌 수 있다.

    **`on_timeout_verify`**: `verify_poll_s`마다(상한까지 기다리지 않고) 부르는
    콜러블 — True면 실패로 단정하지 않고 바로 성공으로 처리한다. 실물로 확인한
    사고: movel_h2r이 목표 지점에 실제로(1~2초 만에) 정확히 도달했는데도 result
    콜백이 끝내 안 와서(get_current_posx로 도달을 직접 확인함 — DDS로 결과
    통지가 유실되는 것으로 보인다, 이 환경의 네트워크 신뢰성 문제라 근본 해결은
    못 한다) 예전엔 `overall_timeout_s`를 다 채우고 나서야 확인했다 — pick 한 번의
    movel 세 번이 각각 60초씩 걸려 파지 사이클 전체가 3분 넘게 걸리는 원인이었다
    (2026-09-04 실물로 확인). 짧은 주기로 먼저 확인해서 이 지연을 줄인다 — 진짜
    결과 콜백이 오면 그쪽을 우선한다(추측이 아니라 실제 결과가 항상 더 정확하다).
    `move_linear`가 이 자리에 실제 도달 여부 확인을 넣는다.

    `verify_poll_s`는 이 환경에서 **거의 모든 이동의 실제 소요시간을 결정한다**. 결과
    통지가 사실상 매번 유실돼(로그상 성공은 전부 on_timeout_verify 경로로 났다) 이 주기가
    곧 "도착했는데도 더 기다리는 시간"이기 때문이다. 5초로 뒀을 때 이동 8~10번짜리
    pick+place 한 사이클에서만 40초 안팎이 순수 대기로 날아갔다(2026-09-06 실물 로그).
    get_current_posx는 한가할 때 10ms 안에 답하므로 1초 주기로 확인해도 서비스에 부담이 없다.

    **`error_monitor`**(`MotionErrorMonitor`): 넘기면 goal 전송 이후에 도착한 컨트롤러
    알람을 매 주기 확인해 **즉시** 실패로 끊는다. 이게 없으면 "goal은 accept됐는데
    로봇은 안 움직이는" 경우가 `overall_timeout_s`(60초)를 다 채운다 —
    `MotionErrorMonitor` 주석의 2026-09-07 place 사고가 그것이다.

    반환: (성공 여부, 원격 액션의 result 객체 또는 None).
    """
    # 이 함수는 pick_server.py/place_server.py의 매 move() 호출마다 도니, 여기가
    # "새 모션을 실제로 로봇에 보내기 직전" 지점이다. 예전엔 여기서 확인 없이 바로
    # send_goal_async를 불러서, Stop을 이미 눌렀는데도(이전 스텝 실패 직후처럼 활성
    # goal이 없던 순간에) 다음 이동이 그대로 나가는 경우가 있었다(2026-09-05, Stop 이후
    # 재실행 사고 조사에서 확인). 새 goal을 보내기 전에 취소 여부부터 본다.
    if goal_handle.is_cancel_requested:
        return False, None
    if not client.wait_for_server(timeout_sec=5.0):
        if logger:
            logger.error("call_action_blocking: 액션 서버 응답 없음 (wait_for_server 타임아웃)")
        return False, None

    sent = threading.Event()
    state: dict = {}

    def on_send_done(future):
        state["handle"] = future.result()
        sent.set()

    # 알람 판정 기준 시각. **send_goal_async보다 먼저 잡아야 한다** — 알람은 수락 직후
    # 수십 ms 안에 오므로, 전송 뒤에 시각을 잡으면 그 알람을 "예전 것"으로 흘려보낸다.
    sent_at = time.monotonic()

    def blocked() -> str | None:
        return error_monitor.since(sent_at) if error_monitor is not None else None

    client.send_goal_async(
        goal, feedback_callback=feedback_callback).add_done_callback(on_send_done)
    sent.wait(timeout=send_timeout_s)
    remote_handle = state.get("handle")
    deadline = time.monotonic() + overall_timeout_s

    if remote_handle is None:
        # goal 수락 응답이 안 왔다. **거부가 아니라 응답 유실일 수 있다** — 실물로 확인:
        # 여기서 곧바로 실패 처리했는데 로봇은 그 목표까지 실제로 이동을 끝냈다
        # (2026-09-06, 카메라 스트림이 loopback DDS를 포화시켜 액션 응답이 유실된 구간.
        # 실패 직후 get_current_posx로 읽은 위치가 목표와 소수점까지 일치했다).
        # 결과 통지 유실을 도착 확인으로 구제하는 것(on_timeout_verify)과 같은 논리로,
        # 수락 응답 유실도 실제 도착 여부로 판정한다 — 응답이 늦게라도 오면 정상 경로로 돌아간다.
        if logger:
            logger.warning(
                f"call_action_blocking: goal 수락 응답이 {send_timeout_s:.0f}초 안에 오지 않았다 "
                "— 응답 유실일 수 있어 실제 도착 여부를 확인하며 기다린다")
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False, None
            blocked_reason = blocked()
            if blocked_reason is not None:
                # 수락 응답조차 못 받았는데 알람이 왔다 — 도착을 기다릴 이유가 없다.
                if logger:
                    logger.error(f"call_action_blocking: 이동 불가 알람 — {blocked_reason}")
                return False, None
            if sent.wait(timeout=0.1):
                remote_handle = state.get("handle")
                break
            if on_timeout_verify is not None and on_timeout_verify():
                if logger:
                    logger.warning("call_action_blocking: 수락 응답은 유실됐지만 로봇은 목표에 "
                                   "도착했다 — 성공 처리")
                # 수락 응답이 늦게 도착해 handle이 생겼을 수 있다. 생겼다면 위와 같은
                # 이유로(_release_remote_goal 참조) 반드시 취소해서 드라이버 쪽 실행
                # 스레드를 정리하고 나간다.
                late_handle = state.get("handle")
                if late_handle is not None:
                    done = threading.Event()
                    late_handle.get_result_async().add_done_callback(lambda _f: done.set())
                    _release_remote_goal(late_handle, done, logger,
                                         "수락 응답이 늦게 온 goal을 정리한다")
                return True, None
        if remote_handle is None:
            if logger:
                logger.error("call_action_blocking: 수락 응답도 없고 도착도 확인되지 않았다 "
                             "— 목표가 실제로 거부됐거나 드라이버가 응답하지 않는다")
            return False, None

    if not remote_handle.accepted:
        if logger:
            logger.error("call_action_blocking: 드라이버가 목표를 거부했다 (accepted=False)")
        return False, None

    finished = threading.Event()
    result_future = remote_handle.get_result_async()
    result_future.add_done_callback(lambda _f: finished.set())

    next_verify = time.monotonic() + verify_poll_s if on_timeout_verify is not None else None
    while not finished.wait(timeout=0.1):
        if goal_handle.is_cancel_requested:
            remote_handle.cancel_goal_async()
            finished.wait(timeout=cancel_timeout_s)
            return False, None
        blocked_reason = blocked()
        if blocked_reason is not None:
            # **컨트롤러가 이 목표를 거부했다.** goal은 accept된 채 남아 있고 결과 통지도
            # 오지 않으므로, 여기서 끊지 않으면 60초를 그대로 버린다. 취소는 타임아웃
            # 경로와 같은 정성으로 확인한다 — 안 끝난 goal 위에 다음 movel을 겹쳐 보내면
            # 그 뒤 모든 이동이 "접수는 되는데 진행이 없는" 상태로 이어진다(2026-09-06).
            if logger:
                logger.error(
                    f"call_action_blocking: 이동 불가 알람 — {blocked_reason}. "
                    "타임아웃을 기다리지 않고 즉시 취소한다")
            remote_handle.cancel_goal_async()
            for _ in range(4):
                if finished.wait(timeout=cancel_timeout_s):
                    break
            else:
                if logger:
                    logger.error("call_action_blocking: 이동 불가 알람 후 취소가 확인되지 "
                                 "않았다 — 이전 목표가 아직 살아있을 수 있다")
            return False, None
        now = time.monotonic()
        if next_verify is not None and now >= next_verify:
            if on_timeout_verify():
                if logger:
                    logger.info("call_action_blocking: on_timeout_verify로 도착 확인, 성공 처리")
                # **여기서 그냥 return하면 안 된다** — 결과를 안 기다리기로 한 것이지
                # goal을 없앤 게 아니다. 이 경로가 이 시스템에서 거의 모든 이동의
                # 정상 종료 경로라(결과 통지가 사실상 매번 유실된다) 예전에는 이동
                # 한 번마다 드라이버에 100Hz 스레드가 하나씩 영구히 쌓였다. 자세한
                # 배경과 증상은 _release_remote_goal 참조.
                _release_remote_goal(remote_handle, finished, logger,
                                     "도착을 직접 확인해 결과를 더 기다리지 않는다")
                return True, None
            next_verify = now + verify_poll_s
        if now > deadline:
            # 실제 액션 결과는 끝내 안 왔다 — 우리가 먼저 포기하고 취소한 것이다.
            # 즉 "로봇/컨트롤러가 실패라고 답했다"가 아니라 "우리가 응답을 못 받아 시간초과시켰다".
            if logger:
                logger.error(
                    f"call_action_blocking: {overall_timeout_s:.0f}초 타임아웃 — 실제 결과를 "
                    "못 받아 강제 취소함 (로봇이 실패라고 응답한 게 아니라 우리가 포기한 것)")
            remote_handle.cancel_goal_async()
            # **취소가 실제로 끝났는지 확인해야 한다.** `finished`는 result_future의 완료
            # 콜백이라, 이게 set되지 않았다는 건 이전 goal이 드라이버 쪽에서 여전히
            # 활성 상태(CANCELING이거나 그냥 계속 실행 중)라는 뜻이다. 예전엔 여기서
            # 5초만 기다리고 확인 없이 그냥 실패를 반환해서, 호출부(pick_server/
            # place_server)가 "이 movel은 끝났다"고 착각한 채 곧바로 새 movel을 같은
            # 액션 서버에 보냈다 — 이전 목표가 안 끝난 채 새 목표가 겹치면 액션 서버가
            # 어떻게 반응할지 우리가 보장 못 하고, 실제로 그 뒤 모든 movel이 "접수는
            #되는데 진행이 없는" 상태로 줄줄이 이어지는 걸 실물로 확인했다(2026-09-06).
            # 취소가 진짜로 끝날 때까지 더 오래, 더 여러 번 확인한다 — 여기서 끝내
            # 확인 안 되면 이 goal이 여전히 살아있을 수 있다는 걸 호출부에 분명히
            # 알려야 하므로 일반 실패와 다르게 로그를 남긴다.
            for _ in range(4):
                if finished.wait(timeout=cancel_timeout_s):
                    if logger:
                        logger.warning("call_action_blocking: 취소 확인됨 (이전 목표 종료)")
                    return False, None
                remote_handle.cancel_goal_async()
            if logger:
                logger.error(
                    "call_action_blocking: 취소가 끝내 확인되지 않았다 — 이전 목표가 "
                    "드라이버 쪽에서 여전히 살아있을 수 있다. 이 상태에서 곧바로 새 "
                    "movel을 보내면 액션 서버가 꼬일 위험이 있다 (수동 확인/재시작 권장)")
            return False, None

    result = result_future.result().result
    success = bool(getattr(result, "success", False))
    if logger and not success:
        # deadline을 넘기지 않고 실제 result 콜백이 왔는데도 실패다 — 이번엔 우리 쪽
        # 타임아웃이 아니라 컨트롤러/드라이버 자신이 "실패"라고 응답한 것이다. 원인이
        # 다르므로(클라이언트 타임아웃 vs 컨트롤러 자체 실패) 반드시 구분해서 남긴다.
        logger.error("call_action_blocking: 액션 결과가 실제로 도착했고 success=False "
                    "(우리 타임아웃이 아니라 컨트롤러/드라이버 자체가 실패로 응답함)")
    return success, result


def move_linear(client, target_pos: list[float], goal_handle,
                vel_mm_s: float, acc_mm_s2: float,
                vel_deg_s: float, acc_deg_s2: float,
                posx_client=None,
                position_tolerance_mm: float = 3.0, rotation_tolerance_deg: float = 3.0,
                logger=None, error_monitor=None) -> tuple[bool, list[float] | None]:
    """MovelH2r 하나를 블로킹으로 실행. `target_pos`는 [x,y,z,rx,ry,rz](mm, deg).

    `posx_client`를 넘기면 타임아웃 시 get_current_posx로 실제 위치를 한 번 더
    확인해서, `position_tolerance_mm`/`rotation_tolerance_deg` 안에 들어와 있으면
    성공으로 처리한다 — call_action_blocking의 on_timeout_verify 참조(결과 통지 유실 대응).

    **위치만이 아니라 회전도 확인해야 한다.** 제자리 회전(위치는 그대로, 회전만 바뀌는
    이동 — place_server의 "바구니 배치 자세로 회전" 등)에서 위치만 보면, 회전이 아직
    끝나지 않았어도 위치는 시작부터 이미 허용오차 안이라 **즉시 "도착"으로 오판**한다.
    그러면 호출부가 회전이 실제로 끝나기 전에 다음 이동(수평 이동)을 계산·전송하게
    되고, 그 다음 이동이 (완료 안 된) 회전 중간값 + 큰 위치 변화를 동시에 요구하는
    처지가 되어 로봇이 아예 안 움직이는 걸 실물로 확인했다(2026-09-06 — 회전행렬로
    비교해보니 그때 보낸 회전 목표가 직전 회전의 시작·끝 어느 쪽과도 42도 떨어진,
    회전 도중의 값이었다). ZYZ는 ry가 180도 근처면 다른 (rx,rz) 조합이 같은 방향을
    나타낼 수 있어(모듈 docstring) 성분별 차이가 아니라 회전행렬 각도차로 비교한다
    (`_rotation_angle_diff_deg`).

    반환: (성공 여부, 도착 시점의 실제 pose 또는 None). 이 pose는 액션 feedback
    (`MovelH2r.Feedback.pos`, 드라이버가 100Hz로 채워 보낸다)에서 그대로 가져온
    것이라 get_current_posx를 다시 호출하지 않고도 다음 이동의 회전값 재사용에
    쓸 수 있다 — movel 직후 aux_control(get_current_posx)이 10~20초 이상
    무응답인 구간이 실물에서 관측됐는데(위 get_current_posx 참조), 예전에는
    pick_server.py/place_server.py가 이동마다 그 서비스를 다시 불러서 이 지연을
    그대로 맞았다. feedback을 못 받았으면(드묾) None — 호출부가 get_current_posx로
    폴백해야 한다.
    """
    from dsr_msgs2.action import MovelH2r

    goal = MovelH2r.Goal()
    goal.target_pos = [float(v) for v in target_pos]
    goal.target_vel = [float(vel_mm_s), float(vel_deg_s)]
    goal.target_acc = [float(acc_mm_s2), float(acc_deg_s2)]

    last_pose: list[float] | None = None

    def on_feedback(feedback_msg):
        nonlocal last_pose
        last_pose = list(feedback_msg.feedback.pos)

    def verify_arrived():
        # 먼저 액션 feedback(`last_pose`, 드라이버가 100Hz로 직접 채워 보낸다)을 본다 —
        # get_current_posx(aux_control 서비스)를 안 거치므로 perception/grasp의 posx
        # 폴링(robot_pose.py, 2Hz)과 자원을 다투지 않는다. 실물로 확인한 사고: pick의
        # 들어올리기 이동이 실제로는 몇 초 만에 끝났는데(사람 눈으로 확인), 이동 직후
        # aux_control이 계속 응답이 없어(문서 상단 참조) get_current_posx만 썼던 예전
        # 코드는 verify_arrived가 60초 내내 실패해 결국 "들어올리기 실패"로 오판했다
        # (2026-09-05, control_watch5.log — width=64.5mm으로 이미 뭔가를 문 채 88초
        # 만에 실패 처리됨). feedback이 허용오차 안이면 그것만으로 충분한 근거로 본다 —
        # 액션 자체가 이 target_pos로 보낸 이동이므로, feedback이 그 근처에 와 있다는
        # 것 자체가 "그 이동이 물리적으로 끝났다"는 직접 증거다.
        def matches(pose6) -> bool:
            pos_ok = all(abs(a - b) <= position_tolerance_mm
                        for a, b in zip(pose6[:3], target_pos[:3]))
            rot_diff = _rotation_angle_diff_deg(pose6[3:6], target_pos[3:6])
            return pos_ok and rot_diff <= rotation_tolerance_deg

        if last_pose is not None:
            if matches(last_pose):
                return True
            if logger:
                # feedback은 받고 있지만 아직 허용오차 밖 — 진짜로 아직 이동 중일 수도,
                # 도착은 했는데 feedback이 마지막 값을 못 갱신했을 수도 있다. 둘을
                # 구분하려고 오차값 자체를 남긴다.
                pos_deltas = [round(abs(a - b), 1) for a, b in zip(last_pose[:3], target_pos[:3])]
                rot_diff = round(_rotation_angle_diff_deg(last_pose[3:6], target_pos[3:6]), 1)
                logger.info(
                    f"verify_arrived: feedback 있음, 허용오차 밖 (위치오차={pos_deltas}mm, "
                    f"회전오차={rot_diff}deg, 허용={position_tolerance_mm}mm/"
                    f"{rotation_tolerance_deg}deg)", throttle_duration_sec=5.0)
        elif logger:
            # aux_control 무응답 구간과 겹치는 게 보통이라(모듈 docstring 참조), 이 경로로
            # 자주 빠지는 것 자체는 정상이다 — 문제는 posx_client 폴백까지 실패하는 경우다.
            logger.info("verify_arrived: 아직 feedback을 못 받음(last_pose=None)",
                        throttle_duration_sec=5.0)
        if posx_client is None:
            return False
        # 이 호출이 블로킹이라 그대로 verify 주기를 늘린다 — 1초 주기로 자주 부르는 만큼
        # 한 번에 오래 붙잡지 않게 짧게 끊는다(한가하면 10ms, 밀리면 다음 주기에 다시 본다).
        current = get_current_posx(posx_client, goal_handle, timeout_s=1.0, retries=1)
        if current is None:
            if logger:
                logger.info("verify_arrived: get_current_posx 폴백도 무응답",
                            throttle_duration_sec=5.0)
            return False
        ok = matches(current)
        if logger:
            logger.info(f"verify_arrived: get_current_posx 폴백으로 확인 → "
                        f"{'도착' if ok else '미도착'} (실제={[round(v, 1) for v in current[:3]]}, "
                        f"목표={[round(v, 1) for v in target_pos[:3]]}, "
                        f"회전오차={round(_rotation_angle_diff_deg(current[3:6], target_pos[3:6]), 1)}deg)",
                        throttle_duration_sec=5.0)
        return ok

    success, _ = call_action_blocking(client, goal, goal_handle, on_timeout_verify=verify_arrived,
                                      feedback_callback=on_feedback, logger=logger,
                                      error_monitor=error_monitor)
    if logger and not success:
        logger.error(
            f"move_linear 실패: feedback 마지막값={last_pose}, 목표={list(target_pos)} "
            f"(둘 다 있는데 실패면 허용오차 초과, feedback이 None이면 콜백이 아예 안 옴)")
    return success, (last_pose if success else None)


def move_joint(client, target_deg: list[float], goal_handle,
              vel_deg_s: float, acc_deg_s2: float, logger=None,
              error_monitor=None) -> bool:
    """MovejH2r 하나를 블로킹으로 실행. `target_deg`는 j1~j6(도).

    movel과 달리 도착 확인(on_timeout_verify)을 못 넘긴다 — 목표가 관절각이라
    get_current_posx(TCP 좌표)로는 도달 여부를 판정할 수 없기 때문이다. 그래서 이쪽은
    결과 통지가 유실되면 그대로 타임아웃 실패다. logger만이라도 넘겨 실패 원인
    (수락 응답 유실 / 컨트롤러 실패 응답 / 우리 타임아웃)을 구분할 수 있게 한다."""
    from dsr_msgs2.action import MovejH2r

    goal = MovejH2r.Goal()
    goal.target_pos = [float(v) for v in target_deg]
    goal.target_vel = [float(vel_deg_s)] * 6
    goal.target_acc = [float(acc_deg_s2)] * 6
    success, _ = call_action_blocking(client, goal, goal_handle, logger=logger,
                                      error_monitor=error_monitor)
    return success


def send_gripper_command(client, command: str, timeout_s: float = 3.0) -> bool:
    """/onrobot/sendCommand(SetCommand) 호출. 'c'=닫기, 'o'=열기.

    힘(force)은 이 문자 명령으로 정확한 N값을 지정할 수 없다 — 서버가 내부에 들고 있는
    값에 상대적으로만 ±25씩 조절되는 구조라(onrobot_rg_control의 genCommand 참조),
    profile별 max_grip_force_n을 여기서 정확히 반영하지 못한다. 서버 기본값(대략 6N
    상당, 보수적인 축)을 그대로 쓴다 — 1단계 제약 중 하나로 남겨둔다.
    """
    from onrobot_rg_msgs.srv import SetCommand

    if not client.wait_for_service(timeout_sec=2.0):
        return False
    request = SetCommand.Request()
    request.command = command

    done = threading.Event()
    client.call_async(request).add_done_callback(lambda _f: done.set())
    done.wait(timeout=timeout_s)
    return True  # 서비스 자체는 접수만 하고 바로 응답한다 — 물리적 완료는 별도로 기다린다


def gripper_width_command(width_m: float, force_n: float | None = None) -> str:
    """`send_gripper_command`에 넘길 문자열 — 정수 문자열은 목표 개폭(0.1mm 단위)으로
    해석된다(onrobot_rg_control.genCommand, sendCommandCallback 경로라 실물 pick/place에
    쓴 'c'와 같은 정확한 변환을 거친다). 'c'/'o'는 완전히 닫기/최대로 열기만 가능해서
    place_into처럼 특정 개폭(gripper_open_m)을 원할 때는 이걸 쓴다.

    `force_n`을 주면 `"<개폭>,<힘>"`으로 만들어 **파지력을 절대값으로** 지정한다
    (2026-09-07에 드라이버 genCommand에 추가한 형식). 안 주면 드라이버가 들고 있는
    직전 힘을 그대로 쓴다 — 예전과 같은 동작이다.

    **왜 힘을 명시해야 하는가**: 드라이버는 기동 시 `rgfr = max_force`로 박아두고
    'i'/'d'로 ±25씩 상대 조절만 지원했다. 그래서 skill_params.yaml의 프로필별
    `max_grip_force_n`(normal 20N, fragile 5N, deformable 12N)이 선언만 되고 한 번도
    적용되지 않았다 — fragile 물체도 RG2 최대 40N으로 쥐고 있었다.
    """
    width = int(round(width_m * 10000))
    if force_n is None:
        return str(width)
    return f"{width},{int(round(force_n * 10))}"


def gripper_width_mm(pose_client, joint_angle: float, timeout_s: float = 1.5) -> float | None:
    """관절각(rad) → 개폭(mm). `/onrobot/pose`(GripperPose)에 `known.theta`로 물어보면
    서버가 자기 그리퍼 타입(RG2/RG6)에 맞는 기하 상수로 변환해 준다 — 그 상수를 여기서
    하드코딩하면 그리퍼 모델을 잘못 가정할 위험이 있다(RG6로 착각했던 사고를 이미 겪었다).

    측정 실패(서비스 미기동·타임아웃)하면 **None**을 돌려준다 — 예전엔 0.0으로
    대체했는데, 호출부(pick_server._pick_real)가 "빈 채로 닫힘" 판정에 이 값을 그대로
    쓰는 바람에 이 서비스가 일시적으로 응답이 없었을 뿐인데도 "확실히 안 물었다"로
    오판할 수 있었다(2026-09-05, 들어올리기 오판 조사 중 함께 확인). 실제로 0.0mm으로
    측정된 것과 "몰라서" 0.0을 내려준 것은 호출부가 반드시 구분해야 한다.
    """
    from onrobot_rg_msgs.srv import GripperPose

    if not pose_client.wait_for_service(timeout_sec=1.0):
        return None
    request = GripperPose.Request()
    request.known.theta = float(joint_angle)

    done = threading.Event()
    future = pose_client.call_async(request)
    future.add_done_callback(lambda _f: done.set())
    done.wait(timeout=timeout_s)
    result = future.result()
    return float(result.pose.x) * 1000.0 if result is not None else None


def wait_gripper_settled(get_joint_angle, goal_handle, timeout_s: float = 8.0,
                         poll_interval_s: float = 0.2, stable_polls: int = 3,
                         get_status=None, status_max_age_s: float = 1.0,
                         motion_start_timeout_s: float = 2.0) -> float | None:
    """그리퍼 동작이 끝날 때까지 기다린다. `/onrobot/sendCommand`는 완료 신호가 없는
    fire-and-forget이라(모듈 docstring 참조) 직접 확인해야 한다.

    `get_joint_angle`은 최신 관절각(rad)을 돌려주는 콜러블(없으면 None) — 호출부가
    `/onrobot_joint_states` 구독의 최신값을 캡처해 넘긴다. 취소되거나 타임아웃되면 None.

    **관절각 정지로 완료를 추측하지 않는다.** RG2가 `gsta` bit0(Busy)로 "지금 움직이는
    중"을 직접 알려주므로(onrobot_rg_msgs/OnRobotRGInput) 그걸 쓴다. `get_status`는
    `(수신시각_monotonic, gsta, ...)` 또는 None을 돌려주는 콜러블이다.

    **왜 바꿨나 (2026-09-08 실물)**: 예전엔 관절각이 `stable_polls`회(0.6초) 안 변하면
    완료로 봤는데, 그 판정은 세 가지를 구분하지 못한다 — ①진짜 완료 ②명령이 아직
    시작되지 않음 ③피드백이 끊겨 값이 얼어붙음. 실측에서 닫기 명령의 시작 지연이
    약 0.6초였고 그게 `stable_polls * poll_interval_s`와 정확히 같아서, 닫기 목표
    17.3mm인데 **52.7mm에서 완료 판정**이 났다. 그 시점의 상태가 그대로 파지 성공
    판정의 근거(`post`)로 쓰여, 닫히는 도중의 과도값으로 성패를 가르고 있었다.

    `status_max_age_s`는 "관측이 살아 있는가"의 기준이다 — 낡았으면 Busy를 안 쓰고
    아래 폴백으로 내려간다. 값이 얼어붙은 것을 정지로 오독하던 ③이 여기서 갈린다.

    `motion_start_timeout_s`는 물리량이 아니라 **명령→동작개시 지연의 상한**이다.
    이 안에 Busy가 한 번도 서지 않으면 "움직일 필요가 없었다"(이미 목표 개폭)로 보고
    끝낸다 — 기동 시 이미 열려 있는 그리퍼를 다시 여는 경우가 그렇다. 실측 지연
    약 0.6초의 3배 여유로 잡았다.

    `/onrobot/status`를 발행하지 않는 드라이버(2026-09-07 이전 빌드)에서는 `get_status`가
    없거나 계속 낡은 값만 오므로, 그때는 **예전 관절각 안정 판정 그대로** 동작한다.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    last = None
    stable = 0
    saw_busy = False
    last_angle = None
    while time.monotonic() < deadline:
        if goal_handle.is_cancel_requested:
            return None
        current = get_joint_angle()
        if current is not None:
            last_angle = current

        status = get_status() if get_status is not None else None
        now = time.monotonic()
        if status is not None and now - status[0] <= status_max_age_s:
            if status[1] & 0x01:            # Busy — 동작 중
                saw_busy = True
            elif saw_busy:                  # 움직였고, 이제 멈췄다
                return last_angle
            elif now - started >= motion_start_timeout_s:
                # 지연 상한이 지나도록 Busy가 서지 않았다 = 움직일 필요가 없었다.
                return last_angle
            stable = 0                      # Busy를 쓰는 동안 각도 안정은 보지 않는다
            last = current
            time.sleep(poll_interval_s)
            continue

        # 폴백: 상태 토픽이 없거나 낡았다 — 예전대로 관절각 정지로 판단한다.
        if current is not None and last is not None and abs(current - last) < 1e-3:
            stable += 1
            if stable >= stable_polls:
                return current
        else:
            stable = 0
        last = current
        time.sleep(poll_interval_s)
    return None
