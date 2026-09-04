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
import threading
import time

from perception_common.geometry import matrix_to_zyz_deg, quaternion_to_matrix

MOVEJ_ACTION = "/dsr01/motion/movej_h2r"
MOVEL_ACTION = "/dsr01/motion/movel_h2r"
GET_CURRENT_POSX_SERVICE = "/dsr01/dsr_controller2/aux_control/get_current_posx"
GRIPPER_COMMAND_SERVICE = "/onrobot/sendCommand"
GRIPPER_JOINT_STATES_TOPIC = "/onrobot_joint_states"


def get_current_posx(client, timeout_s: float = 3.0) -> list[float] | None:
    """get_current_posx를 한 번 동기 호출한다 — [x,y,z,rx,ry,rz](mm, deg).

    perception_common.robot_pose.RobotPoseClient(주기 캐싱, 검출 프레임마다 필요)와
    달리 여기서는 movel 사이사이 딱 한 번씩만 필요해 별도 캐시 없이 그때그때 묻는다.
    """
    from dsr_msgs2.srv import GetCurrentPosx

    if not client.wait_for_service(timeout_sec=2.0):
        return None
    request = GetCurrentPosx.Request()
    request.ref = 0  # DR_BASE

    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _f: done.set())
    done.wait(timeout=timeout_s)
    result = future.result()
    if result is None or not result.success or not result.task_pos_info:
        return None
    data = list(result.task_pos_info[0].data)
    return [float(v) for v in data[:6]] if len(data) >= 6 else None


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


def call_action_blocking(client, goal, goal_handle, send_timeout_s: float = 10.0,
                         cancel_timeout_s: float = 5.0):
    """액션을 보내고 결과를 기다린다(현재 스레드를 막는다). goal_handle이 취소 요청을
    받으면(웹의 정지 버튼) 원격 목표도 함께 취소한다 — 안 그러면 화면엔 "취소됨"으로
    보이는데 로봇은 계속 움직이는 상태가 된다.

    반환: (성공 여부, 원격 액션의 result 객체 또는 None).
    """
    if not client.wait_for_server(timeout_sec=5.0):
        return False, None

    sent = threading.Event()
    state: dict = {}

    def on_send_done(future):
        state["handle"] = future.result()
        sent.set()

    client.send_goal_async(goal).add_done_callback(on_send_done)
    sent.wait(timeout=send_timeout_s)
    remote_handle = state.get("handle")
    if remote_handle is None or not remote_handle.accepted:
        return False, None

    finished = threading.Event()
    result_future = remote_handle.get_result_async()
    result_future.add_done_callback(lambda _f: finished.set())

    while not finished.wait(timeout=0.1):
        if goal_handle.is_cancel_requested:
            remote_handle.cancel_goal_async()
            finished.wait(timeout=cancel_timeout_s)
            return False, None

    result = result_future.result().result
    return bool(getattr(result, "success", False)), result


def move_linear(client, target_pos: list[float], goal_handle,
                vel_mm_s: float, acc_mm_s2: float,
                vel_deg_s: float, acc_deg_s2: float) -> bool:
    """MovelH2r 하나를 블로킹으로 실행. `target_pos`는 [x,y,z,rx,ry,rz](mm, deg)."""
    from dsr_msgs2.action import MovelH2r

    goal = MovelH2r.Goal()
    goal.target_pos = [float(v) for v in target_pos]
    goal.target_vel = [float(vel_mm_s), float(vel_deg_s)]
    goal.target_acc = [float(acc_mm_s2), float(acc_deg_s2)]
    success, _ = call_action_blocking(client, goal, goal_handle)
    return success


def move_joint(client, target_deg: list[float], goal_handle,
              vel_deg_s: float, acc_deg_s2: float) -> bool:
    """MovejH2r 하나를 블로킹으로 실행. `target_deg`는 j1~j6(도)."""
    from dsr_msgs2.action import MovejH2r

    goal = MovejH2r.Goal()
    goal.target_pos = [float(v) for v in target_deg]
    goal.target_vel = [float(vel_deg_s)] * 6
    goal.target_acc = [float(acc_deg_s2)] * 6
    success, _ = call_action_blocking(client, goal, goal_handle)
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


def gripper_width_command(width_m: float) -> str:
    """`send_gripper_command`에 넘길 문자열 — 정수 문자열은 목표 개폭(0.1mm 단위)으로
    해석된다(onrobot_rg_control.genCommand, sendCommandCallback 경로라 실물 pick/place에
    쓴 'c'와 같은 정확한 변환을 거친다). 'c'/'o'는 완전히 닫기/최대로 열기만 가능해서
    place_into처럼 특정 개폭(gripper_open_m)을 원할 때는 이걸 쓴다."""
    return str(int(round(width_m * 10000)))


def gripper_width_mm(pose_client, joint_angle: float, timeout_s: float = 1.5) -> float:
    """관절각(rad) → 개폭(mm). `/onrobot/pose`(GripperPose)에 `known.theta`로 물어보면
    서버가 자기 그리퍼 타입(RG2/RG6)에 맞는 기하 상수로 변환해 준다 — 그 상수를 여기서
    하드코딩하면 그리퍼 모델을 잘못 가정할 위험이 있다(RG6로 착각했던 사고를 이미 겪었다).
    실패하면 0.0(정보성 값이라 실패해도 pick/place 자체를 막지 않는다).
    """
    from onrobot_rg_msgs.srv import GripperPose

    if not pose_client.wait_for_service(timeout_sec=1.0):
        return 0.0
    request = GripperPose.Request()
    request.known.theta = float(joint_angle)

    done = threading.Event()
    future = pose_client.call_async(request)
    future.add_done_callback(lambda _f: done.set())
    done.wait(timeout=timeout_s)
    result = future.result()
    return float(result.pose.x) * 1000.0 if result is not None else 0.0


def wait_gripper_settled(get_joint_angle, goal_handle, timeout_s: float = 8.0,
                         poll_interval_s: float = 0.2, stable_polls: int = 3) -> float | None:
    """그리퍼 관절각(rad)이 더 안 바뀔 때까지 기다린다. `/onrobot/sendCommand`는 완료
    신호가 없는 fire-and-forget이라(모듈 docstring 참조) 직접 폴링해서 확인한다.

    `get_joint_angle`은 최신 관절각을 돌려주는 콜러블(없으면 None) — 호출부가
    `/onrobot_joint_states` 구독의 최신값을 캡처해 넘긴다. 취소되거나 타임아웃되면 None.
    """
    deadline = time.monotonic() + timeout_s
    last = None
    stable = 0
    while time.monotonic() < deadline:
        if goal_handle.is_cancel_requested:
            return None
        current = get_joint_angle()
        if current is not None and last is not None and abs(current - last) < 1e-3:
            stable += 1
            if stable >= stable_polls:
                return current
        else:
            stable = 0
        last = current
        time.sleep(poll_interval_s)
    return None
