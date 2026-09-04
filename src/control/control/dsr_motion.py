"""dsr_msgs2 액션 호출 공용 헬퍼: 블로킹 대기 + 취소 전파, pose(mm, 쿼터니언) → posx 변환.

home_server.py가 MovejH2r을 부르며 만든 패턴(ActionClient를 threading.Event로 동기식
대기하되, 콜백은 MultiThreadedExecutor의 다른 스레드가 처리)을 pick_server.py/
place_server.py의 MovelH2r 호출도 그대로 필요로 해서 여기로 뺐다. 세 서버 모두
ReentrantCallbackGroup을 액션 서버와 이 모듈이 만드는 ActionClient에 공유해야
데드락 없이 동작한다 — 호출부에서 그 그룹을 넘겨준다.
"""
import threading

from perception_common.geometry import matrix_to_zyz_deg, quaternion_to_matrix

MOVEJ_ACTION = "/dsr01/motion/movej_h2r"
MOVEL_ACTION = "/dsr01/motion/movel_h2r"
GRIPPER_ACTION = "/rg6_controller"


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


def move_gripper(client, position_m: float, max_effort_n: float, goal_handle) -> tuple[bool, float]:
    """OnRobot RG6를 /rg6_controller(control_msgs/GripperCommand)로 블로킹 제어.

    `position_m`은 목표 개폭(m, 0=완전히 닫힘). `max_effort_n`이 그리퍼 자체의 힘 제한이다
    — 팔의 접촉감지(compliance.py, 아직 미구현)와는 별개로, 그리퍼 하드웨어가 이 이상
    힘을 주지 않고 멈추는 안전장치라 위치제어만으로도 최소한의 보호가 된다.

    반환: (목표에 도달했는지, 실제 개폭 m).
    """
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position_m)
    goal.command.max_effort = float(max_effort_n)
    success, result = call_action_blocking(client, goal, goal_handle)
    width = float(result.position) if result is not None else 0.0
    return success, width
