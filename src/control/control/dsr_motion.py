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

MOVEJ_ACTION = "/dsr01/motion/movej_h2r"
MOVEL_ACTION = "/dsr01/motion/movel_h2r"
GET_CURRENT_POSX_SERVICE = "/dsr01/dsr_controller2/aux_control/get_current_posx"
GRIPPER_COMMAND_SERVICE = "/onrobot/sendCommand"
GRIPPER_JOINT_STATES_TOPIC = "/onrobot_joint_states"


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


def call_action_blocking(client, goal, goal_handle, send_timeout_s: float = 10.0,
                         cancel_timeout_s: float = 5.0, overall_timeout_s: float = 60.0,
                         on_timeout_verify=None, feedback_callback=None,
                         verify_poll_s: float = 1.0, logger=None):
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
                logger=None) -> tuple[bool, list[float] | None]:
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
                                      feedback_callback=on_feedback, logger=logger)
    if logger and not success:
        logger.error(
            f"move_linear 실패: feedback 마지막값={last_pose}, 목표={list(target_pos)} "
            f"(둘 다 있는데 실패면 허용오차 초과, feedback이 None이면 콜백이 아예 안 옴)")
    return success, (last_pose if success else None)


def move_joint(client, target_deg: list[float], goal_handle,
              vel_deg_s: float, acc_deg_s2: float, logger=None) -> bool:
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
    success, _ = call_action_blocking(client, goal, goal_handle, logger=logger)
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
