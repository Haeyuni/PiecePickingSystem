"""단일 스레드 ROS2 액션 클라이언트 헬퍼: spin_once 루프로 블로킹 대기.

grasp_test는 control 패키지의 액션 서버들과 달리 외부 요청을 동시에 받는 서버가 아니라
순차 실행되는 벤치마크 스크립트다. 그래서 취소 전파를 위한 ReentrantCallbackGroup·
MultiThreadedExecutor(dsr_motion.py 참조)가 필요 없고, LiveSceneCapture.capture()와 같은
spin_once 폴링만으로 충분하다.
"""
import time

import rclpy


def call_blocking(node, client, goal, goal_timeout_s=5.0, result_timeout_s=30.0):
    """액션 하나를 보내고 결과를 기다린다.

    반환: (ok, result, error_code). ok=True는 "결과를 받았다"는 뜻일 뿐이다 — 액션마다
    성공을 표현하는 필드가 다르므로(MovelH2r/MovejH2r은 result.success, GripperCommand는
    success 필드가 아예 없고 position/stalled/reached_goal로 판단) 그 해석은 호출자 몫이다.
    """
    if not client.wait_for_server(timeout_sec=goal_timeout_s):
        return False, None, 'ACTION_SERVER_UNAVAILABLE'

    send_future = client.send_goal_async(goal)
    if not _spin_until_done(node, send_future, goal_timeout_s):
        return False, None, 'ACTION_GOAL_TIMEOUT'
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        return False, None, 'ACTION_GOAL_REJECTED'

    result_future = goal_handle.get_result_async()
    if not _spin_until_done(node, result_future, result_timeout_s):
        return False, None, 'ACTION_RESULT_TIMEOUT'
    return True, result_future.result().result, ''


def _spin_until_done(node, future, timeout_s):
    end = time.monotonic() + timeout_s
    while rclpy.ok() and not future.done() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    return future.done()


def call_service_blocking(node, client, request, timeout_s=3.0):
    """서비스 호출 하나를 blocking으로 기다린다 (call_blocking과 같은 spin_once 폴링
    패턴 — 액션이 아니라 서비스라 별도로 둔다). 반환: (ok, response, error_code)."""
    if not client.wait_for_service(timeout_sec=timeout_s):
        return False, None, 'SERVICE_UNAVAILABLE'
    future = client.call_async(request)
    if not _spin_until_done(node, future, timeout_s):
        return False, None, 'SERVICE_TIMEOUT'
    return True, future.result(), ''
