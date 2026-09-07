"""현재 TCP 자세 조회 (eye-in-hand 변환의 나머지 절반).

두산 드라이버의 `get_current_posx` **서비스를 직접 부른다.** `DSR_ROBOT2`/`DR_init`을
import하지 않는 이유는 그쪽이 전역 상태(`DR_init.__dsr__node`)를 요구해서, perception 노드가
로봇 API의 초기화 순서에 묶이기 때문이다. 서비스 하나만 부르면 그런 결합이 생기지 않는다.
값 자체는 `get_current_posx()`가 감싸는 것과 같은 서비스라 캘리브레이션과 동일한 정의다.

**자세는 주기적으로 받아 캐시한다.** 검출 콜백 안에서 서비스를 동기 호출하면 같은 실행기
안에서 응답을 기다리다 교착된다. 자세는 로봇이 움직일 때만 바뀌고 검출은 1~2Hz라 캐시로 충분하다.
"""
import threading
import time

from dsr_msgs2.srv import GetCurrentPosx

SERVICE_NAME = "/dsr01/dsr_controller2/aux_control/get_current_posx"
DR_BASE = 0


class RobotPoseClient:
    """최근 TCP 자세를 들고 있는다. `posx`는 [x, y, z, rx, ry, rz] (mm, ZYZ 도)."""

    def __init__(self, node, service_name: str = SERVICE_NAME, period_s: float = 0.5,
                 callback_group=None, pending_timeout_s: float = 3.0):
        self._node = node
        self._lock = threading.Lock()
        self._posx: list[float] | None = None
        self._updated_at: float | None = None
        self._pending = False
        self._pending_since: float | None = None
        self._pending_timeout_s = pending_timeout_s
        self._client = node.create_client(GetCurrentPosx, service_name,
                                          callback_group=callback_group)
        # perception·grasp가 각자 이 클래스를 쓰므로 0.2s(5Hz)였을 때 둘이 합쳐 10Hz로
        # 같은 서비스를 두드렸다 — control이 movel 도착 확인차 이 서비스를 부르는 순간
        # (2026-09-05 실물로 확인: place_into가 이미 도착했는데도 응답을 못 받아 60초
        # 타임아웃으로 실패) 이 트래픽에 밀려 응답을 못 받았다. 검출이 1~2Hz라 0.5s로
        # 늦춰도(posx max_age_s=1.0 대비 여유 2배) 신선도는 그대로 유지된다.
        self._timer = node.create_timer(period_s, self._request,
                                        callback_group=callback_group)

    def _request(self) -> None:
        if self._pending:
            # 이전 요청이 응답 없이 너무 오래 걸려 있으면 포기하고 새로 보낸다.
            # 실물로 겪은 사고: 응답 하나가 유실되자(드라이버가 한동안 무응답이던 구간)
            # _pending이 영원히 True로 남아 그 뒤로 다시는 요청을 안 보냈다 —
            # posx()가 계속 None을 돌려주고 perception이 "입력 대기 중"만 반복했다.
            if (self._pending_since is not None
                    and time.monotonic() - self._pending_since > self._pending_timeout_s):
                self._node.get_logger().warning(
                    "get_current_posx 응답이 없어 이전 요청을 포기하고 다시 보낸다",
                    throttle_duration_sec=5.0)
            else:
                return
        if not self._client.service_is_ready():
            return
        self._pending = True
        self._pending_since = time.monotonic()
        request = GetCurrentPosx.Request()
        request.ref = DR_BASE
        self._client.call_async(request).add_done_callback(self._on_response)

    def _on_response(self, future) -> None:
        self._pending = False
        try:
            response = future.result()
        except Exception as e:
            self._node.get_logger().warning(f"get_current_posx 실패: {e}",
                                            throttle_duration_sec=5.0)
            return
        if not response.success or not response.task_pos_info:
            self._node.get_logger().warning("get_current_posx가 success=false를 반환",
                                            throttle_duration_sec=5.0)
            return
        # task_pos_info[0].data = [x, y, z, rx, ry, rz, solution_space]
        data = list(response.task_pos_info[0].data)
        if len(data) < 6:
            self._node.get_logger().warning(f"posx 길이가 6 미만: {data}",
                                            throttle_duration_sec=5.0)
            return
        with self._lock:
            self._posx = [float(v) for v in data[:6]]
            self._updated_at = time.monotonic()

    def posx(self, max_age_s: float) -> list[float] | None:
        """`max_age_s`보다 오래되지 않은 자세. 없으면 None.

        오래된 자세로 계산한 좌표를 내보내느니 **그 프레임을 통째로 버리는 편이 낫다.**
        로봇이 움직이는 동안 옛 자세로 변환하면 물체가 실제와 다른 곳에 있다고 발행되고,
        그 좌표는 그대로 파지 목표가 된다.
        """
        with self._lock:
            if self._posx is None or self._updated_at is None:
                return None
            if time.monotonic() - self._updated_at > max_age_s:
                return None
            return list(self._posx)

    def ready(self) -> bool:
        return self._client.service_is_ready()
