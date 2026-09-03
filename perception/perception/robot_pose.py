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

    def __init__(self, node, service_name: str = SERVICE_NAME, period_s: float = 0.2,
                 callback_group=None):
        self._node = node
        self._lock = threading.Lock()
        self._posx: list[float] | None = None
        self._updated_at: float | None = None
        self._pending = False
        self._client = node.create_client(GetCurrentPosx, service_name,
                                          callback_group=callback_group)
        self._timer = node.create_timer(period_s, self._request,
                                        callback_group=callback_group)

    def _request(self) -> None:
        if self._pending or not self._client.service_is_ready():
            return
        self._pending = True
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
