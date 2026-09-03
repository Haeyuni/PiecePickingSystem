"""request_id 중복 실행 방지 캐시 (인터페이스_정의서.md 1절).

`request_id`는 "재전송 시 같은 ID를 사용해 중복 실행 방지"를 위한 필드지만, **필드가 있다고
방지되지는 않는다** — 받는 쪽이 실제로 기억하고 있어야 한다. 그 기억을 여기서 담당한다.

fake 모드와 실물 모드가 같은 캐시를 쓴다. 중복 방지는 로봇 유무와 무관한 계약이다.
"""
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    result: Any
    stored_at: float


class RequestCache:
    """최근 처리한 request_id의 결과를 짧게 들고 있는다.

    영속 저장이 아니라 짧은 캐시인 이유: 재전송은 통신 실패 직후에 일어나므로 수십 초면
    충분하고, 그보다 오래된 같은 ID는 재전송이 아니라 새 요청으로 보는 편이 안전하다.
    """

    def __init__(self, ttl_s: float = 60.0, max_entries: int = 256):
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._entries: dict[str, _Entry] = {}

    def get(self, request_id: str) -> Any | None:
        self._evict()
        entry = self._entries.get(request_id)
        return entry.result if entry else None

    def put(self, request_id: str, result: Any) -> None:
        if not request_id:
            return  # 빈 request_id는 추적 대상이 아니다
        self._entries[request_id] = _Entry(result=result, stored_at=time.monotonic())
        self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._entries.items() if now - v.stored_at > self._ttl_s]
        for key in expired:
            del self._entries[key]
        # 상한 초과 시 오래된 것부터 버린다
        if len(self._entries) > self._max_entries:
            ordered = sorted(self._entries.items(), key=lambda kv: kv[1].stored_at)
            for key, _ in ordered[: len(self._entries) - self._max_entries]:
                del self._entries[key]
