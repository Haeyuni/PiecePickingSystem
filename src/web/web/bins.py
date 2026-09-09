"""목적지(bin)의 한국어 이름 조회.

control/place_server.py가 쓰는 것과 같은 bins.yaml을 읽지만, web은 좌표·geometry는
전혀 쓰지 않는다(그건 control 고유 책임) — 화면에 보여줄 name_ko만 뽑아 쓴다
(TaskProgress.tsx가 bin_id 대신 이 이름을 쓴다).
"""
import functools
import logging
import os
import pathlib

import yaml

logger = logging.getLogger(__name__)

BINS_CONFIG_PATH = pathlib.Path(os.environ.get("BINS_CONFIG_PATH", "/config/bins.yaml"))


@functools.lru_cache(maxsize=1)
def _bin_names() -> dict[str, str]:
    # 한 번 읽으면 캐시한다 — bins.yaml은 캘리브레이션 때만 바뀌고, 그때는 컨테이너를
    # 재기동하므로(docker-compose.yml 주석 참조) 캐시가 오래된 값을 들고 있을 일이 없다.
    try:
        data = yaml.safe_load(BINS_CONFIG_PATH.read_text()) or {}
    except OSError as e:
        logger.warning("bins.yaml을 읽지 못했습니다 (%s) — 화면에 bin_id를 그대로 보여준다", e)
        return {}
    return {
        bin_id: cfg.get("name_ko", bin_id)
        for bin_id, cfg in (data.get("bins") or {}).items()
    }


def name_ko(bin_id: str | None) -> str | None:
    """bin_id의 한국어 이름. 설정에 없거나 못 읽었으면 None — 호출부가 bin_id로 대체한다."""
    if not bin_id:
        return None
    return _bin_names().get(bin_id)
