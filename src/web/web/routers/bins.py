"""GET /api/bins — 목적지(bin) 목록. 승인 화면의 목적지 수정 드롭다운이 쓴다.

값 자체는 bins.yaml(control/place_server.py와 같은 설정)에서 오고, web은 이름만 읽는다
(web/bins.py 모듈 docstring 참조).
"""
from fastapi import APIRouter

from .. import bins

router = APIRouter()


@router.get("/api/bins")
def list_bins():
    return {"schema_version": "1.0.0", "bins": bins.all_bins()}
