"""저장소 안의 데이터 파일 찾기 (`data/mock`, `model/best.pt` 등).

패키지 밖에 있는 저장소 데이터는 colcon이 설치해 주지 않는다. `__file__` 상대경로만 쓰면
**설치본에서 조용히 깨진다** — 설치되면 이 파일은 `install/perception/lib/python3.12/
site-packages/`로 복사되는데 데이터는 소스 트리에 남아 있기 때문이다.

굽지 않고 찾아 쓰는 이유는 마이그레이션·시드와 같다: 픽스처나 가중치 한 번 바꾸려고
colcon 빌드를 다시 돌릴 이유가 없다.
"""
import logging
import os
import pathlib

logger = logging.getLogger(__name__)


def find_repo_path(*parts: str, env_var: str | None = None) -> pathlib.Path:
    """저장소 기준 상대경로를 실제 경로로 바꾼다.

    순서: 환경변수 → 소스 트리(조상 탐색) → colcon 워크스페이스의 src/*.
    못 찾으면 무엇을 지정하면 되는지 적어 예외를 올린다 — FileNotFoundError가 pathlib
    안쪽에서 튀어나오는 것보다 고치기 쉽다.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            path = pathlib.Path(override)
            if path.exists():
                return path
            # 같은 .env가 컨테이너와 호스트를 함께 섬긴다. 컨테이너 경로(/data/...)를 호스트에서
            # 읽으면 여기 걸린다 — 설정을 무시하되 조용히 넘어가지는 않는다. 캘리브레이션처럼
            # 안전에 직결되는 값일수록 "다른 파일을 읽었다"는 사실이 보여야 한다.
            logger.warning("%s=%s 가 존재하지 않아 저장소에서 다시 찾습니다", env_var, override)

    relative = pathlib.Path(*parts)
    here = pathlib.Path(__file__).resolve()

    for parent in here.parents:                       # 소스에서 직접 실행
        candidate = parent / relative
        if candidate.exists():
            return candidate

    for parent in here.parents:                       # colcon 설치본
        if parent.name == "install":
            workspace_path = parent.parent / relative
            if workspace_path.exists():
                return workspace_path
            break

    hint = f"{env_var} 환경변수" if env_var else "절대경로 파라미터"
    raise RuntimeError(f"저장소에서 '{relative}'를 찾지 못했습니다. {hint}로 지정하세요.")
