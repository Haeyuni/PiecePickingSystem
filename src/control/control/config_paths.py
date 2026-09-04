"""control 패키지 설정 파일 경로 탐색. bins.yaml은 place_server.py가 따로 갖는다
(그 파일만 쓰는 상대적으로 독립된 개념이라) — skill_params.yaml은 home/pick/place
셋 다 읽어야 해서 여기로 뺐다.
"""
import os
import pathlib


def skill_params_path() -> pathlib.Path:
    """skill_params.yaml 위치 (bins_yaml_path와 같은 탐색 순서: 설치본 → 소스 트리)."""
    override = os.environ.get("SKILL_PARAMS_YAML")
    if override:
        return pathlib.Path(override)
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = pathlib.Path(get_package_share_directory("control")) / "config" / "skill_params.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    return pathlib.Path(__file__).resolve().parents[1] / "config" / "skill_params.yaml"
