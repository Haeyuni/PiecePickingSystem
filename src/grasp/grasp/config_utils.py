"""ROS runtime 없이도 확인 가능한 grasp 설정 경로 도우미."""
import os
import pathlib


def asset_path(assets: dict, name: str) -> pathlib.Path | None:
    """assets_dir 상대 경로 또는 절대 경로를 환경변수까지 펼쳐 해석한다."""
    value = str(assets.get(name, "") or "")
    if not value:
        return None
    path = pathlib.Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    root = os.path.expandvars(str(assets.get("assets_dir", "") or ""))
    return pathlib.Path(root).expanduser() / path if root else path
