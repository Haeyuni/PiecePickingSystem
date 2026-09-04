"""perception 노드 기동.

**시스템 파이썬이 아니라 저장소의 `.venv`로 실행한다.** ultralytics는 apt로 오는 패키지가
아니라 venv에만 있고, 그 venv는 `--system-site-packages`로 만들어져 rclpy도 함께 보인다
(perception/README.md). `ros2 run`이 만드는 실행 스크립트는 시스템 파이썬 셔뱅을 달고
나오므로 그대로는 ultralytics를 찾지 못한다 — 그래서 인터프리터를 prefix로 지정한다.

컨테이너로 옮기면(시스템명세서 5.2절) 이 prefix는 필요 없어진다. 그때까지의 다리다.
"""
import os
import pathlib

from launch import LaunchDescription
from launch_ros.actions import Node


def venv_python() -> str:
    """저장소의 .venv 인터프리터. 없으면 빈 문자열(= prefix 없이 시스템 파이썬)."""
    override = os.environ.get("PERCEPTION_PYTHON")
    if override:
        return override
    try:
        from perception_common.paths import find_repo_path

        return str(find_repo_path(".venv", "bin", "python"))
    except Exception:
        return ""


def generate_launch_description():
    prefix = venv_python()
    return LaunchDescription([
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            prefix=prefix or None,
        ),
    ])
