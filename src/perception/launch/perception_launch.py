"""perception 노드 기동.

**시스템 파이썬이 아니라 저장소의 `.venv`로 실행한다.** ultralytics는 apt로 오는 패키지가
아니라 venv에만 있고, 그 venv는 `--system-site-packages`로 만들어져 rclpy도 함께 보인다
(perception/README.md). `ros2 run`이 만드는 실행 스크립트는 시스템 파이썬 셔뱅을 달고
나오므로 그대로는 ultralytics를 찾지 못한다 — 그래서 인터프리터를 prefix로 지정한다.

컨테이너로 옮기면(시스템명세서 5.2절) 이 prefix는 필요 없어진다. 그때까지의 다리다.

**파라미터를 launch 인자로 받는다.** 예전에는 이 파일에 `parameters=[...]`가 아예 없어서
`ros2 launch`로는 `detector`를 못 바꾸고 `ros2 run ... --ros-args -p detector:=vlm_sam`로만
가능했다(온디맨드 전환 착수 시점 지적, docs/on-demand-perception.md 5절). 문자열 파라미터만
여기서 노출한다 — 숫자 파라미터를 LaunchConfiguration으로 넘기면 문자열로 들어가 노드의
`float(...)/int(...)` 캐스팅과 부딪힌다(grasp_launch.py가 config_path/strategy만 노출하는
것과 같은 이유). 숫자를 조정하려면 `-p` 오버라이드나 파라미터 YAML을 쓴다
(perception/README.md).
"""
import os
import pathlib

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
        # node.py의 declare_parameter 기본값과 같다 — 여기서 바꾸지 않으면 그 값 그대로 쓴다.
        # detector만 PERCEPTION_DETECTOR 환경변수로도 바꿀 수 있다 — Dockerfile의 CMD가
        # `ros2 launch ...`를 고정 인자로 부르므로(docker-entrypoint.sh), compose가 launch
        # 인자를 직접 건드리지 않고도 detector를 고르게 하려면 이 경로가 필요하다. 다른
        # detector=vlm_sam 전용 값들(SAM_MODEL/PLANNER_URL)은 node.py가 이미 env_var로
        # 읽는다(_build_detector 참조) — 여기서 다시 다리를 놓을 필요가 없다.
        DeclareLaunchArgument(
            'detector', default_value=os.environ.get('PERCEPTION_DETECTOR', 'vlm_sam')),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('sam_model_path', default_value=''),
        DeclareLaunchArgument('planner_url', default_value=''),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('color_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument(
            'depth_topic', default_value='/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('info_topic', default_value='/camera/color/camera_info'),
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            prefix=prefix or None,
            parameters=[{
                'detector': LaunchConfiguration('detector'),
                'model_path': LaunchConfiguration('model_path'),
                'sam_model_path': LaunchConfiguration('sam_model_path'),
                'planner_url': LaunchConfiguration('planner_url'),
                'device': LaunchConfiguration('device'),
                'color_topic': LaunchConfiguration('color_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'info_topic': LaunchConfiguration('info_topic'),
            }],
        ),
    ])
