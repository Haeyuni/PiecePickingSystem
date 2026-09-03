from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """control 노드 4종을 한 프로세스로 띄운다.

    각각 별도 프로세스로 띄우면 robot_state의 단일 소유자(store)를 공유할 수 없다 —
    상세는 control/control/main.py 주석 참조.
    """
    return LaunchDescription([
        Node(package='control', executable='control_node', name='control_node',
             output='screen'),
    ])
