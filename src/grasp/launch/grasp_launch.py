from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='grasp',
            executable='grasp_node',
            name='grasp_node',
            output='screen',
        ),
    ])
