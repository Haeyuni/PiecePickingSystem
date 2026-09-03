from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='control', executable='pick_server', name='pick_server', output='screen'),
        Node(package='control', executable='place_server', name='place_server', output='screen'),
        Node(package='control', executable='robot_state_publisher_node',
             name='robot_state_publisher_node', output='screen'),
        Node(package='control', executable='safety_monitor', name='safety_monitor', output='screen'),
    ])
