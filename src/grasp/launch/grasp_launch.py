import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory('grasp'), 'config', 'grasp_params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value=default_config_path),
        DeclareLaunchArgument('strategy', default_value=''),
        Node(
            package='grasp',
            executable='grasp_node',
            name='grasp_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
                'strategy': LaunchConfiguration('strategy'),
            }],
        ),
    ])
