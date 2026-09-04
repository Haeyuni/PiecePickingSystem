from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', description='NPZ scene path or filename in ~/Downloads'),
        DeclareLaunchArgument('results_dir', default_value='results', description='Output directory'),
        Node(
            package='grasp_benchmark',
            executable='compare_models',
            name='grasp_model_comparison',
            output='screen',
            arguments=['--scene', LaunchConfiguration('scene'), '--results-dir', LaunchConfiguration('results_dir')],
        ),
    ])
