import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    python = os.environ.get('GRASP_TEST_PYTHON', '')
    if not python:
        candidate = Path.cwd() / '.venv' / 'bin' / 'python'
        python = str(candidate) if candidate.is_file() else None
    return LaunchDescription([
        DeclareLaunchArgument('config'), DeclareLaunchArgument('methods', default_value='all'),
        DeclareLaunchArgument('rounds', default_value='3'), DeclareLaunchArgument('input_mode', default_value='live'),
        DeclareLaunchArgument('reset_mode', default_value='manual'), DeclareLaunchArgument('execute', default_value='false'),
        Node(package='grasp_test', executable='grasp_test', name='grasp_test', output='screen', prefix=python,
             arguments=['--config', LaunchConfiguration('config'), '--methods', LaunchConfiguration('methods'),
                        '--rounds', LaunchConfiguration('rounds'), '--input-mode', LaunchConfiguration('input_mode'),
                        '--reset-mode', LaunchConfiguration('reset_mode'), '--execute', LaunchConfiguration('execute')]),
    ])
