import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jje',
    maintainer_email='jje320594@gmail.com',
    description='ROS2 스킬 실행기(pick/place_into) 및 순응제어',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pick_server = control.pick_server:main',
            'place_server = control.place_server:main',
            'robot_state_publisher_node = control.robot_state_publisher:main',
            'safety_monitor = control.safety_monitor:main',
        ],
    },
)
