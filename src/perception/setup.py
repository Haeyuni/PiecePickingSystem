import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'perception'

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
    description='RGB-D 손목 카메라 기반 물체 검출·세그멘테이션, world_state_raw/instance_masks 발행',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception.node:main',
            # 검출 모델 준비 전까지 /world_state 발행자 역할 (개발계획 B4)
            'fake_world_publisher = perception.fake_world_publisher:main',
        ],
    },
)
