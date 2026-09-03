import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'grasp'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jje',
    maintainer_email='jje320594@gmail.com',
    description='포인트클라우드 기반 파지 자세 추정, world_state 최종 발행',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_node = grasp.node:main',
        ],
    },
)
