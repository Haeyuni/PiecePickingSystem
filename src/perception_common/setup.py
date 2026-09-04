from setuptools import find_packages, setup

package_name = 'perception_common'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jje',
    maintainer_email='jje320594@gmail.com',
    description='perception/grasp 공용: 좌표 변환(geometry), 이미지 변환(image_utils), '
                 '저장소 경로 탐색(paths), TCP 자세 조회(robot_pose)',
    license='TODO',
    tests_require=['pytest'],
)
