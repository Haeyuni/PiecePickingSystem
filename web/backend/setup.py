from setuptools import find_packages, setup

package_name = 'web'

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
    description='FastAPI 명령 입력·모니터링 UI, rclpy 브리지',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'web_main = web.main:main',
        ],
    },
)
