from pathlib import Path

from setuptools import find_packages, setup

package_name = 'grasp_benchmark'
root = Path(__file__).parent
assets = []
for path in root.glob('launch/*.py'):
    assets.append((f'share/{package_name}/launch', [str(path)]))
for path in root.glob('docker/**/*'):
    if path.is_file():
        assets.append((f'share/{package_name}/{path.parent.relative_to(root)}', [str(path)]))
assets.append((f'share/ament_index/resource_index/packages', [f'resource/{package_name}']))
assets.append((f'share/{package_name}', ['package.xml', 'README.md']))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=assets,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jje',
    maintainer_email='jje320594@gmail.com',
    description='Independent Docker-based offline grasp model comparison',
    license='TODO',
    entry_points={'console_scripts': ['compare_models = grasp_benchmark.runner:main']},
)
