from setuptools import find_packages, setup

package_name = 'bpp_orchestrator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lhs',
    maintainer_email='lhs@example.com',
    description='Packee BPP 서비스를 처리하는 ROS 2 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bpp_node = bpp_orchestrator.bpp_node:main',
            'bpp_complete_mock = bpp_orchestrator.mock_complete_server:main',
            'packee_main = bpp_orchestrator.packee_main:main',
        ],
    },
)
