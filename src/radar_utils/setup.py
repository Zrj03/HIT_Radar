from setuptools import find_packages, setup

package_name = 'radar_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chenx',
    maintainer_email='chenx_dust@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'icp = radar_utils.icp:main',
            'record = radar_utils.record:main',
            'marker_pub = radar_utils.marker_pub:main',
            'selector = radar_utils.selector:main',
            'compressed_image_restamper = radar_utils.compressed_image_restamper:main',
            'compressed_image_decoder = radar_utils.compressed_image_decoder:main',
            'pointcloud_restamper = radar_utils.pointcloud_restamper:main',
            'bag_image_cloud_bridge = radar_utils.bag_image_cloud_bridge:main',
            'camera_ffmpeg_recorder = radar_utils.camera_ffmpeg_recorder:main',
            'camera_param_tuner = radar_utils.camera_param_tuner:main',
            'camera_extrinsic_tuner = radar_utils.camera_extrinsic_tuner:main',
        ],
    },
)
