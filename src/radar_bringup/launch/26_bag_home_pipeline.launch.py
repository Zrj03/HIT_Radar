from launch import LaunchDescription, actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from tf2_geometry_msgs.tf2_geometry_msgs import _decompose_affine

import numpy as np
import os
import re


node_params = os.path.join(
    get_package_share_directory('radar_bringup'),
    'config',
    'config.26.home.yaml'
)

default_bag_path = os.path.join(
    '/home/ajian/clear_radar-main',
    'radar',
    'rosbag2_2025_05_12-17_00_12_0.db3'
)


def load_lidar_to_camera_matrix():
    calibration_path = os.path.join(
        get_package_share_directory('radar_bringup'),
        'config',
        'calibration.yaml'
    )
    with open(calibration_path, 'r', encoding='utf-8') as f:
        text = f.read()
    match = re.search(r'lidar_to_camera:.*?data:\s*\[(.*?)\]', text, re.S)
    if not match:
        raise RuntimeError(f'Cannot find lidar_to_camera in {calibration_path}')
    values = [float(v) for v in re.split(r'[,\s]+', match.group(1).strip()) if v]
    if len(values) != 16:
        raise RuntimeError(f'lidar_to_camera in {calibration_path} must contain 16 values')
    return np.array(values, dtype=float).reshape((4, 4))


def get_matrix_tf_broadcaster(cali: np.array, fr: str, child_fr: str, condition=None):
    quat, trans = _decompose_affine(cali)
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace='radar',
        name=fr + '_to_' + child_fr,
        parameters=[{'use_sim_time': True}],
        arguments=['--x', str(trans[0]),
                   '--y', str(trans[1]),
                   '--z', str(trans[2]),
                   '--qw', str(quat[0]),
                   '--qx', str(quat[1]),
                   '--qy', str(quat[2]),
                   '--qz', str(quat[3]),
                   '--frame-id', fr,
                   '--child-frame-id', child_fr],
        output='log',
        condition=condition,
    )


def generate_launch_description():
    bag_path = LaunchConfiguration('bag_path')
    enable_foxglove = LaunchConfiguration('enable_foxglove')
    enable_visualizers = LaunchConfiguration('enable_visualizers')
    enable_camera_extrinsic_tuner = LaunchConfiguration('enable_camera_extrinsic_tuner')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_path',
            default_value=default_bag_path,
            description='DB3 rosbag file or rosbag directory to play'
        ),
        DeclareLaunchArgument(
            'enable_foxglove',
            default_value='true',
            description='Launch foxglove_bridge for offline visualization'
        ),
        DeclareLaunchArgument(
            'enable_visualizers',
            default_value='false',
            description='Launch OpenCV window visualizers'
        ),
        DeclareLaunchArgument(
            'enable_camera_extrinsic_tuner',
            default_value='false',
            description='Launch camera extrinsic tuner and disable static lidar->camera TF'
        ),
        actions.ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', bag_path,
                '--clock',
                '--remap',
                '/livox/lidar:=/radar/lidar_mid70/pc_raw_bag',
                '/image_raw/compressed:=/radar/hik_6mm/image/compressed_raw',
            ],
            output='screen',
        ),
        Node(
            package='radar_utils',
            executable='bag_image_cloud_bridge',
            parameters=[
                {'use_sim_time': True},
                {'compressed_image_topic': '/radar/hik_6mm/image/compressed_raw'},
                {'pointcloud_topic': '/radar/lidar_mid70/pc_raw_bag'},
                {'image_output_topic': '/radar/hik_6mm/image'},
                {'pointcloud_output_topic': '/radar/lidar_mid70/pc_raw'},
                {'camera_frame_id': 'hik_6mm_frame'},
                {'lidar_frame_id': 'lidar_mid70_frame'},
            ],
            output='log',
        ),
        Node(
            package='hik_camera',
            executable='info_pub',
            namespace='radar/hik_6mm',
            parameters=[
                {'use_sim_time': True},
                node_params,
            ],
            output='log',
        ),
        get_matrix_tf_broadcaster(
            load_lidar_to_camera_matrix(),
            'hik_6mm_frame',
            'lidar_mid70_frame',
            condition=UnlessCondition(enable_camera_extrinsic_tuner)
        ),
        Node(
            package='radar_utils',
            executable='camera_extrinsic_tuner',
            name='camera_extrinsic_tuner',
            namespace='radar',
            condition=IfCondition(enable_camera_extrinsic_tuner),
            parameters=[
                node_params,
                {'use_sim_time': True},
                {'image_topic': '/radar/hik_6mm/image'},
                {'compressed_image_topic': '/radar/hik_6mm/image/compressed_raw'},
                {'camera_info_topic': '/radar/hik_6mm/camera_info'},
                {'world_frame': 'world'},
                {'lidar_frame': 'lidar_mid70_frame'},
                {'camera_frame': 'hik_6mm_frame'},
                {'calibration_file': '/home/ajian/clear_radar-main/src/radar_bringup/config/calibration.yaml'},
                {'use_compressed_image': False},
                {'auto_save': True},
                {'broadcast_tf': True},
            ],
            output='both',
        ),
        Node(
            package='pc_aligner',
            executable='pc_aligner',
            namespace='radar',
            parameters=[
                node_params,
                {'use_sim_time': True},
                {'startup_manual_align': False},
            ],
            output='both',
        ),
        Node(
            package='rm_radar_pipeline',
            executable='rm_radar_pipeline_node',
            name='rm_radar_pipeline',
            namespace='radar',
            parameters=[
                node_params,
                {'use_sim_time': True},
                {'sync_queue_size': 20},
            ],
            output='both',
        ),
        Node(
            package='target_multiplexer',
            executable='target_multiplexer',
            namespace='radar',
            parameters=[node_params, {'use_sim_time': True}],
            output='log',
        ),
        Node(
            package='target_visualizer',
            executable='rm_radar_image_visualizer',
            namespace='radar',
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(enable_visualizers),
            output='log',
        ),
        Node(
            package='result_visualizer',
            executable='result_visualizer',
            namespace='radar',
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(enable_visualizers),
            output='log',
        ),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            condition=IfCondition(enable_foxglove),
            output='log',
        ),
    ])
