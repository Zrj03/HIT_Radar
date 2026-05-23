from launch import LaunchDescription, actions
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from tf2_geometry_msgs.tf2_geometry_msgs import _get_quat_from_mat, _build_affine, _decompose_affine

from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch.actions import TimerAction, Shutdown
from launch import LaunchDescription

import numpy as np
import os

debug = True
use_real_lidar = LaunchConfiguration('use_real_lidar')
bag_path = LaunchConfiguration('bag_path')

node_params = os.path.join(
    get_package_share_directory('radar_bringup'),
    'config',
    'config.26.national.yaml'
)

default_bag_path = os.path.join(
    '/home/ajian/radar_ros_ws-main',
    'radar',
    'rosbag2_2025_05_12-17_02_26'
)


def get_xyzw_tf_broadcaster(cali: list, fr: str, child_fr: str):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace='radar',
        name=fr+'_to_'+child_fr,
        parameters=[{'use_sim_time': True}],
        arguments=['--x', str(cali[0]),
                   '--y', str(cali[1]),
                   '--z', str(cali[2]),
                   '--qx', str(cali[3]),
                   '--qy', str(cali[4]),
                   '--qz', str(cali[5]),
                   '--qw', str(cali[6]),
                   '--frame-id', fr,
                   '--child-frame-id', child_fr],)


def get_matrix_tf_broadcaster(cali: np.array, fr: str, child_fr: str):
    quat, trans = _decompose_affine(cali)
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace='radar',
        name=fr+'_to_'+child_fr,
        parameters=[{'use_sim_time': True}],
        arguments=['--x', str(trans[0]),
                   '--y', str(trans[1]),
                   '--z', str(trans[2]),
                   '--qw', str(quat[0]),
                   '--qx', str(quat[1]),
                   '--qy', str(quat[2]),
                   '--qz', str(quat[3]),
                   '--frame-id', fr,
                   '--child-frame-id', child_fr],)


def get_vision_container(cam_name: str):
    if not debug:
        return (ComposableNodeContainer(
            name=cam_name + '_vision_container',
            namespace='radar',
            package='rclcpp_components',
            executable='component_container_isolated',
            arguments=['--use_multi_threaded_executor'],
            composable_node_descriptions=[
                # ComposableNode(
                #     package='hik_camera',
                #     plugin='hik_camera::HikCameraNode',
                #     name='hik_camera',
                #     namespace='radar/' + cam_name,
                #     parameters=[node_params],
                #     extra_arguments=[{'use_intra_process_comms': True}]
                # ),
                ComposableNode(
                    package='img_recognizer',
                    plugin='img_recognizer::RecognizerNode',
                    name='img_recognizer',
                    namespace='radar/' + cam_name,
                    parameters=[
                        node_params,
                        {'use_sim_time': True,
                         'img_compressed': True,
                         'sync_max_interval': 0.25},
                        ],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
            ],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown(),
        ),)
    else:
        return (
            Node(
                package='radar_utils',
                executable='compressed_image_restamper',
                namespace='radar/' + cam_name,
                parameters=[
                    {'use_sim_time': True,
                     'input_topic': 'image/compressed_raw',
                     'output_topic': 'image/compressed',
                     'frame_id': cam_name + '_frame'},
                ],
            ),
            Node(
                package='hik_camera',
                executable='info_pub',
                namespace='radar/' + cam_name,
                parameters=[
                    {'use_sim_time': True},
                    node_params,
                ],
            ),
            Node(
                package='img_recognizer',
                executable='img_recognizer_node',
                namespace='radar/' + cam_name,
                parameters=[
                    node_params,
                    {'use_sim_time': True,
                     'img_compressed': True,
                     'sync_max_interval': 0.25},
                ],
            ),
        )


def get_pc_container():
    if not debug:
        return (ComposableNodeContainer(
            name='pc_container',
            namespace='radar',
            package='rclcpp_components',
            executable='component_container_isolated',
            arguments=['--use_multi_threaded_executor'],
            composable_node_descriptions=[
                # ComposableNode(
                #     package='livox_v1_lidar',
                #     plugin='livox_v1_lidar::LidarPublisher',
                #     name='livox_v1_lidar',
                #     namespace='radar/' + 'lidar_mid70',
                #     parameters=[node_params],
                #     extra_arguments=[{'use_intra_process_comms': True}]
                # ),
                ComposableNode(
                    package='pc_detector',
                    plugin='pc_detector::DetectorNode',
                    name='pc_detector',
                    namespace='radar',
                    parameters=[node_params,
                                {'use_sim_time': True}],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='nn_detector',
                    plugin='nn_detector::DetectorNode',
                    name='nn_detector',
                    namespace='radar',
                    parameters=[node_params,
                                {'use_sim_time': True}],
                    remappings=[
                        ('lidar_mid70/livox/pointcloud', 'lidar_mid70/pc_raw'),
                    ],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
            ],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown(),
        ),)
    else:
        return (
            Node(
                package='livox_v1_lidar',
                executable='livox_v1_lidar_node',
                name='livox_v1_lidar',
                namespace='radar/' + 'lidar_mid70',
                parameters=[node_params],
                condition=IfCondition(use_real_lidar),
            ),
            Node(
                package='pc_detector',
                executable='pc_detector_node',
                name='pc_detector',
                namespace='radar',
                parameters=[node_params,
                            {'use_sim_time': True}],
            ),
            Node(
                package='nn_detector',
                executable='nn_detector_node',
                name='nn_detector',
                namespace='radar',
                parameters=[node_params,
                            {'use_sim_time': True}],
                remappings=[
                    ('lidar_mid70/livox/pointcloud', 'lidar_mid70/pc_raw'),
                ],
            ),
        )


def generate_launch_description():
    startup_manual_align = LaunchConfiguration('startup_manual_align')
    default_team_color = LaunchConfiguration('default_team_color')
    return LaunchDescription([
        DeclareLaunchArgument(
            'startup_manual_align',
            default_value='true',
            description='Whether to run manual alignment on startup in bag mode'
        ),
        DeclareLaunchArgument(
            'use_real_lidar',
            default_value='false',
            description='Whether to start the live Livox driver alongside bag playback'
        ),
        DeclareLaunchArgument(
            'default_team_color',
            default_value='blue',
            description='Default team color when judge_bridge is not running'
        ),
        DeclareLaunchArgument(
            'bag_path',
            default_value=default_bag_path,
            description='Rosbag directory to play back for national offline testing'
        ),
        actions.ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', bag_path,
                '--clock',
                '--remap',
                '/livox/lidar:=/radar/lidar_mid70/pc_raw',
                '/image_raw/compressed:=/radar/hik_6mm/image/compressed_raw',
            ],
            output='screen',
        ),
        *get_vision_container('hik_6mm'),
        get_xyzw_tf_broadcaster(
            [
                0.059292495250701904,
                -0.10663162916898727,
                0.004948855843394995,
                0.49931320973839743,
                0.49561537544531076,
                0.4977604777199716,
                0.5072338976832756
            ], 'lidar_mid70_frame', 'hik_6mm_frame'
        ),
        *get_pc_container(),
        get_xyzw_tf_broadcaster(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            'world',
            'map'
        ),
        get_xyzw_tf_broadcaster(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            'lidar_mid70_frame',
            'livox_frame'
        ),


        Node(
            package='radar_utils',
            executable='marker_pub',
            namespace='radar',
            parameters=[{"mesh": "rm_2026_19M.stl"}],
            output='both',
        ),
        Node(
            package='pc_aligner',
            executable='pc_aligner',
            namespace='radar',
            parameters=[node_params,
                        {'startup_manual_align': startup_manual_align},
                        {'use_sim_time': True}],
            output='both',
        ),
        Node(
            package='target_matcher',
            executable='target_matcher',
            namespace='radar',
            parameters=[node_params,
                        {'use_sim_time': True}],
            output='both',
        ),
        # Node(
        #     package='target_multiplexer',
        #     executable='target_multiplexer',
        #     namespace='radar',
        #     output='both',
        # ),
        Node(
            package='dv_trigger',
            executable='dv_trigger',
            namespace='radar',
            parameters=[node_params,
                        {'use_sim_time': True}],
            output='both',
        ),
        Node(
            package='radar_supervisor',
            executable='radar_supervisor',
            namespace='radar',
            parameters=[node_params,
                        {'use_sim_time': True}],
            output='both',
        ),
        # Node(
        #     package='judge_bridge',
        #     executable='judge_bridge',
        #     name='judge_bridge',
        #     namespace='radar',
        #     parameters=[node_params,
        #                 {'use_sim_time': True}],
        #     output='both',
        # ),
        # Foxglove bridge disabled.
        # Node(
        #     package='foxglove_bridge',
        #     executable='foxglove_bridge',
        # ),
        Node(
            package='target_visualizer',
            executable='target_visualizer',
            namespace='radar',
            output='both',
        ),
        Node(
            package='result_visualizer',
            executable='result_visualizer',
            namespace='radar',
            parameters=[{'default_team_color': default_team_color}],
            output='both',
        ),
        # Gimbal Serial node：接收/radar/uav_target并通过串口控制云台
        # Node(
        #     package='serial_node',
        #     executable='gimbal_serial',
        #     name='gimbal_serial',
        #     output='both',
        #     parameters=[
        #         {'port': '/dev/ttyACM0'},
        #         {'baud': 115200},
        #         {'use_sim_time': True},
        #     ],
        # ),
    ])
