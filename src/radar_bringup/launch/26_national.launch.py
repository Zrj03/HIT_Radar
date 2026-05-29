from launch import LaunchDescription, actions
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from tf2_geometry_msgs.tf2_geometry_msgs import _get_quat_from_mat, _build_affine, _decompose_affine

from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, Node
from launch.actions import TimerAction, Shutdown
from launch import LaunchDescription

import numpy as np
import os
import re

debug = False

node_params = os.path.join(
    get_package_share_directory('radar_bringup'),
    'config',
    'config.26.national.yaml'
)


def get_xyzw_tf_broadcaster(cali: list, fr: str, child_fr: str):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace='radar',
        name=fr+'_to_'+child_fr,
        arguments=['--x', str(cali[0]),
                   '--y', str(cali[1]),
                   '--z', str(cali[2]),
                   '--qx', str(cali[3]),
                   '--qy', str(cali[4]),
                   '--qz', str(cali[5]),
                   '--qw', str(cali[6]),
                   '--frame-id', fr,
                   '--child-frame-id', child_fr],)


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
        name=fr+'_to_'+child_fr,
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
        condition=condition,)


def get_vision_container(cam_name: str):
    if not debug:
        return (ComposableNodeContainer(
            name=cam_name + '_vision_container',
            namespace='radar',
            package='rclcpp_components',
            executable='component_container_isolated',
            arguments=['--use_multi_threaded_executor'],
            composable_node_descriptions=[
                ComposableNode(
                    package='hik_camera',
                    plugin='hik_camera::HikCameraNode',
                    name='hik_camera',
                    namespace='radar/' + cam_name,
                    parameters=[node_params],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
            ],
            output='log',
            emulate_tty=True,
            on_exit=Shutdown(),
        ),)
    else:
        return (Node(
                package='hik_camera',
                executable='hik_camera_node',
                namespace='radar/' + cam_name,
                parameters=[node_params],
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
                ComposableNode(
                    package='livox_v1_lidar',
                    plugin='livox_v1_lidar::LidarPublisher',
                    name='livox_v1_lidar',
                    namespace='radar/' + 'lidar_mid70',
                    parameters=[node_params],
                    extra_arguments=[{'use_intra_process_comms': False}]
                ),
            ],
            output='log',
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
            ),
        )


def generate_launch_description():
    enable_judge_bridge = LaunchConfiguration('enable_judge_bridge')
    enable_gimbal_serial = LaunchConfiguration('enable_gimbal_serial')
    enable_camera_param_tuner = LaunchConfiguration('enable_camera_param_tuner')
    enable_camera_extrinsic_tuner = LaunchConfiguration('enable_camera_extrinsic_tuner')
    record_rosbag = LaunchConfiguration('record_rosbag')
    rosbag_output_dir = LaunchConfiguration('rosbag_output_dir')
    record_camera_video = LaunchConfiguration('record_camera_video')
    camera_video_output_dir = LaunchConfiguration('camera_video_output_dir')

    mvs_sdk_path = '/opt/MVS'
    mvs_lib_path = os.path.join(mvs_sdk_path, 'lib', '64')
    current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    combined_ld_library_path = (
        f'{mvs_lib_path}:{current_ld_library_path}'
        if current_ld_library_path else mvs_lib_path
    )

    return LaunchDescription([
        actions.SetEnvironmentVariable('MVCAM_SDK_PATH', mvs_sdk_path),
        actions.SetEnvironmentVariable('MVCAM_COMMON_RUNENV', os.path.join(mvs_sdk_path, 'lib')),
        actions.SetEnvironmentVariable('MVCAM_SOFTWARE_LIBENV', os.path.join(mvs_sdk_path, 'lib')),
        actions.SetEnvironmentVariable('MVCAM_GENICAM_CLPROTOCOL', os.path.join(mvs_sdk_path, 'lib', 'CLProtocol')),
        actions.SetEnvironmentVariable('ALLUSERSPROFILE', os.path.join(mvs_sdk_path, 'MVFG')),
        actions.SetEnvironmentVariable('LD_LIBRARY_PATH', combined_ld_library_path),
        DeclareLaunchArgument(
            'enable_judge_bridge',
            default_value='true',
            description='Whether to launch judge_bridge (requires /dev/Referee_System)'
        ),
        DeclareLaunchArgument(
            'enable_gimbal_serial',
            default_value='true',
            description='Whether to launch gimbal_serial (requires /dev/radar)'
        ),
        DeclareLaunchArgument(
            'enable_camera_param_tuner',
            default_value='true',
            description='Whether to launch camera_param_tuner GUI'
        ),
        DeclareLaunchArgument(
            'enable_camera_extrinsic_tuner',
            default_value='false',
            description='Launch T-DT style camera extrinsic tuner and disable static lidar->camera TF'
        ),
        DeclareLaunchArgument(
            'record_rosbag',
            default_value='false',
            description='Record a compact rosbag with the LiDAR point cloud topic'
        ),
        DeclareLaunchArgument(
            'rosbag_output_dir',
            default_value='/home/ajian/clear_radar-main/radar',
            description='Directory where rosbag2 timestamp folders will be created'
        ),
        DeclareLaunchArgument(
            'record_camera_video',
            default_value='false',
            description='Record the camera image topic to an mp4 file using ffmpeg'
        ),
        DeclareLaunchArgument(
            'camera_video_output_dir',
            default_value='/home/ajian/clear_radar-main/radar',
            description='Directory where camera mp4 files will be created'
        ),
        actions.ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '--compression-mode', 'message',
                '--compression-format', 'zstd',
                '-e',
                r'^/radar/lidar_mid70/pc_raw$'
            ],
            cwd=rosbag_output_dir,
            output='both',
            condition=IfCondition(record_rosbag),
        ),
        Node(
            package='radar_utils',
            executable='camera_ffmpeg_recorder',
            name='camera_ffmpeg_recorder',
            output='both',
            emulate_tty=True,
            condition=IfCondition(record_camera_video),
            parameters=[{
                'image_topic': '/radar/hik_6mm/image',
                'output_dir': camera_video_output_dir,
                'filename_prefix': 'national_camera',
                'fps': 30.0,
                'codec': 'h264_nvenc',
            }],
        ),
        *get_vision_container('hik_6mm'),
        Node(
            package='radar_utils',
            executable='camera_extrinsic_tuner',
            name='camera_extrinsic_tuner',
            namespace='radar',
            condition=IfCondition(enable_camera_extrinsic_tuner),
            parameters=[node_params],
            output='both',
        ),
        get_matrix_tf_broadcaster(
            load_lidar_to_camera_matrix(),
            'lidar_mid70_frame',
            'hik_6mm_frame',
            condition=UnlessCondition(enable_camera_extrinsic_tuner)
        ),
        *get_pc_container(),
        # TF broadcast handled by pc_aligner during alignment
        # Do not publish map->lidar_mid70_frame here to avoid TF loops
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'world', 'map'],
            output='log',
        ),

        # 手动标定节点：调整下面的x, y, z和yaw(单位:弧度)使其与地图对齐
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     namespace='radar',
        #     name='map_to_lidar_manual',
        #     arguments=['--x', '0.0', 
        #                '--y', '0.0', 
        #                '--z', '0.0', 
        #                '--yaw', '0.0', 
        #                '--pitch', '0.0', 
        #                '--roll', '0.0', 
        #                '--frame-id', 'map', 
        #                '--child-frame-id', 'lidar_mid70_frame'],
        # ),

        Node(
            package='radar_utils',
            executable='marker_pub',
            namespace='radar',
            parameters=[{"mesh": "rm_2026_19M.stl"}],
            output='log',
        ),
        Node(
            package='pc_aligner',
            executable='pc_aligner',
            namespace='radar',
            parameters=[node_params],
            output='both',
        ),
        Node(
            package='rm_radar_pipeline',
            executable='rm_radar_pipeline_node',
            name='rm_radar_pipeline',
            namespace='radar',
            parameters=[node_params],
            output='log',
        ),
        Node(
            package='target_multiplexer',
            executable='target_multiplexer',
            namespace='radar',
            parameters=[node_params],
            output='log',
        ),
        Node(
            package='dv_trigger',
            executable='dv_trigger',
            namespace='radar',
            parameters=[node_params],
            output='log',
        ),
        # judge_bridge 有裁判系统串口依赖，默认启用；需要关闭时传 enable_judge_bridge:=false
        Node(
            package='judge_bridge',
            executable='judge_bridge',
            name='judge_bridge',
            namespace='radar',
            condition=IfCondition(enable_judge_bridge),
            parameters=[node_params],
            output='log',
        ),

        # Foxglove bridge disabled.
        # Node(
        #     package='foxglove_bridge',
        #     executable='foxglove_bridge',
        # ),
        Node(
            package='target_visualizer',
            executable='rm_radar_image_visualizer',
            namespace='radar',
            output='log',
        ),
        Node(
            package='result_visualizer',
            executable='result_visualizer',
            namespace='radar',
            output='log',
        ),
        Node(
            package='radar_utils',
            executable='camera_param_tuner',
            name='camera_param_tuner',
            condition=IfCondition(enable_camera_param_tuner),
            output='log',
        ),
        # Gimbal Serial node：接收/radar/uav_target并通过串口控制云台
        Node(
            package='serial_node',
            executable='gimbal_serial',
            name='gimbal_serial',
            condition=IfCondition(enable_gimbal_serial),
            output='log',
            parameters=[
                {'port': '/dev/radar'},
                {'baud': 115200},
                {'target_y_offset': -0.5},
            ],
        ),
    ])
