import copy

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2


class BagImageCloudBridge(Node):
    def __init__(self):
        super().__init__('bag_image_cloud_bridge')
        self.declare_parameter('compressed_image_topic', '/radar/hik_6mm/image/compressed_raw')
        self.declare_parameter('pointcloud_topic', '/radar/lidar_mid70/pc_raw_bag')
        self.declare_parameter('image_output_topic', '/radar/hik_6mm/image')
        self.declare_parameter('pointcloud_output_topic', '/radar/lidar_mid70/pc_raw')
        self.declare_parameter('camera_frame_id', 'hik_6mm_frame')
        self.declare_parameter('lidar_frame_id', 'lidar_mid70_frame')

        compressed_image_topic = self.get_parameter('compressed_image_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        image_output_topic = self.get_parameter('image_output_topic').value
        pointcloud_output_topic = self.get_parameter('pointcloud_output_topic').value

        self.latest_cloud = None
        self.image_pub = self.create_publisher(Image, image_output_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, pointcloud_output_topic, 10)
        self.cloud_sub = self.create_subscription(PointCloud2, pointcloud_topic, self.cloud_callback, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, compressed_image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'bag bridge: {compressed_image_topic} + {pointcloud_topic} -> '
            f'{image_output_topic} + {pointcloud_output_topic}')

    def cloud_callback(self, msg: PointCloud2):
        self.latest_cloud = msg

    def image_callback(self, msg: CompressedImage):
        if self.latest_cloud is None:
            return

        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn('failed to decode compressed image')
            return

        stamp = self.get_clock().now().to_msg()

        image_msg = Image()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.get_parameter('camera_frame_id').value
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.encoding = 'bgr8'
        image_msg.is_bigendian = False
        image_msg.step = image.shape[1] * image.shape[2]
        image_msg.data = image.tobytes()

        cloud_msg = copy.deepcopy(self.latest_cloud)
        cloud_msg.header.stamp = stamp
        cloud_msg.header.frame_id = self.get_parameter('lidar_frame_id').value

        self.cloud_pub.publish(cloud_msg)
        self.image_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BagImageCloudBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
