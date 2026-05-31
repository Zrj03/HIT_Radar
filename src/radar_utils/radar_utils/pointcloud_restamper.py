import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class PointCloudRestamper(Node):
    def __init__(self):
        super().__init__('pointcloud_restamper')
        self.declare_parameter('input_topic', 'pc_raw_in')
        self.declare_parameter('output_topic', 'pc_raw')
        self.declare_parameter('frame_id', '')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(PointCloud2, output_topic, 10)
        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.get_logger().info(f'pointcloud restamp relay: {input_topic} -> {output_topic}')

    def callback(self, msg: PointCloud2):
        out = msg
        out.header.stamp = self.get_clock().now().to_msg()

        frame_id = self.get_parameter('frame_id').value
        if frame_id:
            out.header.frame_id = frame_id

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudRestamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
