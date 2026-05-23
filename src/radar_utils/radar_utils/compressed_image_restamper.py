import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class CompressedImageRestamper(Node):
    def __init__(self):
        super().__init__('compressed_image_restamper')
        self.declare_parameter('input_topic', 'image/compressed_raw')
        self.declare_parameter('output_topic', 'image/compressed')
        self.declare_parameter('frame_id', '')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.pub = self.create_publisher(CompressedImage, output_topic, 10)
        self.sub = self.create_subscription(CompressedImage, input_topic, self.callback, 10)

        self.get_logger().info(f'restamp relay: {input_topic} -> {output_topic}')

    def callback(self, msg: CompressedImage):
        out = CompressedImage()
        out = msg
        out.header.stamp = self.get_clock().now().to_msg()

        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        if frame_id:
            out.header.frame_id = frame_id

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageRestamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
