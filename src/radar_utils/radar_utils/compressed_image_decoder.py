import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class CompressedImageDecoder(Node):
    def __init__(self):
        super().__init__('compressed_image_decoder')
        self.declare_parameter('input_topic', 'image/compressed')
        self.declare_parameter('output_topic', 'image')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('restamp', False)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(Image, output_topic, 10)
        self.sub = self.create_subscription(CompressedImage, input_topic, self.callback, 10)
        self.get_logger().info(f'compressed decode relay: {input_topic} -> {output_topic}')

    def callback(self, msg: CompressedImage):
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn('failed to decode compressed image')
            return

        out = Image()
        out.header = msg.header
        if self.get_parameter('restamp').value:
            out.header.stamp = self.get_clock().now().to_msg()

        frame_id = self.get_parameter('frame_id').value
        if frame_id:
            out.header.frame_id = frame_id

        out.height = image.shape[0]
        out.width = image.shape[1]
        out.encoding = 'bgr8'
        out.is_bigendian = False
        out.step = image.shape[1] * image.shape[2]
        out.data = image.tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
