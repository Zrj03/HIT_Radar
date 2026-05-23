import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RmRadarImageVisualizer(Node):
    def __init__(self):
        super().__init__('rm_radar_image_visualizer')
        self.declare_parameter('image_topic', 'rm_radar_pipeline/visualization')
        self.declare_parameter('window_name', 'rm_radar_pipeline')
        self.declare_parameter('im_show', True)
        self.image_topic = self.get_parameter('image_topic').value
        self.window_name = self.get_parameter('window_name').value
        self.window_created = False
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 1)
        self.get_logger().info(f'Visualizing {self.image_topic} in window {self.window_name}')

    def image_callback(self, msg: Image):
        if not self.get_parameter('im_show').value:
            return
        try:
            image = self.msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return

        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    @staticmethod
    def msg_to_bgr(msg: Image):
        if msg.encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif msg.encoding == 'mono8':
            channels = 1
        else:
            raise ValueError(f'unsupported encoding: {msg.encoding}')

        if msg.height == 0 or msg.width == 0:
            raise ValueError('empty image shape')
        min_step = msg.width * channels
        if msg.step < min_step:
            raise ValueError('invalid image step')

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))[:, :min_step]
        if channels == 3:
            image = rows.reshape((msg.height, msg.width, 3))
            if msg.encoding == 'rgb8':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = image.copy()
        else:
            image = rows.reshape((msg.height, msg.width))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image


def main():
    rclpy.init()
    node = RmRadarImageVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
