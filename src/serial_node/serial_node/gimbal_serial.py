import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import serial
import struct
import math
from .ser_api import Get_CRC16_Check_Sum


class GimbalSerial(Node):
    def __init__(self):
        super().__init__('gimbal_serial')

        # 订阅无人机目标话题
        self.sub = self.create_subscription(
            PointStamped, '/radar/uav_target', self.callback, 10)

        # 声明串口参数
        self.declare_parameter('port', '/dev/radar')
        self.declare_parameter('baud', 115200)

        self.port_name = self.get_parameter('port').value
        self.baud_rate = self.get_parameter('baud').value

        # 尝试打开串口
        try:
            self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=0.01)
            self.get_logger().info(
                f'已打开云台串口: {self.port_name} 波特率: {self.baud_rate}')
        except Exception as e:
            self.get_logger().error(f'串口打开失败 {self.port_name}: {e}')
            self.ser = None

    def callback(self, msg: PointStamped):
        if not self.ser or not self.ser.is_open:
            return

        x = float(msg.point.x)
        y = float(msg.point.y)
        z = float(msg.point.z)

        # 计算 yaw / pitch
        yaw_rad = math.atan2(y, x)
        pitch_rad = math.atan2(z, math.hypot(x, y))

        # 计算延迟（秒）
        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        latency = (now - msg_time).nanoseconds / 1e9

        # 构建数据包
        # 协议: SOF(0xA5) + Header(0xA5) + Reserved(1B) + Yaw(4B) + Pitch(4B) + Latency(4B) + CRC16(2B) + TAIL(0xFE)
        sof = 0xA5
        header = 0xA5
        reserved = 0
        tail = 0xFE

        data_pack = struct.pack('<BBBfff', sof, header, reserved, yaw_rad, pitch_rad, latency)
        crc = Get_CRC16_Check_Sum(data_pack, len(data_pack))
        full_packet = data_pack + struct.pack('<HB', crc, tail)

        try:
            self.ser.write(full_packet)
            self.get_logger().info(
                f'Sent: yaw={yaw_rad:.2f}, pitch={pitch_rad:.2f}, lat={latency * 1000:.1f}ms')
        except Exception as e:
            self.get_logger().error(f'Serial write error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GimbalSerial()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
