import os
import signal
import subprocess
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraFfmpegRecorder(Node):
    def __init__(self):
        super().__init__('camera_ffmpeg_recorder')
        self.declare_parameter('image_topic', '/radar/hik_6mm/image')
        self.declare_parameter('output_dir', '/home/ajian/clear_radar-main/radar')
        self.declare_parameter('filename_prefix', 'camera_video')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('codec', 'h264_nvenc')
        self.declare_parameter('crf', 23)

        self.proc = None
        self.frame_shape = None
        self.output_path = None
        self.active_codec = None
        self.fallback_used = False
        self.frame_count = 0
        topic = self.get_parameter('image_topic').value
        self.sub = self.create_subscription(
            Image, topic, self.image_callback, qos_profile_sensor_data)
        self.no_frame_timer = self.create_timer(5.0, self.warn_if_no_frames)
        self.get_logger().info(
            f'Camera ffmpeg recorder subscribed to {topic} with sensor_data QoS')

    def image_callback(self, msg: Image):
        self.frame_count += 1
        try:
            frame = self.msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'Skip frame: {exc}')
            return

        height, width = frame.shape[:2]
        if self.proc is None:
            self.start_ffmpeg(width, height)
        elif self.frame_shape != (height, width):
            self.get_logger().warn(
                f'Skip frame with changed size {width}x{height}; recording size is '
                f'{self.frame_shape[1]}x{self.frame_shape[0]}')
            return

        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            self.get_logger().error(f'ffmpeg pipe closed: {exc}')
            self.stop_ffmpeg()
            if self.active_codec != 'libx264' and not self.fallback_used:
                self.fallback_used = True
                self.get_logger().warn('Fallback to libx264 camera recording.')
                self.start_ffmpeg(width, height, codec_override='libx264')
                self.proc.stdin.write(frame.tobytes())

    def warn_if_no_frames(self):
        if self.frame_count == 0:
            topic = self.get_parameter('image_topic').value
            self.get_logger().warn(
                f'No camera frames received yet on {topic}; check topic name, camera node, and QoS')

    def start_ffmpeg(self, width: int, height: int, codec_override=None):
        output_dir = self.get_parameter('output_dir').value
        prefix = self.get_parameter('filename_prefix').value
        fps = float(self.get_parameter('fps').value)
        codec = codec_override or self.get_parameter('codec').value
        crf = int(self.get_parameter('crf').value)
        os.makedirs(output_dir, exist_ok=True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        codec_tag = 'x264' if codec == 'libx264' else codec
        self.output_path = os.path.join(output_dir, f'{prefix}_{stamp}_{codec_tag}.mp4')
        cmd = [
            'ffmpeg',
            '-y',
            '-loglevel', 'warning',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{fps:g}',
            '-i', '-',
            '-an',
            '-vcodec', codec,
        ]
        if codec == 'h264_nvenc':
            cmd += ['-preset', 'p4', '-cq', str(crf)]
        else:
            cmd += ['-preset', 'veryfast', '-crf', str(crf)]
        cmd += ['-pix_fmt', 'yuv420p', self.output_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frame_shape = (height, width)
        self.active_codec = codec
        self.get_logger().info(f'Recording camera video to {self.output_path} with {codec}')

    def stop_ffmpeg(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3.0)
        finally:
            self.get_logger().info(f'Camera video recorder stopped: {self.output_path}')

    @staticmethod
    def msg_to_bgr(msg: Image):
        if msg.encoding not in ('bgr8', 'rgb8', 'mono8'):
            raise ValueError(f'unsupported encoding: {msg.encoding}')
        channels = 1 if msg.encoding == 'mono8' else 3
        min_step = msg.width * channels
        if msg.height == 0 or msg.width == 0 or msg.step < min_step:
            raise ValueError('invalid image shape or step')

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))[:, :min_step]
        image = rows.reshape((msg.height, msg.width, channels))

        if msg.encoding == 'bgr8':
            return np.ascontiguousarray(image)
        if msg.encoding == 'rgb8':
            return np.ascontiguousarray(image[:, :, ::-1])
        return np.ascontiguousarray(np.repeat(image, 3, axis=2))


def main(args=None):
    rclpy.init(args=args)
    node = CameraFfmpegRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.stop_ffmpeg()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
