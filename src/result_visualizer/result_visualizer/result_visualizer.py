import os
from array import array
import rclpy
import rclpy.qos
import numpy as np
from ament_index_python import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from radar_interface.msg import MatchResult, MatchedTarget, RadarMarkData

import cv2

configs = {
    "real_width": 28.0,
    "real_height": 15.0,
}

# Target:
# uint64 id
# float64[2] position
# float64[4] pos_covariance
# float64[2] velocity
# float64[4] vel_covariance
# float64 calc_z
# float64[3] observed_pos
# uint32 uncertainty

class ResultVisualizer(Node):
    mark = 0
    mark_masks = [32, 1, 2, 4, 8, 16]

    def __init__(self):
        super().__init__('result_visualizer')
        self.declare_parameter('im_show', True)
        self.declare_parameter('show_ally', True)
        self.declare_parameter('show_enemy', True)
        self.declare_parameter('default_team_color', 'red')
        self.get_logger().info('Initializing result_visualizer...')
        default_team_color = str(self.get_parameter('default_team_color').value).strip().lower()
        self.team_color = default_team_color != 'blue'
        self.ori_img = cv2.imread(os.path.join(get_package_share_directory('radar_bringup'), 'resource', 'RM2026-1.png'))
        self.bridge = None
        self.cv_bridge_disabled = False
        
        # 限制原图最大宽度，防止高分辨率地图导致窗口过大
        max_visual_width = 1200
        if self.ori_img is not None and self.ori_img.shape[1] > max_visual_width:
            scale = max_visual_width / self.ori_img.shape[1]
            new_h = int(self.ori_img.shape[0] * scale)
            self.ori_img = cv2.resize(self.ori_img, (max_visual_width, new_h))
            
        self.img_pub = self.create_publisher(Image, 'result_image', 10)
        self.target_sub = self.create_subscription(
            MatchResult, 'rm_radar_pipeline/match_result', self.target_callback, 10)
        self.color_sub = self.create_subscription(
            Bool, 'judge/color', self.team_color_callback, 10)
        self.mark_data_sub = self.create_subscription(
            RadarMarkData, 'judge/radar_mark_data', self.mark_data_callback, 10)
        self.get_logger().info('Initialized result_visualizer.')

    def _try_init_bridge(self):
        if self.bridge is not None or self.cv_bridge_disabled:
            return
        if int(np.__version__.split('.')[0]) >= 2:
            self.cv_bridge_disabled = True
            self.get_logger().warn(
                'NumPy>=2 detected; skip cv_bridge import to avoid ABI crash. result_image will not be published.')
            return
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.get_logger().info('cv_bridge initialized for result image publishing.')
        except Exception as e:
            self.cv_bridge_disabled = True
            self.get_logger().error(
                f'cv_bridge unavailable, disable result_image publish: {e}')

    def team_color_callback(self, msg: Bool):
        self.team_color = msg.data

    def mark_data_callback(self, msg: RadarMarkData):
        self.mark = msg.mark_progress

    def target_callback(self, msg: MatchResult):
        now_img = self.ori_img.copy()

        if self.team_color:
            cv2.putText(now_img, 'RED', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            cv2.putText(now_img, 'BLUE', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        ally_is_red = self.team_color
        enemy_is_red = not self.team_color

        ally_color = (0, 0, 255) if ally_is_red else (255, 0, 0)
        enemy_color = (0, 0, 255) if enemy_is_red else (255, 0, 0)
        cv2.putText(now_img, 'ALLY', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ally_color, 2)
        cv2.putText(now_img, 'ENEMY', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, enemy_color, 2)

        def draw(is_red: bool, num: int, target: MatchedTarget, is_ally: bool):
            if target.id == -1:
                return
            try:
                im_x = int(
                    target.position[0] / configs['real_width'] * self.ori_img.shape[1])
                im_y = int(
                    (1 - target.position[1] / configs['real_height']) * self.ori_img.shape[0])
                color = (0, 0, 255) if is_red else (255, 0, 0)
                # 我方和敌方都使用对应颜色的实心圆圈
                cv2.circle(now_img, (im_x, im_y), 20, color, -1)
                # 绘制序号和进度
                cv2.putText(now_img, str(num), (im_x - 10, im_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if is_red and not self.team_color and num < len(self.mark_masks) and (self.mark & self.mark_masks[num]):
                    cv2.putText(now_img, "MARK", (im_x - 20, im_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if not is_red and self.team_color and num < len(self.mark_masks) and (self.mark & self.mark_masks[num]):
                    cv2.putText(now_img, "MARK", (im_x - 20, im_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except IndexError as e:
                self.get_logger().error(f"Error in drawing: {e}")

        if ally_is_red:
            ally_targets, enemy_targets = msg.red, msg.blue
            ally_red, enemy_red = True, False
        else:
            ally_targets, enemy_targets = msg.blue, msg.red
            ally_red, enemy_red = False, True

        if self.get_parameter('show_ally').value:
            for i, target in enumerate(ally_targets):
                draw(ally_red, i, target, True)
        if self.get_parameter('show_enemy').value:
            for i, target in enumerate(enemy_targets):
                draw(enemy_red, i, target, False)

        self._try_init_bridge()
        if self.bridge is not None:
            img = self.bridge.cv2_to_imgmsg(now_img, encoding="bgr8")
            self.img_pub.publish(img)
        if self.get_parameter('im_show').value:
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.imshow('result', now_img)
            cv2.waitKey(1)


def main():
    rclpy.init()
    result_visualizer = ResultVisualizer()
    rclpy.spin(result_visualizer)
    result_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
