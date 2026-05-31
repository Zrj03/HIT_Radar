import math
import os
import threading
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    return _normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))


def _transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = _normalize_quaternion(np.array([q.x, q.y, q.z, q.w], dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _matrix_to_transform(matrix: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    qx, qy, qz, qw = _rotation_to_quaternion(matrix[:3, :3])
    msg.transform.rotation.x = float(qx)
    msg.transform.rotation.y = float(qy)
    msg.transform.rotation.z = float(qz)
    msg.transform.rotation.w = float(qw)
    return msg


class CameraExtrinsicTuner(Node):
    def __init__(self) -> None:
        super().__init__("camera_extrinsic_tuner")
        self.image_topic = self.declare_parameter("image_topic", "/radar/hik_6mm/image").value
        self.compressed_image_topic = self.declare_parameter(
            "compressed_image_topic", "/radar/hik_6mm/image/compressed").value
        self.camera_info_topic = self.declare_parameter("camera_info_topic", "/radar/hik_6mm/camera_info").value
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.lidar_frame = self.declare_parameter("lidar_frame", "lidar_mid70_frame").value
        self.camera_frame = self.declare_parameter("camera_frame", "hik_6mm_frame").value
        self.calibration_file = self.declare_parameter("calibration_file", "calibration.yaml").value
        self.display_width = int(self.declare_parameter("display_width", 1536).value)
        self.display_height = int(self.declare_parameter("display_height", 1125).value)
        self.auto_save = bool(self.declare_parameter("auto_save", True).value)
        self.broadcast_tf = bool(self.declare_parameter("broadcast_tf", True).value)
        self.use_compressed_image = bool(self.declare_parameter("use_compressed_image", False).value)
        self.points_3d, self.point_names = self._load_points()

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.original_size: Optional[Tuple[int, int]] = None
        self.display_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.first_image_logged = False
        self.selected_points: List[Tuple[float, float]] = []
        self.pending_point: Optional[Tuple[int, int]] = None
        self.last_lidar_to_camera: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        if self.use_compressed_image:
            self.create_subscription(
                CompressedImage,
                self.compressed_image_topic,
                self._compressed_image_callback,
                qos_profile_sensor_data)
        self.create_timer(0.1, self._ui_timer)
        self.create_timer(0.1, self._broadcast_timer)

        cv2.namedWindow("camera_extrinsic_tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera_extrinsic_tuner", self.display_width, self.display_height)
        cv2.namedWindow("camera_extrinsic_roi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera_extrinsic_roi", 420, 420)
        cv2.setMouseCallback("camera_extrinsic_tuner", self._mouse_callback)
        self.get_logger().info(
            "Camera extrinsic tuner ready. Left click point, WASD refine, n confirm, u undo, Enter solve, q quit.")

    def _load_points(self) -> Tuple[np.ndarray, List[str]]:
        flat = list(self.declare_parameter("calibration_points", [
            5.471, 7.5, 0.0,
            10.936, 3.839, 0.868,
            25.49, 7.5, 1.24524,
            16.925, 11.375, 1.745,
            20.20, 4.2, 0.8,
        ]).value)
        if len(flat) < 12 or len(flat) % 3 != 0:
            raise RuntimeError("calibration_points must contain at least 4 XYZ points")
        names = list(self.declare_parameter("calibration_point_names", [
            "self_fortress",
            "self_tower",
            "enemy_base",
            "enemy_tower",
            "enemy_high",
        ]).value)
        points = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
        if len(names) != len(points):
            names = [f"point_{i + 1}" for i in range(len(points))]
        return points, names

    def _resolve_calibration_path(self) -> str:
        if os.path.isabs(self.calibration_file):
            return self.calibration_file
        bringup_share = get_package_share_directory("radar_bringup")
        return os.path.join(bringup_share, "config", self.calibration_file)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((1, -1))
        self.get_logger().info("Camera intrinsics received from camera_info.")

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}")
            return
        self._set_image(image)

    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            self._set_image(image)

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ("bgr8", "rgb8", "mono8"):
            raise ValueError(f"unsupported image encoding: {msg.encoding}")
        channels = 1 if msg.encoding == "mono8" else 3
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        array = array[:, :msg.width * channels].reshape((msg.height, msg.width, channels))
        if msg.encoding == "rgb8":
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if msg.encoding == "mono8":
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        return array.copy()

    def _set_image(self, image: np.ndarray) -> None:
        with self.lock:
            self.original_size = (image.shape[1], image.shape[0])
            self.display_image = cv2.resize(image, (self.display_width, self.display_height))
            should_log = not self.first_image_logged
            self.first_image_logged = True
        if should_log:
            self.get_logger().info(
                f"Camera image received: {self.original_size[0]}x{self.original_size[1]} -> "
                f"{self.display_width}x{self.display_height}")

    def _mouse_callback(self, event, x, y, _flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.pending_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._update_roi(x, y)

    def _update_roi(self, x: int, y: int) -> None:
        with self.lock:
            image = None if self.display_image is None else self.display_image.copy()
        if image is None:
            return
        margin = 40
        x = max(margin, min(x, image.shape[1] - margin - 1))
        y = max(margin, min(y, image.shape[0] - margin - 1))
        roi = image[y - margin:y + margin, x - margin:x + margin]
        roi = cv2.resize(roi, (420, 420), interpolation=cv2.INTER_NEAREST)
        cv2.line(roi, (210, 70), (210, 350), (0, 0, 255), 1)
        cv2.line(roi, (70, 210), (350, 210), (0, 0, 255), 1)
        cv2.imshow("camera_extrinsic_roi", roi)

    def _ui_timer(self) -> None:
        with self.lock:
            image = None if self.display_image is None else self.display_image.copy()
            pending = self.pending_point
            selected = list(self.selected_points)
        if image is None:
            cv2.waitKey(1)
            return

        for idx, point in enumerate(selected):
            display_pt = self._original_to_display(point)
            cv2.circle(image, display_pt, 6, (0, 255, 0), -1)
            cv2.putText(image, str(idx + 1), (display_pt[0] + 8, display_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if pending is not None:
            cv2.circle(image, pending, 7, (0, 0, 255), 2)
            cv2.line(image, (pending[0] - 15, pending[1]), (pending[0] + 15, pending[1]), (0, 0, 255), 1)
            cv2.line(image, (pending[0], pending[1] - 15), (pending[0], pending[1] + 15), (0, 0, 255), 1)

        next_idx = min(len(selected), len(self.point_names) - 1)
        status = f"{len(selected)}/{len(self.points_3d)}"
        if len(selected) < len(self.points_3d):
            status += f" next: {self.point_names[next_idx]} {self.points_3d[next_idx].tolist()}"
        cv2.putText(image, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(image, "click, WASD refine, n confirm, u undo, Enter solve, q quit",
                    (30, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("camera_extrinsic_tuner", image)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)

    def _handle_key(self, key: int) -> None:
        if key == 255:
            return
        with self.lock:
            pending = self.pending_point
            if pending is not None and key in (ord("w"), ord("a"), ord("s"), ord("d")):
                x, y = pending
                if key == ord("w"):
                    y -= 1
                elif key == ord("s"):
                    y += 1
                elif key == ord("a"):
                    x -= 1
                elif key == ord("d"):
                    x += 1
                x = max(0, min(x, self.display_width - 1))
                y = max(0, min(y, self.display_height - 1))
                self.pending_point = (x, y)
                return
            if key == ord("n") and pending is not None:
                self.selected_points.append(self._display_to_original(pending))
                self.pending_point = None
                return
            if key == ord("u"):
                if self.pending_point is not None:
                    self.pending_point = None
                elif self.selected_points:
                    self.selected_points.pop()
                return
            if key == ord("r"):
                self.selected_points.clear()
                self.pending_point = None
                return
        if key in (10, 13):
            self._solve()
        elif key == ord("q"):
            rclpy.shutdown()

    def _display_to_original(self, point: Tuple[int, int]) -> Tuple[float, float]:
        if self.original_size is None:
            return float(point[0]), float(point[1])
        width, height = self.original_size
        return point[0] * width / self.display_width, point[1] * height / self.display_height

    def _original_to_display(self, point: Tuple[float, float]) -> Tuple[int, int]:
        if self.original_size is None:
            return int(point[0]), int(point[1])
        width, height = self.original_size
        return int(point[0] * self.display_width / width), int(point[1] * self.display_height / height)

    def _solve(self) -> None:
        with self.lock:
            image_points = np.asarray(self.selected_points, dtype=np.float64)
        if len(image_points) != len(self.points_3d):
            self.get_logger().warn("Need exactly all configured points before solving.")
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn("Camera intrinsics are not available yet.")
            return

        ok, rvec, tvec = cv2.solvePnP(
            self.points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            self.get_logger().error("solvePnP failed.")
            return
        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :3] = rotation
        camera_from_world[:3, 3] = tvec.reshape(3)

        try:
            world_from_lidar_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.lidar_frame,
                Time(),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.get_logger().error(
                f"Cannot compute lidar->camera before pointcloud alignment TF is ready: {exc}")
            return

        world_from_lidar = _transform_to_matrix(world_from_lidar_msg)
        lidar_pose_translation = world_from_lidar[:3, 3]
        lidar_pose_rotation = world_from_lidar[:3, :3]
        lidar_pose_is_default = (
            np.linalg.norm(lidar_pose_translation) < 1e-3
            and np.linalg.norm(lidar_pose_rotation - np.eye(3, dtype=np.float64)) < 1e-3
        )
        if lidar_pose_is_default:
            self.get_logger().error(
                "Refusing to save calibration: world->lidar is still the default identity transform. "
                "Wait for pc_aligner to publish a valid field alignment before pressing Enter.")
            return
        self.get_logger().info(
            "Using world->lidar translation="
            f"{lidar_pose_translation.reshape(-1).tolist()} for lidar->camera solve.")
        lidar_to_camera = camera_from_world @ world_from_lidar
        self.last_lidar_to_camera = lidar_to_camera
        self.get_logger().info(f"solvePnP rvec={rvec.reshape(-1).tolist()} tvec={tvec.reshape(-1).tolist()}")
        self.get_logger().info(f"lidar_to_camera=\n{lidar_to_camera}")
        if self.auto_save:
            self._save_calibration(lidar_to_camera)

    def _save_calibration(self, lidar_to_camera: np.ndarray) -> None:
        path = self._resolve_calibration_path()
        self._write_calibration(path, lidar_to_camera)
        share_path = os.path.join(
            get_package_share_directory("radar_bringup"),
            "config",
            "calibration.yaml",
        )
        if os.path.abspath(share_path) != os.path.abspath(path):
            self._write_calibration(share_path, lidar_to_camera)

    def _write_calibration(self, path: str, lidar_to_camera: np.ndarray) -> None:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            self.get_logger().error(f"Failed to open calibration file for writing: {path}")
            return
        fs.write("camera_intrinsic", self.camera_matrix)
        fs.write("camera_distortion", self.dist_coeffs)
        fs.write("lidar_to_camera", lidar_to_camera)
        fs.release()
        self.get_logger().info(f"Saved lidar_to_camera calibration to {path}")

    def _broadcast_timer(self) -> None:
        if not self.broadcast_tf or self.last_lidar_to_camera is None:
            return
        msg = _matrix_to_transform(
            self.last_lidar_to_camera,
            self.camera_frame,
            self.lidar_frame,
            self.get_clock().now().to_msg(),
        )
        self.tf_broadcaster.sendTransform(msg)


def main() -> None:
    rclpy.init()
    node = CameraExtrinsicTuner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
