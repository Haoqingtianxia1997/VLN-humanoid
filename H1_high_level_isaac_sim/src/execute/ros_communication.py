import threading
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry


class RosVlmNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Create publishers
        self.target_pub = self.create_publisher(PoseStamped, '/goal_position', 1)

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw',
            self.rgb_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/info',
            self.camera_info_callback, 1)
        self.position_sub = self.create_subscription(
            PoseStamped, '/droid_slam/pose',
            self.position_callback, 1)
        self.trigger_sub = self.create_subscription(
            Bool, '/mpc/running',
            self.trigger_callback, 1)
        
        # Thread-safe storage for received data
        self.rgb = None
        self.depth = None
        self.camera_info = None
        self.pose = None
        self.mpc_running = False

        self.data_lock = threading.Lock()
        self.trigger_lock = threading.Lock()

    def _image_msg_to_ndarray(self, msg: Image) -> np.ndarray:
        """
        Convert ROS Image message to numpy array without cv_bridge.
        Supports: rgb8, bgr8, mono8, mono16 / 16UC1, 32FC1
        """
        encoding = msg.encoding.lower()
        mapping = {
            "rgb8": ("u1", 3),
            "bgr8": ("u1", 3),
            "mono8": ("u1", 1),
            "mono16": ("u2", 1),
            "16uc1": ("u2", 1),
            "32fc1": ("f4", 1),
            "32f": ("f4", 1),
        }
        if encoding not in mapping:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

        dt_char, channels = mapping[encoding]
        dtype = np.dtype(dt_char)
        if msg.is_bigendian:
            dtype = dtype.newbyteorder('>')
        else:
            dtype = dtype.newbyteorder('<')

        elems_per_row = msg.step // dtype.itemsize
        raw = np.frombuffer(msg.data, dtype=dtype)
        try:
            raw = raw.reshape((msg.height, elems_per_row))
        except Exception as e:
            raw = raw.copy()
            raw = raw.reshape((msg.height, msg.step // dtype.itemsize))

        useful = raw[:, : msg.width * channels]
        if channels > 1:
            img = useful.reshape((msg.height, msg.width, channels))
            if encoding == "bgr8":
                img = img[..., ::-1]
        else:
            img = useful.reshape((msg.height, msg.width))
        return np.ascontiguousarray(img)

    def rgb_callback(self, msg: Image):
        try:
            arr = self._image_msg_to_ndarray(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return
        with self.data_lock:
            self.rgb = arr

    def depth_callback(self, msg: Image):
        try:
            arr = self._image_msg_to_ndarray(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return
        with self.data_lock:
            self.depth = arr

    def camera_info_callback(self, msg: CameraInfo):
        with self.data_lock:
            self.camera_info = msg
        
        try:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(f"Error destroying camera_info_sub: {e}")

    def position_callback(self, msg: PoseStamped):
        with self.data_lock:
            self.pose = msg.pose

    def trigger_callback(self, msg: Bool):
        with self.trigger_lock:
            self.mpc_running = msg.data

    def publish_target_pose(self, pose: Pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose
        self.target_pub.publish(pose_stamped)
        self.get_logger().info(f"Published target position: {pose.position} and quat {pose.orientation}")

    def get_current_data(self) -> None | tuple[np.ndarray, np.ndarray, CameraInfo, Pose]:
        """Thread-safe getter for current RGB and depth images"""
        with self.data_lock:
            return self.rgb, self.depth, self.camera_info, self.pose

    def get_path_execution_running(self):
        """Thread-safe getter for MPC running state"""
        with self.trigger_lock:
            return self.mpc_running
