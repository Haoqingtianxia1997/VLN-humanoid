import cv2
import rclpy
import threading
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3, Quaternion, PoseStamped
from nav_msgs.msg import Path

from isaaclab.sensors.imu.imu_data import ImuData
from isaaclab.sensors.camera.camera_data import CameraData


class InfoPublisher(Node):
    """ROS2 node for publishing camera data and IMU data"""
    def __init__(self):
        super().__init__('info_publisher')
        
        # Create publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 1)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 1)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 1)
        self.root_pose_pub = self.create_publisher(PoseStamped, '/robot/root_pose', 1)
        self.flag_publisher_ = self.create_publisher(Bool, '/mpc/running', 1)
        
        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/droid_slam/pose',
            self.pose_callback,
            1
        )
        self.trajectory_sub = self.create_subscription(
            Path,
            '/robot/trajectory',
            self.trajectory_callback,
            1
        )
        
        # CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Thread-safe storage for received data
        self.current_pose = None
        self.current_trajectory = None
        self.pose_lock = threading.Lock()
        self.trajectory_lock = threading.Lock()
        
        self.get_logger().info('📸 Information publisher initialized')
        self.get_logger().info('   RGB topic: /camera/rgb/image_raw')
        self.get_logger().info('   Depth topic: /camera/depth/image_raw')
        self.get_logger().info('🎯 IMU publisher initialized')
        self.get_logger().info('   IMU topic: /imu/data')
        self.get_logger().info('🤖 Root quaternion publisher initialized')
        self.get_logger().info('   Root quaternion topic: /robot/root_quaternion')
        self.get_logger().info('📍 Robot pose subscriber initialized')
        self.get_logger().info('   Pose topic: /robot/pose')
        self.get_logger().info('🛤️ Trajectory subscriber initialized')
        self.get_logger().info('   Trajectory topic: /robot/trajectory')
    
    def pose_callback(self, msg: PoseStamped):
        """Callback function for robot pose subscription (runs in separate thread)"""
        try:
            with self.pose_lock:
                self.current_pose = msg
            
            # Extract position and orientation for logging
            pos = msg.pose.position
            orient = msg.pose.orientation
            
            self.get_logger().info(
                f'📍 Received robot pose: '
                f'pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), '
                f'orient=({orient.w:.3f}, {orient.x:.3f}, {orient.y:.3f}, {orient.z:.3f})'
            )
        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {e}')
    
    def trajectory_callback(self, msg: Path):
        """Callback function for trajectory subscription (runs in separate thread)"""
        try:
            with self.trajectory_lock:
                self.current_trajectory = msg
            
            self.get_logger().info(
                f'🛤️ Received trajectory with {len(msg.poses)} waypoints'
            )
        except Exception as e:
            self.get_logger().error(f'Error in trajectory callback: {e}')
    
    def get_current_pose(self):
        """Thread-safe getter for current robot pose"""
        with self.pose_lock:
            return self.current_pose
    
    def get_current_trajectory(self):
        """Thread-safe getter for current trajectory"""
        with self.trajectory_lock:
            return self.current_trajectory
    
    def publish_rgb(self, rgb_image: np.ndarray):
        """Publish RGB image as uncompressed image"""
        try:
            # Ensure the image is in the correct format (uint8)
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Use cv_bridge to convert RGB image to ROS Image message
            # Note: cv_bridge expects BGR format, so we convert RGB to BGR
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Create Image message using cv_bridge
            msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_rgb_frame"
            
            self.rgb_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish RGB image: {e}')
    
    def publish_depth(self, depth_array: np.ndarray):
        """Publish depth image"""
        try:
            # Convert depth to 16-bit (common format for depth images)
            # Scale depth values to millimeters and convert to uint16
            depth_m = (depth_array).astype(np.uint16)
            
            # Create Image message
            msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="16UC1")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_depth_frame"
            
            self.depth_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish depth image: {e}')

    def publish_imu(self, imu_data: ImuData):
        """Publish IMU data (6-axis: linear acceleration + angular velocity)"""
        try:
            # Create IMU message
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "imu_link"
            
            # Extract data from IMU sensor (assuming single environment)
            if imu_data.lin_acc_b is not None:
                lin_acc = imu_data.lin_acc_b[0].cpu().numpy()  # Shape: (3,)
                msg.linear_acceleration.x = float(lin_acc[0])
                msg.linear_acceleration.y = float(lin_acc[1])
                msg.linear_acceleration.z = float(lin_acc[2])
            
            if imu_data.ang_vel_b is not None:
                ang_vel = imu_data.ang_vel_b[0].cpu().numpy()  # Shape: (3,)
                msg.angular_velocity.x = float(ang_vel[0])
                msg.angular_velocity.y = float(ang_vel[1])
                msg.angular_velocity.z = float(ang_vel[2])
            
            # Set orientation if available (from world frame data)
            if hasattr(imu_data, 'quat_w') and imu_data.quat_w is not None:
                quat = imu_data.quat_w[0].cpu().numpy()  # Shape: (4,) - (w, x, y, z)
                msg.orientation.w = float(quat[0])
                msg.orientation.x = float(quat[1])
                msg.orientation.y = float(quat[2])
                msg.orientation.z = float(quat[3])
            
            # Set covariances (can be tuned based on sensor characteristics)
            # ROS2 requires float arrays for covariance matrices (3x3 = 9 elements)
            msg.linear_acceleration_covariance = [
                0.01, 0.0, 0.0,  # Row 1
                0.0, 0.01, 0.0,  # Row 2  
                0.0, 0.0, 0.01   # Row 3
            ]
            msg.angular_velocity_covariance = [
                0.01, 0.0, 0.0,  # Row 1
                0.0, 0.01, 0.0,  # Row 2
                0.0, 0.0, 0.01   # Row 3
            ]
            msg.orientation_covariance = [
                0.01, 0.0, 0.0,  # Row 1
                0.0, 0.01, 0.0,  # Row 2
                0.0, 0.0, 0.01   # Row 3
            ]
            
            self.imu_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish IMU data: {e}')

    def publish_root_pose(self, pose: np.ndarray):
        """Publish robot root pose"""
        try:
            # Create PoseStamped message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"

            # Set position (x, y, z)
            msg.pose.position.x = float(pose[0])
            msg.pose.position.y = float(pose[1])
            msg.pose.position.z = float(pose[2])

            # Set orientation (w, x, y, z)
            msg.pose.orientation.w = float(pose[3])
            msg.pose.orientation.x = float(pose[4])
            msg.pose.orientation.y = float(pose[5])
            msg.pose.orientation.z = float(pose[6])

            self.root_pose_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish root pose: {e}')