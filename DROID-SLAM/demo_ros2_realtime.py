import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import time
import argparse
import threading
import queue
from collections import deque

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge
import message_filters
import socket
import json


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


class DroidSlamNode(Node):
    def __init__(self, rgb_topic, depth_topic, calib_file, args):
        super().__init__('droid_slam_node')
        
        self.bridge = CvBridge()
        self.args = args
        self.droid = None
        self.frame_count = 0
        
        self.idx = 0
        
        # UDP接收器相关
        self.udp_socket = None
        self.latest_trajectory_data = None
        self.trajectory_lock = threading.Lock()
        
        # UDP发送器相关（用于发送角度数据）
        self.angle_udp_socket = None
        self.angle_udp_host = "127.0.0.1"
        self.angle_udp_port = 12347  # 使用不同的端口发送角度数据
        
        # 路径UDP接收器相关
        self.path_udp_socket = None
        self.path_udp_port = 12348
        self.latest_planned_path = None
        self.path_lock = threading.Lock()
        
        # 全局路径规划相关
        self.goal_position = None
        self.goal_lock = threading.Lock()
        self.path_planning_enabled = False
        
        # 初始化UDP接收器
        if args.publish_pose:
            self._init_udp_receiver()
            self._init_angle_udp_sender()
            self._init_path_udp_receiver()
        
        
        # Create pose publishers (only if enabled)
        if args.publish_pose:
            self.pose_pub = self.create_publisher(PoseStamped, 'droid_slam/pose', 1)
            self.odom_pub = self.create_publisher(Odometry, 'droid_slam/odometry', 1)
            
            # 添加路径发布器
            self.path_pub = self.create_publisher(Path, '/robot/trajectory', 1)
            
            self.pose_sub = self.create_subscription(PoseStamped, '/robot/root_pose', self.pose_callback, 1)
            self.goal_sub = self.create_subscription(PoseStamped, '/goal_position', self.goal_callback, 1)

        else:
            self.pose_pub = None
            self.odom_pub = None
            self.path_pub = None
  
            self.pose_sub = None
            self.goal_sub = None
            
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.01  
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        
        self.pose_isaac = PoseStamped()
        self.quaternion = None
        self.pose_isaac_lock = threading.Lock()
        
        # Store previous pose for velocity calculation
        self.previous_pose = None
        self.previous_time = None
        
        # intrinsics
        calib = np.loadtxt(calib_file, delimiter=" ")
        fx, fy, cx, cy = calib[:4]
        self.K = np.eye(3)
        self.K[0,0] = fx
        self.K[0,2] = cx
        self.K[1,1] = fy
        self.K[1,2] = cy
        

    # Callbacks
    def pose_callback(self, msg: PoseStamped):
        """处理位姿消息并发送torso相对于世界坐标系的四元数"""
        with self.pose_isaac_lock:
            self.pose_isaac.header.stamp = msg.header.stamp
            self.pose_isaac.pose.position.x = msg.pose.position.x
            self.pose_isaac.pose.position.y = msg.pose.position.y
            self.pose_isaac.pose.position.z = msg.pose.position.z
            self.pose_isaac.pose.orientation.w = msg.pose.orientation.w
            self.pose_isaac.pose.orientation.x = msg.pose.orientation.x
            self.pose_isaac.pose.orientation.y = msg.pose.orientation.y
            self.pose_isaac.pose.orientation.z = msg.pose.orientation.z

            self.quaternion = {
                'w': msg.pose.orientation.w,
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z
            }

        if self.args.publish_pose and self.angle_udp_socket is not None and self.quaternion is not None:
            quaternion_data = {
                'torso_to_world_quat': self.quaternion,
                'timestamp': self.idx
            }
            self.idx += 1
            self._send_angle_data_udp(quaternion_data)
        
    def goal_callback(self, msg: PoseStamped):
        """处理目标位置消息"""
        with self.goal_lock:
            self.goal_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self.path_planning_enabled = True

        self.get_logger().info(f"Received goal position: [{msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}]")

        # 通过UDP发送目标位置到可视化器进行路径规划
        if self.args.publish_pose and self.angle_udp_socket is not None:
            goal_data = {
                'goal_position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'type': 'goal_position',
                'timestamp': time.time()
            }
            self._send_angle_data_udp(goal_data)

    def synchronized_callback(self, rgb_msg, depth_msg):
            
        # Convert ROS message to OpenCV format
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        # Process image data
        processed_data = self.process_frame(rgb_image, depth_image)
        
        if processed_data is not None:
            t, image_tensor, depth_tensor, intrinsics = processed_data
            
            # Show image if visualization is enabled
            if not self.args.disable_vis:
                show_image(image_tensor[0])

            # Initialize DROID (on first frame)
            if self.droid is None:
                self.args.image_size = [image_tensor.shape[2], image_tensor.shape[3]]
                self.droid = DroidAsync(self.args) if self.args.asynchronous else Droid(self.args)

            # Track current frame
            self.droid.track(t, image_tensor, depth=depth_tensor, intrinsics=intrinsics)

            # Publish camera pose after tracking (with a small delay to ensure processing is complete)
            self.publish_camera_pose(rgb_msg.header.stamp)
            
            # Publish planned path if available
            self.publish_planned_path(rgb_msg.header.stamp)

        else:
            self.get_logger().warning('process_frame returned None')
        

    # UDP Configuration
    def _init_udp_receiver(self):
        """初始化UDP接收器"""
        try:
            self.udp_socket = self.create_udp_receiver(host="127.0.0.1", port=12346, timeout=0.001)
            if self.udp_socket:
                # 启动UDP接收线程
                self.udp_thread = threading.Thread(target=self._udp_receiver_thread, daemon=True)
                self.udp_thread.start()
                self.get_logger().info("UDP receiver thread started")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize UDP receiver: {e}")
            self.udp_socket = None
    
    def _init_angle_udp_sender(self):
        """初始化UDP发送器用于发送角度数据"""
        try:
            self.angle_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.get_logger().info("Angle UDP sender initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize angle UDP sender: {e}")
            self.angle_udp_socket = None
    
    def _init_path_udp_receiver(self):
        """初始化路径UDP接收器"""
        try:
            self.path_udp_socket = self.create_udp_receiver(
                host="127.0.0.1", 
                port=self.path_udp_port, 
                timeout=0.001
            )
            if self.path_udp_socket:
                # 启动路径UDP接收线程
                self.path_udp_thread = threading.Thread(target=self._path_udp_receiver_thread, daemon=True)
                self.path_udp_thread.start()
                self.get_logger().info(f"Path UDP receiver thread started on port {self.path_udp_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize path UDP receiver: {e}")
            self.path_udp_socket = None
    
    def _send_angle_data_udp(self, angle_data):
        """通过UDP发送角度数据"""
        if self.angle_udp_socket is None:
            return
            
        try:
            # 将角度数据序列化为JSON
            json_data = json.dumps(angle_data)
            data_bytes = json_data.encode('utf-8')
            
            # 发送数据
            self.angle_udp_socket.sendto(data_bytes, (self.angle_udp_host, self.angle_udp_port))
            
        except Exception as e:
            self.get_logger().error(f"Failed to send angle data via UDP: {e}")
    
    def _udp_receiver_thread(self):
        """UDP接收线程"""
        while rclpy.ok():
            try:
                if self.udp_socket:
                    trajectory_data = self.receive_trajectory_data_udp(self.udp_socket)
                    if trajectory_data:
                        with self.trajectory_lock:
                            self.latest_trajectory_data = trajectory_data
                        # self.get_logger().debug("Received trajectory data via UDP")
            except Exception as e:
                # self.get_logger().error(f"Error in UDP receiver thread: {e}")
                pass
            time.sleep(0.001)  # 小延迟避免CPU占用过高
    
    def _path_udp_receiver_thread(self):
        """路径UDP接收线程"""
        while rclpy.ok():
            try:
                if self.path_udp_socket:
                    path_data = self.receive_path_data_udp(self.path_udp_socket)
                    if path_data:
                        with self.path_lock:
                            self.latest_planned_path = path_data
                        self.get_logger().info(f"Received planned path with {len(path_data.get('path', []))} waypoints")
            except Exception as e:
                # self.get_logger().error(f"Error in path UDP receiver thread: {e}")
                pass
            time.sleep(0.001)  # 小延迟避免CPU占用过高

    def create_udp_receiver(self,host="127.0.0.1", port=12345, timeout=1.0):
        """创建UDP接收器用于接收轨迹数据"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((host, port))
            sock.settimeout(timeout)
            print(f"UDP receiver created and listening on {host}:{port}")
            return sock
        except Exception as e:
            print(f"Failed to create UDP receiver: {e}")
            return None

    def receive_trajectory_data_udp(self, sock):
        """从UDP接收轨迹数据"""
        try:
            data, addr = sock.recvfrom(65536)  # 64KB buffer
            json_data = data.decode('utf-8')
            trajectory_data = json.loads(json_data)
            return trajectory_data
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Failed to receive trajectory data via UDP: {e}")
            return None
    
    def receive_path_data_udp(self, sock):
        """从UDP接收路径规划数据"""
        try:
            data, addr = sock.recvfrom(65536)  # 64KB buffer
            json_data = data.decode('utf-8')
            path_data = json.loads(json_data)
            # 验证数据类型
            if path_data.get('type') == 'planned_path':
                return path_data
            else:
                return None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Failed to receive path data via UDP: {e}")
            return None

    # Publisher
    def publish_camera_pose(self, stamp):
  
        if not self.args.publish_pose:
            return
            

        trajectory_data = None
        with self.trajectory_lock:
            trajectory_data = self.latest_trajectory_data
        
        if trajectory_data is not None and 'latest_position' in trajectory_data:
        
            latest_pos = trajectory_data['latest_position']

            latest_pos = np.array([latest_pos['x'], latest_pos['y'], latest_pos['z']])

            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp

            pose_msg.pose.position.x = float(latest_pos[0])  
            pose_msg.pose.position.y = float(latest_pos[1])  
            pose_msg.pose.position.z = float(latest_pos[2])  
            if self.quaternion is not None:
                pose_msg.pose.orientation.w = self.quaternion['w']
                pose_msg.pose.orientation.x = self.quaternion['x']
                pose_msg.pose.orientation.y = self.quaternion['y']
                pose_msg.pose.orientation.z = self.quaternion['z']
            else:
                pose_msg.pose.orientation.w = 1.0
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
            
            self.pose_pub.publish(pose_msg)
            
            odom_msg = Odometry()
            odom_msg.header.stamp = stamp
        
            odom_msg.pose.pose = pose_msg.pose
            

            current_time = stamp.sec + stamp.nanosec * 1e-9
            if self.previous_pose is not None and self.previous_time is not None:
                dt = current_time - self.previous_time
                if dt > 0:
                    dx = latest_pos[0] - self.previous_pose[0]
                    dy = latest_pos[1] - self.previous_pose[1]
                    dz = latest_pos[2] - self.previous_pose[2]

                    odom_msg.twist.twist.linear.x = dx / dt
                    odom_msg.twist.twist.linear.y = dy / dt
                    odom_msg.twist.twist.linear.z = dz / dt
        
            if self.odom_pub is not None:
                self.odom_pub.publish(odom_msg)
            

            self.previous_pose = latest_pos.copy()
            self.previous_time = current_time      
    
    def publish_planned_path(self, stamp):
        """发布规划的路径"""
        if self.path_pub is None:
            return
        
        path_data = None
        with self.path_lock:
            path_data = self.latest_planned_path
            
        if path_data is not None and 'path' in path_data:
            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self.args.map_frame_id
            
            for waypoint in path_data['path']:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = stamp
                pose_stamped.header.frame_id = self.args.map_frame_id
                
                # waypoint是[x, y]格式，z设为0
                pose_stamped.pose.position.x = float(waypoint[0])
                pose_stamped.pose.position.y = float(waypoint[1])
                pose_stamped.pose.position.z = 0.0
                
                # 设置默认朝向
                pose_stamped.pose.orientation.w = 1.0
                pose_stamped.pose.orientation.x = 0.0
                pose_stamped.pose.orientation.y = 0.0
                pose_stamped.pose.orientation.z = 0.0
                
                path_msg.poses.append(pose_stamped)
            
            self.path_pub.publish(path_msg)
            self.get_logger().info(f"Published planned path with {len(path_msg.poses)} waypoints")
    
    def process_frame(self, rgb_image, depth_image):
        """Process RGB and depth images, returning the format needed by DROID-SLAM"""
        
            
        h0, w0, _ = rgb_image.shape

        # Adjust resolution for higher density point cloud
        h1 = int(h0 * np.sqrt((720 * 540) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((720 * 540) / (h0 * w0)))

        rgb_image = cv2.resize(rgb_image, (w1, h1))
        rgb_image = rgb_image[:h1-h1%8, :w1-w1%8]
        rgb_tensor = torch.as_tensor(rgb_image).permute(2, 0, 1)

        # Process depth image
        if depth_image is not None:
            
            # Depth image processing
            depth = torch.as_tensor(depth_image.astype(np.float32) / self.args.depth_scale)

            # Apply depth range limits
            depth = torch.clamp(depth, self.args.min_depth, self.args.max_depth)

            # Filter invalid depth values
            depth = torch.where(depth > 0, depth, torch.tensor(0.0))

            # Adjust depth image size
            depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
            depth = depth[:h1-h1%8, :w1-w1%8]

        else:
            depth = None

        # Adjust intrinsics
        intrinsics = torch.as_tensor([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        self.frame_count += 1
        return self.frame_count, rgb_tensor[None], depth, intrinsics

    def terminate_slam(self):
        """Terminate SLAM and save results"""
        # 关闭UDP socket
        if self.udp_socket:
            try:
                self.udp_socket.close()
                self.get_logger().info('UDP socket closed')
            except Exception as e:
                self.get_logger().error(f'Error closing UDP socket: {str(e)}')
        
        # 关闭角度UDP socket
        if self.angle_udp_socket:
            try:
                self.angle_udp_socket.close()
                self.get_logger().info('Angle UDP socket closed')
            except Exception as e:
                self.get_logger().error(f'Error closing angle UDP socket: {str(e)}')
        
        # 关闭路径UDP socket
        if self.path_udp_socket:
            try:
                self.path_udp_socket.close()
                self.get_logger().info('Path UDP socket closed')
            except Exception as e:
                self.get_logger().error(f'Error closing path UDP socket: {str(e)}')
        
        # 清理目标位置
        if hasattr(self, 'goal_lock'):
            with self.goal_lock:
                self.goal_position = None
                self.path_planning_enabled = False
        
        if self.droid is not None:
            self.get_logger().info('Terminating DROID-SLAM...')
            # Note: Since this is a real-time stream, we cannot pass image_stream to terminate
            # Need to check if DROID's terminate method supports no-argument calls
            try:
                traj_est = self.droid.terminate()
                self.get_logger().info('DROID-SLAM terminated successfully')
                return traj_est
            except Exception as e:
                self.get_logger().error(f'Error terminating DROID-SLAM: {str(e)}')
                return None
        return None


def save_reconstruction(droid, save_path):
    """Save reconstruction results"""
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)

def main():
    parser = argparse.ArgumentParser(description='DROID-SLAM with ROS2 input')

    # ROS2 specific arguments
    parser.add_argument("--rgb_topic", type=str, required=True, help="ROS2 RGB image topic")
    parser.add_argument("--depth_topic", type=str, required=True, help="ROS2 depth image topic")
    parser.add_argument("--calib", type=str, required=True, help="path to calibration file")

    # Depth processing arguments
    parser.add_argument("--depth_scale", type=float, default=1.0, help="depth scale factor")
    parser.add_argument("--min_depth", type=float, default=0.1, help="minimum valid depth in meters")
    parser.add_argument("--max_depth", type=float, default=100000.0, help="maximum valid depth in meters")

    # DROID-SLAM arguments
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[480, 640])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames within this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal suppression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    
    # Pose publishing arguments
    parser.add_argument("--publish_pose", action="store_true", help="enable pose publishing to ROS topics")
    parser.add_argument("--camera_frame_id", type=str, default="camera_link", help="frame ID for camera in TF tree")
    parser.add_argument("--map_frame_id", type=str, default="map", help="frame ID for map/world coordinate system")
    
    # 全局路径规划说明:
    # 启用 --publish_pose 参数后，系统会订阅 /goal_position (geometry_msgs/Point) 话题
    # 发送目标位置命令: ros2 topic pub /goal_position geometry_msgs/Point "x: 2.0, y: 3.0, z: 0.0"
    # 路径会自动在占用栅格地图中显示为蓝色，目标点为紫色

    args = parser.parse_args()

    # Set DROID arguments
    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    # Enable upsampling if high-resolution depth is needed
    if args.reconstruction_path is not None:
        args.upsample = True

    # Initialize ROS2
    rclpy.init()

    # Create DROID-SLAM node
    slam_node = DroidSlamNode(args.rgb_topic, args.depth_topic, args.calib, args)
    
    try:
        print(f"Starting DROID-SLAM, subscribing to topics:")
        print(f"  RGB: {args.rgb_topic}")
        print(f"  Depth: {args.depth_topic}")
        print("Press Ctrl+C to stop...")

        # Run the node
        rclpy.spin(slam_node)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping DROID-SLAM...")

    finally:
        # Terminate SLAM and save results
        traj_est = slam_node.terminate_slam()

        if args.reconstruction_path is not None and slam_node.droid is not None:
            save_reconstruction(slam_node.droid, args.reconstruction_path)
            print(f"Reconstruction results saved to: {args.reconstruction_path}")

        # Clean up ROS2 resources
        slam_node.destroy_node()
        rclpy.shutdown()
        print("DROID-SLAM stopped")

if __name__ == '__main__':
    main()
