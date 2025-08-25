#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time


class TrajectoryVisualizerNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer_node')
        
        # 订阅SLAM位置话题
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            'droid_slam/pose', 
            self.pose_callback, 
            10
        )
        
        # 存储轨迹数据
        self.trajectory_points = []
        self.trajectory_lock = threading.Lock()
        
        # 初始化3D轨迹可视化
        self._init_3d_trajectory_plot()
        
        # 创建定时器定期更新图形
        self.update_timer = self.create_timer(0.1, self.update_plot)  # 10Hz更新频率
        
        self.get_logger().info('Trajectory Visualizer Node started, subscribing to droid_slam/pose')

    def pose_callback(self, msg: PoseStamped):
        """接收位置消息的回调函数"""
        # 提取x, y, z坐标
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 添加到轨迹点列表
        with self.trajectory_lock:
            self.trajectory_points.append([x, y, z])
            
            # 保存所有轨迹点，不限制数量
        
        self.get_logger().debug(f'Received pose: x={x:.3f}, y={y:.3f}, z={z:.3f}')

    def _init_3d_trajectory_plot(self):
        """初始化3D轨迹可视化（与droid_visualizer.py相同的处理方式）"""
        # 设置matplotlib非阻塞模式
        plt.ion()
        
        # 创建3D图
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('DROID-SLAM Real-time Trajectory (3D) - ROS2 Subscriber')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 初始化轨迹线和点
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
        
        # 设置坐标轴等比例
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.legend()
        
        plt.show(block=False)

    def update_plot(self):
        """定期更新3D轨迹图（与droid_visualizer.py相同的处理方式）"""
        try:
            with self.trajectory_lock:
                trajectory_points_copy = self.trajectory_points.copy()
            
            # 更新轨迹线
            if len(trajectory_points_copy) > 1:
                x_data = [p[0] for p in trajectory_points_copy]
                y_data = [p[1] for p in trajectory_points_copy]
                z_data = [p[2] for p in trajectory_points_copy]
                
                self.trajectory_line.set_data_3d(x_data, y_data, z_data)
                
                # 更新当前位置点
                self.current_point.set_data_3d([x_data[-1]], [y_data[-1]], [z_data[-1]])
                
                # 动态调整坐标轴范围，保持各轴分度值相同（与修改后的droid_visualizer.py相同）
                margin = 0.5  # 边距
                
                # 计算每个轴的数据范围
                x_min, x_max = min(x_data) - margin, max(x_data) + margin
                y_min, y_max = min(y_data) - margin, max(y_data) + margin
                z_min, z_max = min(z_data) - margin, max(z_data) + margin
                
                # 找到最大的数据范围
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                max_range = max(x_range, y_range, z_range)
                
                # 计算每个轴的中心点
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                z_center = (z_min + z_max) / 2
                
                # 使用最大范围设置所有轴的范围，保持相同的分度值
                half_range = max_range / 2
                self.ax.set_xlim(x_center - half_range, x_center + half_range)
                self.ax.set_ylim(y_center - half_range, y_center + half_range)
                self.ax.set_zlim(z_center - half_range, z_center + half_range)
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f"Error updating 3D trajectory plot: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            plt.close(self.fig)
            self.get_logger().info('Matplotlib figure closed')
        except Exception as e:
            self.get_logger().error(f'Error closing matplotlib figure: {e}')


def main():
    rclpy.init()
    
    # 创建轨迹可视化节点
    visualizer_node = TrajectoryVisualizerNode()
    
    try:
        print("Starting Trajectory Visualizer Node...")
        print("Subscribing to topic: droid_slam/pose")
        print("Press Ctrl+C to stop...")
        
        # 运行节点
        rclpy.spin(visualizer_node)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping visualizer...")
    
    finally:
        # 清理资源
        visualizer_node.cleanup()
        visualizer_node.destroy_node()
        rclpy.shutdown()
        print("Trajectory Visualizer Node stopped")


if __name__ == '__main__':
    main()
