#!/usr/bin/env python3

"""
简化版IMU轨迹绘制脚本
功能：订阅IMU数据，实时绘制机器人在X-Y平面的运动轨迹
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time

class SimpleTrajectoryTracker(Node):
    def __init__(self):
        super().__init__('simple_trajectory_tracker')
        
        # 订阅IMU数据
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        
        # 状态变量
        self.position = np.array([0.0, 0.0])  # X-Y位置
        self.velocity = np.array([0.0, 0.0])  # X-Y速度
        self.last_time = None
        
        # 轨迹数据
        self.trajectory = deque(maxlen=500)  # 保存最近500个点
        self.trajectory.append([0.0, 0.0])
        
        # 数据锁
        self.data_lock = threading.Lock()
        
        # 滤波和校准
        self.init_samples = 0
        self.acc_bias = np.array([0.0, 0.0])
        self.is_calibrated = False
        
        self.get_logger().info('🎯 简化轨迹跟踪器启动')
        self.get_logger().info('正在校准传感器...')
        
    def imu_callback(self, msg):
        """IMU数据回调"""
        current_time = time.time()
        
        # 提取X-Y加速度（忽略Z轴）
        acc_x = msg.linear_acceleration.x
        acc_y = msg.linear_acceleration.y
        
        # 传感器校准（前50个样本）
        if not self.is_calibrated:
            if self.init_samples < 50:
                self.acc_bias[0] += acc_x
                self.acc_bias[1] += acc_y
                self.init_samples += 1
                return
            else:
                self.acc_bias /= 50
                self.is_calibrated = True
                self.last_time = current_time
                self.get_logger().info(f'✅ 校准完成！偏置: [{self.acc_bias[0]:.3f}, {self.acc_bias[1]:.3f}]')
                return
        
        # 计算时间间隔
        if self.last_time is None:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt <= 0 or dt > 0.2:  # 跳过异常时间间隔
            self.last_time = current_time
            return
        
        with self.data_lock:
            # 偏置补偿
            acc_corrected = np.array([acc_x - self.acc_bias[0], 
                                    acc_y - self.acc_bias[1]])
            
            # 简单阈值过滤（减少噪声积累）
            acc_threshold = 0.3
            acc_corrected = np.where(np.abs(acc_corrected) > acc_threshold, 
                                   acc_corrected, 0.0)
            
            # 速度积分
            self.velocity += acc_corrected * dt
            
            # 应用阻尼（模拟摩擦）
            damping = 0.98
            self.velocity *= damping
            
            # 位置积分
            self.position += self.velocity * dt
            
            # 保存轨迹点
            self.trajectory.append(self.position.copy())
        
        self.last_time = current_time
        
        # 定期打印状态
        if not hasattr(self, '_last_print'):
            self._last_print = current_time
        elif current_time - self._last_print > 2.0:
            distance = np.linalg.norm(self.position)
            speed = np.linalg.norm(self.velocity)
            self.get_logger().info(
                f'🧭 位置: [{self.position[0]:.2f}, {self.position[1]:.2f}]m, '
                f'距起点: {distance:.2f}m, 速度: {speed:.2f}m/s'
            )
            self._last_print = current_time

class RealTimeVisualizer:
    def __init__(self, tracker):
        self.tracker = tracker
        
        # 创建图形界面
        plt.style.use('seaborn-v0_8')  # 使用更美观的样式
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        self.ax.set_title('机器人实时运动轨迹', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X 位置 (米)', fontsize=12)
        self.ax.set_ylabel('Y 位置 (米)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 绘图元素
        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2, 
                                           alpha=0.7, label='运动轨迹')
        self.current_point, = self.ax.plot([], [], 'ro', markersize=10, 
                                         label='当前位置')
        self.start_point, = self.ax.plot([0], [0], 'go', markersize=8, 
                                        label='起始点')
        
        # 轨迹历史（淡化显示）
        self.trail_points, = self.ax.plot([], [], 'b.', alpha=0.3, 
                                         markersize=3)
        
        self.ax.legend(loc='upper right')
        
        # 初始设置
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        
        # 创建动画
        self.ani = FuncAnimation(self.fig, self.update_plot, 
                               interval=50, blit=False)
        
        # 添加文本显示
        self.status_text = self.ax.text(0.02, 0.98, '', 
                                       transform=self.ax.transAxes,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', 
                                               facecolor='wheat', 
                                               alpha=0.8),
                                       fontsize=10)
        
    def update_plot(self, frame):
        """更新绘图"""
        with self.tracker.data_lock:
            if len(self.tracker.trajectory) < 2:
                return
            
            trajectory = np.array(list(self.tracker.trajectory))
            
            # 更新轨迹线
            self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
            
            # 更新当前位置
            current_pos = trajectory[-1]
            self.current_point.set_data([current_pos[0]], [current_pos[1]])
            
            # 更新历史轨迹点
            if len(trajectory) > 10:
                trail = trajectory[:-10]  # 除了最近10个点
                self.trail_points.set_data(trail[:, 0], trail[:, 1])
            
            # 动态调整视图范围
            if len(trajectory) > 5:
                margin = 1.0
                x_min, x_max = np.min(trajectory[:, 0]) - margin, np.max(trajectory[:, 0]) + margin
                y_min, y_max = np.min(trajectory[:, 1]) - margin, np.max(trajectory[:, 1]) + margin
                
                # 确保最小视图范围
                x_range = max(x_max - x_min, 2.0)
                y_range = max(y_max - y_min, 2.0)
                
                x_center = (x_max + x_min) / 2
                y_center = (y_max + y_min) / 2
                
                self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
                self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            
            # 更新状态信息
            distance = np.linalg.norm(current_pos)
            speed = np.linalg.norm(self.tracker.velocity)
            points_count = len(trajectory)
            
            status_info = (f'轨迹点数: {points_count}\n'
                          f'当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f})m\n'
                          f'距起点: {distance:.2f}m\n'
                          f'当前速度: {speed:.2f}m/s')
            
            self.status_text.set_text(status_info)
    
    def show(self):
        """显示可视化界面"""
        plt.tight_layout()
        plt.show()

def main():
    rclpy.init()
    
    # 创建轨迹跟踪器
    tracker = SimpleTrajectoryTracker()
    
    # 创建可视化器
    visualizer = RealTimeVisualizer(tracker)
    
    # 在后台线程运行ROS2节点
    def ros_spin():
        try:
            rclpy.spin(tracker)
        except Exception as e:
            print(f"ROS错误: {e}")
    
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()
    
    print("\n🎯 IMU轨迹可视化工具")
    print("=" * 40)
    print("📊 实时绘制机器人运动轨迹")
    print("🎮 控制说明：")
    print("   - 等待传感器校准完成")
    print("   - 蓝线显示运动轨迹")  
    print("   - 红点表示当前位置")
    print("   - 绿点表示起始位置")
    print("   - 关闭窗口退出程序")
    print("=" * 40)
    
    try:
        visualizer.show()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        tracker.destroy_node()
        rclpy.shutdown()
        print("程序已退出")

if __name__ == '__main__':
    main()
