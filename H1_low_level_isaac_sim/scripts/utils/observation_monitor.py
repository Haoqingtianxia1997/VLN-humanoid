#!/usr/bin/env python3
"""
Robot observation monitoring and analysis tool
Used to analyze motion control performance of H1 robot in Isaac Lab
"""

import numpy as np
import torch
import time
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from collections import deque


class ObservationMonitor:
    """Robot observation monitoring class"""
    
    def __init__(self, max_history: int = 1000, print_interval: int = 30):
        """
        Initialize observation monitor
        
        Args:
            max_history: Maximum number of historical records
            print_interval: Print interval (steps)
        """
        self.max_history = max_history
        self.print_interval = print_interval
        self.step_count = 0
        
        self.cum_velocity_sum = np.zeros(3)  # integration of velocities
        self.cum_time = 0.0                  # accumulated time
        self.avg_velocity = np.zeros(3)
        
        # Historical data storage
        self.history = {
            "timestamp": deque(maxlen=max_history),
            "target_vel": deque(maxlen=max_history),      # [vx, vy, wz]
            "actual_vel_b": deque(maxlen=max_history),    # Robot frame velocity
            "actual_vel_w": deque(maxlen=max_history),    # World frame velocity
            "position": deque(maxlen=max_history),        # [x, y, z]
            "orientation": deque(maxlen=max_history),     # [qw, qx, qy, qz]
            "gravity_proj": deque(maxlen=max_history),    # Gravity projection
            "tilt_angle": deque(maxlen=max_history),      # Tilt angle
            "vel_error": deque(maxlen=max_history),       # Velocity tracking error
            "height": deque(maxlen=max_history),          # Robot height
        }
        
        # Statistical information
        self.stats = {
            "total_distance": 0.0,
            "max_vel_error": 0.0,
            "max_tilt_angle": 0.0,
            "avg_height": 0.0,
            "start_time": time.time()
        }
        
        print("🔍 Observation monitor initialization complete")
        print(f"   - History capacity: {max_history}")
        print(f"   - Print interval: {print_interval} steps")
    
    def update(self, robot, env, target_cmd: List[float], step_dt: float):
        """
        Update observation data
        
        Args:
            robot: Robot asset object
            env: Environment object
            target_cmd: Target command [vx, vy, wz]
            step_dt: Time step
        """
        self.step_count += 1
        current_time = self.step_count * step_dt
        
        # Get observation data
        actual_lin_vel_b = robot.data.root_lin_vel_b[0]  # Robot frame linear velocity
        actual_ang_vel_b = robot.data.root_ang_vel_b[0]  # Robot frame angular velocity
        actual_lin_vel_w = robot.data.root_lin_vel_w[0]  # World frame linear velocity
        actual_ang_vel_w = robot.data.root_ang_vel_w[0]  # World frame angular velocity
        
        # integrate actual velocities and divide by time step to get average velocity
        actual_vel_now = np.array([
            actual_lin_vel_b[0].item(),
            actual_lin_vel_b[1].item(),
            actual_ang_vel_b[2].item()
        ])
        self.cum_velocity_sum += actual_vel_now * step_dt
        self.cum_time += step_dt

        actual_pos_w = robot.data.root_pos_w[0] - env.scene.env_origins[0]
        actual_quat_w = robot.data.root_quat_w[0]
        gravity_proj = robot.data.projected_gravity_b[0]
        
        # Calculate derived quantities
        tilt_angle = torch.acos(torch.clamp(-gravity_proj[2], -1, 1)) * 180 / 3.14159
        vel_error = torch.sqrt(
            (actual_lin_vel_b[0] - target_cmd[0])**2 + 
            (actual_lin_vel_b[1] - target_cmd[1])**2 + 
            (actual_ang_vel_b[2] - target_cmd[2])**2
        )
        
        # Store historical data
        self.history["timestamp"].append(current_time)
        self.history["target_vel"].append(target_cmd.copy())
        self.history["actual_vel_b"].append([
            actual_lin_vel_b[0].item(), 
            actual_lin_vel_b[1].item(), 
            actual_ang_vel_b[2].item()
        ])
        self.history["actual_vel_w"].append([
            actual_lin_vel_w[0].item(), 
            actual_lin_vel_w[1].item(), 
            actual_ang_vel_w[2].item()
        ])
        self.history["position"].append([
            actual_pos_w[0].item(), 
            actual_pos_w[1].item(), 
            actual_pos_w[2].item()
        ])
        self.history["orientation"].append([
            actual_quat_w[0].item(), 
            actual_quat_w[1].item(), 
            actual_quat_w[2].item(), 
            actual_quat_w[3].item()
        ])
        self.history["gravity_proj"].append([
            gravity_proj[0].item(), 
            gravity_proj[1].item(), 
            gravity_proj[2].item()
        ])
        self.history["tilt_angle"].append(tilt_angle.item())
        self.history["vel_error"].append(vel_error.item())
        self.history["height"].append(actual_pos_w[2].item())
        
        # Update statistical information
        self.stats["max_vel_error"] = max(self.stats["max_vel_error"], vel_error.item())
        self.stats["max_tilt_angle"] = max(self.stats["max_tilt_angle"], tilt_angle.item())
        if len(self.history["height"]) > 0:
            self.stats["avg_height"] = np.mean(list(self.history["height"]))
        
        # Calculate total walking distance
        if len(self.history["position"]) > 1:
            prev_pos = self.history["position"][-2]
            curr_pos = self.history["position"][-1]
            distance_delta = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            self.stats["total_distance"] += distance_delta
        
        # Periodic printing
        if self.step_count % self.print_interval == 0:
            self.print_status()
    
    def print_status(self):
        """Print current status"""
        if len(self.history["timestamp"]) == 0:
            return
            
        # Get latest data
        latest_time = self.history["timestamp"][-1]
        latest_target = self.history["target_vel"][-1]
        latest_actual = self.history["actual_vel_b"][-1]
        latest_pos = self.history["position"][-1]
        latest_tilt = self.history["tilt_angle"][-1]
        latest_vel_error = self.history["vel_error"][-1]
        latest_height = self.history["height"][-1]
        
        print(f"\n{'='*80}")
        print(f"🤖 Robot Status Monitor - Steps: {self.step_count}, Time: {latest_time:.2f}s")
        print(f"{'='*80}")
        
        # Velocity tracking
        print(f"📊 Velocity Tracking:")
        print(f"   Target velocity: vx={latest_target[0]:6.3f}, vy={latest_target[1]:6.3f}, wz={latest_target[2]:6.3f}")
        print(f"   Actual velocity: vx={latest_actual[0]:6.3f}, vy={latest_actual[1]:6.3f}, wz={latest_actual[2]:6.3f}")
        
        # === 新增：print 平均速度 ===
        if self.cum_time > 1e-6:
            self.avg_velocity = self.cum_velocity_sum / self.cum_time
        else:
            self.avg_velocity = np.zeros(3)
        print(f"   Average velocity: vx={self.avg_velocity[0]:6.3f}, vy={self.avg_velocity[1]:6.3f}, wz={self.avg_velocity[2]:6.3f}")
        
        print(f"   Velocity error: Δvx={latest_actual[0]-latest_target[0]:6.3f}, Δvy={latest_actual[1]-latest_target[1]:6.3f}, Δwz={latest_actual[2]-latest_target[2]:6.3f}")
        print(f"   Error magnitude: {latest_vel_error:.3f} (max: {self.stats['max_vel_error']:.3f})")
        
        # Position and pose
        print(f"📍 Position and Pose:")
        print(f"   World position: x={latest_pos[0]:6.3f}, y={latest_pos[1]:6.3f}, z={latest_pos[2]:6.3f}")
        print(f"   Tilt angle: {latest_tilt:5.1f}° (max: {self.stats['max_tilt_angle']:5.1f}°)")
        print(f"   Robot height: {latest_height:.3f}m (avg: {self.stats['avg_height']:.3f}m)")
        
        # Motion statistics
        print(f"📈 Motion Statistics:")
        print(f"   Total walking distance: {self.stats['total_distance']:.3f}m")
        print(f"   Runtime: {time.time() - self.stats['start_time']:.1f}s")
        
        # Performance evaluation
        print(f"⚡ Performance Evaluation:")
        if latest_vel_error < 0.1:
            print(f"   Velocity tracking: ✅ Excellent (error < 0.1)")
        elif latest_vel_error < 0.3:
            print(f"   Velocity tracking: ⚠️  Good (error < 0.3)")
        else:
            print(f"   Velocity tracking: ❌ Needs improvement (error > 0.3)")
            
        if latest_tilt < 10:
            print(f"   Pose stability: ✅ Excellent (tilt < 10°)")
        elif latest_tilt < 20:
            print(f"   Pose stability: ⚠️  Average (tilt < 20°)")
        else:
            print(f"   Pose stability: ❌ Unstable (tilt > 20°)")
        
        print(f"{'='*80}")
    
    def save_data(self, filename: str = "robot_observation_log.json"):
        """Save observation data to file"""
        data = {
            "metadata": {
                "total_steps": self.step_count,
                "max_history": self.max_history,
                "save_time": time.time(),
                "stats": self.stats
            },
            "history": {k: list(v) for k, v in self.history.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Observation data saved to: {filename}")
    
    def plot_tracking_performance(self, save_plot: bool = True):
        """Plot velocity tracking performance charts"""
        if len(self.history["timestamp"]) < 10:
            print("Insufficient data to plot charts")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        times = list(self.history["timestamp"])
        
        # Velocity tracking
        target_vels = np.array(list(self.history["target_vel"]))
        actual_vels = np.array(list(self.history["actual_vel_b"]))
        
        axes[0, 0].plot(times, target_vels[:, 0], 'r--', label='Target vx', linewidth=2)
        axes[0, 0].plot(times, actual_vels[:, 0], 'r-', label='Actual vx', linewidth=1)
        axes[0, 0].plot(times, target_vels[:, 1], 'g--', label='Target vy', linewidth=2)
        axes[0, 0].plot(times, actual_vels[:, 1], 'g-', label='Actual vy', linewidth=1)
        axes[0, 0].set_title('Linear Velocity Tracking')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Angular velocity tracking
        axes[0, 1].plot(times, target_vels[:, 2], 'b--', label='Target wz', linewidth=2)
        axes[0, 1].plot(times, actual_vels[:, 2], 'b-', label='Actual wz', linewidth=1)
        axes[0, 1].set_title('Angular Velocity Tracking')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angular velocity (rad/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Position trajectory
        positions = np.array(list(self.history["position"]))
        axes[1, 0].plot(positions[:, 0], positions[:, 1], 'k-', linewidth=2)
        axes[1, 0].plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        axes[1, 0].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        axes[1, 0].set_title('2D Trajectory')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axis('equal')
        
        # Performance metrics
        vel_errors = list(self.history["vel_error"])
        tilt_angles = list(self.history["tilt_angle"])
        
        ax2 = axes[1, 1]
        ax3 = ax2.twinx()
        
        line1 = ax2.plot(times, vel_errors, 'r-', label='Velocity error', linewidth=2)
        line2 = ax3.plot(times, tilt_angles, 'b-', label='Tilt angle', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity error', color='r')
        ax3.set_ylabel('Tilt angle (°)', color='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax3.tick_params(axis='y', labelcolor='b')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True)
        ax2.set_title('Performance Metrics')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('robot_tracking_performance.png', dpi=300, bbox_inches='tight')
            print("📊 Performance chart saved as: robot_tracking_performance.png")
        
        plt.show()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if len(self.history["vel_error"]) == 0:
            return {}
        
        vel_errors = list(self.history["vel_error"])
        tilt_angles = list(self.history["tilt_angle"])
        
        summary = {
            "velocity_tracking": {
                "mean_error": np.mean(vel_errors),
                "max_error": np.max(vel_errors),
                "std_error": np.std(vel_errors),
                "good_tracking_ratio": np.mean(np.array(vel_errors) < 0.2)  # Ratio of errors less than 0.2
            },
            "stability": {
                "mean_tilt": np.mean(tilt_angles),
                "max_tilt": np.max(tilt_angles),
                "stable_ratio": np.mean(np.array(tilt_angles) < 15)  # Ratio of tilts less than 15 degrees
            },
            "motion": {
                "total_distance": self.stats["total_distance"],
                "avg_height": self.stats["avg_height"],
                "duration": self.history["timestamp"][-1] if self.history["timestamp"] else 0
            }
        }
        
        return summary


# Usage examples and test code
if __name__ == "__main__":
    # Create monitor
    monitor = ObservationMonitor(max_history=500, print_interval=10)
    
    # Simulate data test
    print("🧪 Running simulated data test...")
    
    for i in range(100):
        # Simulate robot data (in actual use, this data comes from Isaac Lab)
        time.sleep(0.01)  # Simulate time step
        
        # This is just for testing, in actual use you need to pass real robot and env objects
        # monitor.update(robot, env, [1.0, 0.0, 0.0], 0.01)
    
    print("Test complete!")
