# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates wheeled robot navigation in a prebuilt USD environment.

In this example, we use a differential drive wheeled robot (Jetbot-style) for navigation.
The robot can be controlled via UDP commands for real-time control.

.. code-block:: bash

    # Run the script with a wheeled robot
    ./isaaclab.sh -p isaac_inference_wheeled_robot.py --udp

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on wheeled robot navigation in a warehouse.")
parser.add_argument("--udp", action="store_true", help="Enable UDP command listener for real-time control")
parser.add_argument("--udp-host", type=str, default="localhost", help="UDP listener host address")
parser.add_argument("--udp-port", type=int, default=12345, help="UDP listener port")
parser.add_argument("--infinite-episode", action="store_true", help="Completely disable episode timeout")
parser.add_argument("--enable-trajectory", action="store_true", help="Enable trajectory following")
parser.add_argument("--trajectory-scale", type=float, default=3.0, help="Scale factor for trajectory size")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli, ros2=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import math
import torch
import numpy as np
from datetime import datetime
import dataclasses

import omni
import cv2

from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from ros.isaac_publisher import InfoPublisher
import rclpy

# Import the simplified UDP listener for wheeled robots
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.udp.udp_cmd_listener_wheeled import UDPCmdListener 

# Configure the wheeled robot (Jetbot-style differential drive)
WHEELED_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # Start slightly above ground
        joint_pos={".*": 0.0},  # All joints start at 0
        joint_vel={".*": 0.0},  # All joints start with 0 velocity
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # Both wheel joints
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,  # For velocity control
            damping=10.0,
        ),
    },
)

# Simple scene configuration for wheeled robot
@dataclasses.dataclass
class WheeledRobotSceneCfg(InteractiveSceneCfg):
    """Configuration for wheeled robot scene."""
    
    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
    )
    
    # Robot
    robot: ArticulationCfg = WHEELED_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Camera mounted on robot
    camera = CameraCfg(
        prim_path="/World/envs/env_0/Robot/chassis/front_cam",
        update_period=0.02,
        width=1024, height=768,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.5),  # Mounted on front of chassis
            rot=(0.45451948, -0.54167522, 0.54167522, -0.45451948),
            # rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

import dataclasses

def differential_drive_control(vx: float, wz: float) -> tuple:
    """
    Convert linear and angular velocities to wheel velocities for differential drive.
    
    Args:
        vx: Linear velocity in x direction (m/s)
        wz: Angular velocity around z axis (rad/s)
        
    Returns:
        tuple: (left_wheel_vel, right_wheel_vel) in rad/s
    """
    # Robot parameters (adjust based on actual robot)
    wheel_radius = 0.032  # meters (Jetbot wheel radius)
    wheel_separation = 0.114  # meters (distance between wheels)
    
    # Calculate wheel velocities
    left_wheel_vel = (vx - wz * wheel_separation / 2.0) / wheel_radius
    right_wheel_vel = (vx + wz * wheel_separation / 2.0) / wheel_radius
    
    return left_wheel_vel, right_wheel_vel

def main():
    """Main function."""
    
    rclpy.init()

    # Initialize UDP command listener (if enabled)
    udp_listener = None
    if args_cli.udp:
        udp_listener = UDPCmdListener(host=args_cli.udp_host, port=args_cli.udp_port)
        udp_listener.start()
        print(f"🚗 UDP wheeled robot control enabled, listening address: {args_cli.udp_host}:{args_cli.udp_port}")
        print("Command format: echo 'vx vy wz' | nc -u localhost 12345")
        print("Example: echo '0.5 0.0 0.2' | nc -u localhost 12345")
        print("Note: vy is ignored for differential drive robot")
        print("Use: python udp_cmd_client_wheeled.py for specialized control")
    else:
        print("📝 Using hardcoded command control (use --udp to enable UDP control)")

    # Set simulation parameters FIRST
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Setup scene configuration after simulation context
    scene_cfg = WheeledRobotSceneCfg(num_envs=1, env_spacing=2.0)
    
    # Create scene
    scene = InteractiveScene(scene_cfg)
    
    camera_publisher = InfoPublisher()

    # Set camera view
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # Play the simulator
    sim.reset()
    
    # Initialize counters and timestamps
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Initialize fixed folder name for image saving
    image_counter = 0
    
    print("Starting wheeled robot navigation...")
    print(f"📸 Camera images will be saved to: camera_feed/wheeled/")
    print(f"    RGB images: camera_feed/wheeled/rgb/")
    print(f"    Depth data: camera_feed/wheeled/depth/")
    print(f"    Format: rgb_XXXXXX_HHMMSS_mmm.png, depth_XXXXXX_HHMMSS_mmm.npy")
    
    # Control variables
    target_vx, target_wz = 0.0, 0.0  # Linear and angular velocities
    
    while simulation_app.is_running():
        # Get target velocities
        if udp_listener:
            # UDP control mode - get wheeled robot commands
            cmd = udp_listener.get_current_cmd()
            target_vx, target_wz = cmd[0], cmd[3]  # vx and wz from [vx, vy, heading, wz]
        else:
            # Demonstration mode - simple forward motion with turning
            if count % 300 < 150:
                target_vx, target_wz = 0.3, 0.0  # Move forward
            elif count % 300 < 200:
                target_vx, target_wz = 0.0, 0.5  # Turn in place
            else:
                target_vx, target_wz = 0.0, 0.0  # Stop
        
        # Convert to wheel velocities for differential drive
        left_wheel_vel, right_wheel_vel = differential_drive_control(target_vx, target_wz)
        
        # Apply wheel velocities (assuming 2 wheels: left and right)
        wheel_actions = torch.tensor([[left_wheel_vel, right_wheel_vel]], dtype=torch.float32)
        scene["robot"].set_joint_velocity_target(wheel_actions)
        
        # Write data to simulation
        scene.write_data_to_sim()
        
        # Step simulation
        sim.step()
        
        # Update scene
        scene.update(sim_dt)
        
        # Get camera data and save images (if camera exists)
        try:
            cam = scene["camera"]
            rgb = cam.data.output["rgb"]
            depth = cam.data.output["depth"]

            rgb0 = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
            depth0 = depth[0].cpu().numpy() if depth.dim() == 4 else depth.cpu().numpy()
            
            rgb_image = rgb0[..., :3]  # Take only RGB channels
            camera_publisher.publish_rgb(rgb_image)
            
            # Process depth image
            depth_image = depth0.squeeze()
            camera_publisher.publish_depth(depth_image)
            
            # Spin ROS2 node to process callbacks (including subscribers)
            rclpy.spin_once(camera_publisher, timeout_sec=0.001)
            
            # Create camera_feed directory with fixed "wheeled" subdirectory and separate rgb/depth folders
            camera_dir = f"camera_feed/wheeled"
            rgb_dir = f"{camera_dir}/rgb"
            depth_dir = f"{camera_dir}/depth"
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            
            # Generate timestamped filenames
            current_time = datetime.now().strftime("%H%M%S_%f")[:-3]  # 精确到毫秒
            rgb_filename = f"{rgb_dir}/rgb_{image_counter:06d}_{current_time}.png"
            depth_npy_filename = f"{depth_dir}/depth_{image_counter:06d}_{current_time}.npy"
            
            # Save RGB image
            rgb_image = rgb0[..., :3]
            # cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # Save depth data as .npy file
            # np.save(depth_npy_filename, depth0.squeeze())
            
            # Increment image counter
            image_counter += 1
            
            # Print status every 100 frames
            if image_counter % 100 == 0:
                print(f"📸 Saved {image_counter} image pairs | Robot cmd: vx={target_vx:.2f}, wz={target_wz:.2f}")
        except:
            # Camera might not be available, continue without saving images
            pass
        
        # Update counters
        sim_time += sim_dt
        count += 1

    # Clean up UDP listener
    if udp_listener:
        udp_listener.stop()
        print("🛑 UDP wheeled robot listener stopped")
    
    print(f"📸 Session completed! Total saved: {image_counter} image pairs")
    print(f"    RGB images: camera_feed/wheeled/rgb/")
    print(f"    Depth data: camera_feed/wheeled/depth/")


if __name__ == "__main__":
    main()
    simulation_app.close()
