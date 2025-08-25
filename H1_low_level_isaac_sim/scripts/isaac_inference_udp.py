# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--udp", action="store_true", help="Enable UDP command listener for real-time control")
parser.add_argument("--udp-host", type=str, default="localhost", help="UDP listener host address")
parser.add_argument("--udp-port", type=int, default=12345, help="UDP listener port")
parser.add_argument("--infinite-episode", action="store_true", help="Completely disable episode timeout")
parser.add_argument("--enable-trajectory", action="store_true", help="Enable MPC trajectory following")
parser.add_argument("--trajectory-scale", type=float, default=3.0, help="Scale factor for trajectory size")
parser.add_argument("--hover-time", type=float, default=10.0, help="Time in seconds for robot to hover before landing (for camera adjustment)")

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
import torch
import rclpy
import numpy as np
from datetime import datetime
from ros.isaac_publisher import InfoPublisher
from std_msgs.msg import Bool

import omni

from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.sensors.imu.imu_data import ImuData
import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
# from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY

from isaaclab_assets import H1_CFG 

from isaacsim.core.utils.extensions import enable_extension
import omni.graph.core as og 

from utils.udp.udp_cmd_listener import UDPCmdListener 
from utils.observation_monitor import ObservationMonitor 
from utils.mpc_follow_trajectory import mpc_control, get_robot_state, create_trajectory_visualization, print_trajectory_info, generate_track, dt
from utils.pid_control import pid_control_wz, correct_velocity_command
from ros.isaac_ros_graphs import setup_ros2_bridge_for_rgbd

# camera light in isaac sim by default
import omni.kit.actions.core
action_registry = omni.kit.actions.core.get_action_registry()
action = action_registry.get_action(
    "omni.kit.viewport.menubar.lighting",
    "set_lighting_mode_camera"
)
action.execute()


def main():
    """Main function."""
    # Initialize ROS2
    rclpy.init()
    
    # Create camera publisher
    info_publisher = InfoPublisher()
    
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)
    # load the trained jit policy for standing

    policy_path_stand = os.path.abspath("policy/policy_stand_rough.pt")
    file_content_stand = omni.client.read_file(policy_path_stand)[2]
    file_stand = io.BytesIO(memoryview(file_content_stand).tobytes())
    policy_stand = torch.jit.load(file_stand, map_location=args_cli.device)

    # Initialize UDP command listener (if enabled)
    udp_listener = None
    if args_cli.udp:
        udp_listener = UDPCmdListener(host=args_cli.udp_host, port=args_cli.udp_port)
        udp_listener.start()
        print(f"🎧 UDP control enabled, listening address: {args_cli.udp_host}:{args_cli.udp_port}")
        print("Command format: echo 'vx vy wz' | nc -u localhost 12345")
        print("Example: echo '1.0 0.0 0.2' | nc -u localhost 12345")
    else:
        print("📝 Using hardcoded command control (use --udp to enable UDP control)")

    # Initialize trajectory following (if enabled)
    trajectory = None
    if args_cli.enable_trajectory:
        trajectory = generate_track(scale=args_cli.trajectory_scale)
        print(f"🎯 MPC trajectory following enabled with scale={args_cli.trajectory_scale}")
        print(f"   Trajectory has {len(trajectory)} points")
    else:
        print("📝 MPC trajectory following disabled (use --enable-trajectory to enable)")

    # setup environment
    env_cfg = H1RoughEnvCfg_PLAY()
    # env_cfg = H1FlatEnvCfg_PLAY()
    
    env_cfg.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    
    # Fix robot initial orientation (remove random yaw)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},  # Fixed yaw at 0.0
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    
    # Disable debug visualization arrows for velocity commands
    env_cfg.commands.base_velocity.debug_vis = False
    
    # Ensure robot starts in a stable configuration
    # Disable some termination conditions that might interfere with initialization
    if hasattr(env_cfg, 'terminations'):
        if hasattr(env_cfg.terminations, 'base_contact'):
            env_cfg.terminations.base_contact = None  # Disable base contact termination during init
    
    # infinite episode length
    if args_cli.infinite_episode:
        env_cfg.episode_length_s = float('inf')  # Set to infinity
        # Remove timeout termination
        env_cfg.terminations.time_out = None
        print("Infinite episode mode enabled - no automatic timeout!")
        
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd",
    )
    
    env_cfg.scene.camera = CameraCfg(
        prim_path="/World/envs/env_0/Robot/torso_link/front_cam",
        update_period=0.02,
        width=1024, height=768,
        data_types=["rgb", "depth"],   # "depth" or "distance_to_image_plane" both work
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, # aperture from tutorial https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/cameras/create-perspective-camera.html
            clipping_range=(0.1, 100000), # depth range limit
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.7),
            # quaternion format (w, x, y, z)
            # rot=(0.7071, 0, 0, -0.7071),  # No rotation(rotation around z axis -90 degrees, make camera x axis forward and y axis pointing left, z axis always keep upwards)
            # rot=(0.5, -0.5, 0.5, -0.5),  # z axis forward
            rot=(0.45451948, -0.54167522, 0.54167522, -0.45451948), # camera z axis downwards 10 degrees, quaternion format (w, x, y, z)
            convention="ros",
        ),
    )
    
    # Add IMU sensor configuration
    env_cfg.scene.imu = ImuCfg(
        prim_path="/World/envs/env_0/Robot/torso_link",
        update_period=0.01,  # 100Hz update rate for IMU
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # IMU at torso center
            rot=(1, 0, 0, 0),  # No rotation relative to torso
        ),
        gravity_bias=(0.0, 0.0, 9.81),  # Standard gravity bias
    )
    
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)


    # enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.core.nodes")

    # run inference with the policy
    obs, _ = env.reset()

    # Phase 1: Hovering initialization - robot stays in air for camera adjustment
    hover_time = 10.0
    hover_steps = int(hover_time / env.step_dt)
    
    robot = env.scene["robot"]

    setup_ros2_bridge_for_rgbd(env)
    
    # Create trajectory visualization if trajectory following is enabled
    if args_cli.enable_trajectory and trajectory is not None:
        print_trajectory_info(trajectory)
        try:
            create_trajectory_visualization(env, trajectory, stride=10)
        except Exception as e:
            print(f"⚠️ Failed to create trajectory visualization: {e}")
            print("   Continuing without trajectory visualization...")

    # Initialize observation monitor
    obs_monitor = ObservationMonitor(max_history=1000, print_interval=30)
    
    # Initialize image counter for publishing statistics
    image_counter = 0
    
    print("Monitoring robot observations...")
    print(f"📸 Camera images will be published via ROS2:")
    print(f"    RGB topic: /camera/rgb/image_raw")
    print(f"    Depth topic: /camera/depth/image_raw")
    print(f"🎯 IMU data will be published via ROS2:")
    print(f"    IMU topic: /imu/data")
    print(f"🤖 Robot root quaternion will be published via ROS2:")
    print(f"    Root quaternion topic: /robot/root_quaternion")
    print(f"📍 Robot pose subscription active:")
    print(f"    Pose topic: /robot/pose")
    print(f"🛤️ Trajectory subscription active:")
    print(f"    Trajectory topic: /robot/trajectory")
    print(f"    Use 'ros2 topic echo' or 'rviz2' to view the data")
    print(f"    Use 'ros2 topic pub' to send pose/trajectory data")

    # Initialize MPC step counter
    mpc_step_counter = 0
    MPC_STEPS = int(round(dt / env.step_dt))  # MPC steps per control step
    current_mpc_cmd = np.array([0.0, 0.0, 0.0])  # [vx, vy, wz]
    target_vx, target_vy, target_wz = 0.0, 0.0, 0.0  # Initialize target velocities
    
    current_pose = [0,0,0,1,0,0,0]  # Initialize current pose list for MPC
    last_goal_position = []
    # Initialize policy transition state machine
    INITIALIZE = "initialize"
    POLICY_STATE_STANDING = "standing"
    POLICY_STATE_WALKING = "walking"
    POLICY_STATE_SLOWING_DOWN = "slowing_down"
    POLICY_STATE_STARTING_UP = "starting_up"

    policy_state = INITIALIZE  # Start in initialize state
    transition_timer = 0.0  # Timer for transition states
    TRANSITION_DURATION = 2.0  # 2 seconds transition time
    prev_target_nonzero = False  # Track if previous target was non-zero
    start_capture = False  # Flag to start capturing images
    continue_mpc = False  # Flag to continue MPC
    
    # Initialize observation commands to zero (standing commands)
    print("🔧 Initializing observation commands...")
    obs["policy"][:, 9:13] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)
    
 
    loop_index = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            loop_index += 1
            # Get robot for observation monitoring
            robot = env.scene["robot"]

            if trajectory is not None and continue_mpc:
                # MPC trajectory following mode
                if mpc_step_counter % MPC_STEPS == 0:
                    # Update MPC control command
                    robot_state, _ = get_robot_state(robot)
                    # TODO:
                    robot_state[:2] = np.array(current_pose[:2])
                    
                    current_mpc_cmd = mpc_control(robot_state, trajectory)
                    print(f"🎯 MPC: robot_state=[{robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f}], cmd=[{current_mpc_cmd[0]:.2f}, {current_mpc_cmd[1]:.2f}, {current_mpc_cmd[2]:.2f}]")


                # every MPC_STEPS, apply the command
                target_vx, target_vy, target_wz = current_mpc_cmd

                cmd_tensor = correct_velocity_command(target_vx, target_vy, -target_wz, obs_monitor, args_cli, env)

            elif udp_listener:
                # UDP control mode
                target_vx, target_vy, _, target_wz = udp_listener.get_current_cmd()
                cmd_tensor = correct_velocity_command(target_vx, target_vy, -target_wz, obs_monitor, args_cli, env)

            else:
                target_vx, target_vy, target_wz = 0.0, 0.0, 0.0
                cmd_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)

            # Check if target is non-zero (excluding trajectory mode logic)
            current_target_nonzero = (target_vx != 0.0 or target_vy != 0.0 or target_wz != 0.0) or (trajectory is not None and continue_mpc)
            
            # Policy transition state machine
            if policy_state == INITIALIZE:
                # Initial state, wait for robot to stabilize
                start_capture = True
                if loop_index > hover_steps:
                    policy_state = POLICY_STATE_STANDING
                    print("🤖 Robot is now initialized and standing!")
                    transition_timer = 0.0
            elif policy_state == POLICY_STATE_STANDING:
                if current_target_nonzero:
                    # Transition from standing to starting up
                    policy_state = POLICY_STATE_STARTING_UP
                    transition_timer = 0.0
                    print("🚶 Transitioning from standing to walking (starting up)")
                    
            elif policy_state == POLICY_STATE_WALKING:
                if not current_target_nonzero:
                    # Transition from walking to slowing down
                    policy_state = POLICY_STATE_SLOWING_DOWN
                    transition_timer = 0.0
                    print("🛑 Transitioning from walking to standing (slowing down)")
                    
            elif policy_state == POLICY_STATE_SLOWING_DOWN:
                transition_timer += env.step_dt
                if transition_timer >= TRANSITION_DURATION:
                    # Transition complete, switch to standing
                    policy_state = POLICY_STATE_STANDING
                    print("🧍 Transition complete: now standing")
                elif current_target_nonzero:
                    # Target changed back to non-zero during slowing down
                    policy_state = POLICY_STATE_WALKING
                    print("🚶 Target changed back to non-zero, resuming walking")
                    
            elif policy_state == POLICY_STATE_STARTING_UP:
                transition_timer += env.step_dt
                if transition_timer >= TRANSITION_DURATION:
                    # Transition complete, switch to walking
                    policy_state = POLICY_STATE_WALKING
                    print("🚶 Transition complete: now walking")
                elif not current_target_nonzero:
                    # Target changed back to zero during starting up
                    policy_state = POLICY_STATE_STANDING
                    print("🧍 Target changed back to zero, returning to standing")
            
            if policy_state != INITIALIZE:
                # Select policy and determine actual command based on state
                if policy_state == POLICY_STATE_STANDING:
                    action = policy_stand(obs["policy"])
                    actual_cmd_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)
                    
                elif policy_state == POLICY_STATE_SLOWING_DOWN:
                    action = policy(obs["policy"])
                    # Use zero velocity command during slowing down
                    actual_cmd_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)
                    
                elif policy_state == POLICY_STATE_STARTING_UP:
                    action = policy(obs["policy"])
                    # Use zero velocity command during starting up
                    actual_cmd_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)

                elif policy_state == POLICY_STATE_WALKING:
                    action = policy(obs["policy"])
                    actual_cmd_tensor = cmd_tensor
                
                obs, _, _, _, _ = env.step(action)

                target_cmd = [target_vx, target_vy, target_wz]
                obs_monitor.update(robot, env, target_cmd, env.step_dt)
                
                mpc_step_counter += 1
                
            else:
                root_state = robot.data.root_state_w.clone()
        
                # Maintain hovering position (freeze in air)
                # Keep the robot at a fixed height and prevent falling
                root_state[:, 2] = 1.05  # Fixed Z height (torso at 1.05m)
                root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args_cli.device)  # Fixed orientation (w,x,y,z)
                root_state[:, 7:] = 0.0  # Zero velocities
                
                # Apply the hovering state
                robot.write_root_state_to_sim(root_state)
                
                # Keep joints in default standing position without running policy
                # Just step the simulation to maintain the hovering state
                env.sim.step()
                actual_cmd_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], device=args_cli.device)
                
            
            
            
            

            cam: CameraCfg = env.scene["camera"]
            rgb: torch.Tensor = cam.data.output["rgb"]
            depth: torch.Tensor = cam.data.output["depth"]

            rgb0 = rgb[0].cpu().numpy() if rgb.dim() == 4 else rgb.cpu().numpy()
            depth0 = depth[0].cpu().numpy() if depth.dim() == 4 else depth.cpu().numpy()


            # Get IMU data
            imu: ImuCfg = env.scene["imu"]
            imu_data: ImuData = imu.data



            # # Create camera_feed directory with fixed "legged" subdirectory and separate rgb/depth folders
            # camera_dir = f"camera_feed/legged"
            # rgb_dir = f"{camera_dir}/rgb"
            # depth_dir = f"{camera_dir}/depth"
            # os.makedirs(rgb_dir, exist_ok=True)
            # os.makedirs(depth_dir, exist_ok=True)

            # if start_capture:
            #     # Generate timestamped filenames
            #     current_time = datetime.now().strftime("%H%M%S_%f")[:-3]  # 精确到毫秒
            #     rgb_filename = f"{rgb_dir}/rgb_{image_counter:06d}_{current_time}.png"
            #     depth_npy_filename = f"{depth_dir}/depth_{image_counter:06d}_{current_time}.npy"
            #     # depth_png_filename = f"{depth_dir}/depth_{image_counter:06d}_{current_time}.png"
                
            #     # Save RGB image
            #     rgb_image = rgb0[..., :3]
            #     cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                
            #     # Save depth data as .npy file
            #     np.save(depth_npy_filename, depth0.squeeze())
                
            #     # save depth as a grayscale image
            #     depth_vis = depth0.squeeze()
            #     # filter out invalid depth values
            #     depth_valid = depth_vis[(depth_vis > 0) & np.isfinite(depth_vis)]
            #     if depth_valid.size > 0:
            #         dmin, dmax = depth_valid.min(), depth_valid.max()
            #     else:
            #         dmin, dmax = 0, 1

            #     # from 0-1 to 0-255
            #     depth_norm = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
            #     depth_uint8 = (depth_norm * 255).astype(np.uint8)

            #     # cv2.imwrite("camera_feed/depth.png", depth_uint8)

            #     # # alternatively save depth as a color map, uncomment the following lines and comment the above cv2.imwrites
            #     # depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            #     # cv2.imwrite(depth_png_filename, depth_color)
                
            #     # Increment image counter
            #     image_counter += 1
                
            #     # Print image saving statistics every 100 frames
            #     if image_counter % 100 == 0:
            #         print(f"📸 Saved {image_counter} image pairs to camera_feed/legged/rgb/ and /depth/")



            if start_capture:
                # Publish RGB and depth images via ROS2
                # if not first_frame_quat_published:
                robot_state, root_pose = get_robot_state(robot)
                        # first_frame_quat_published = True

                # Publish first frame root pose
                info_publisher.publish_root_pose(root_pose)

                # Process RGB image
                rgb_image = rgb0[..., :3]  # Take only RGB channels
                info_publisher.publish_rgb(rgb_image)
                
                # Process depth image
                depth_image = depth0.squeeze()
                info_publisher.publish_depth(depth_image)
                
                # Publish IMU data
                info_publisher.publish_imu(imu_data)
                
                # Spin ROS2 node to process callbacks (including subscribers)
                rclpy.spin_once(info_publisher, timeout_sec=0.001)

                pose_data = info_publisher.get_current_pose()
                
                if not continue_mpc:
                    info_publisher.flag_publisher_.publish(Bool(data=True))
                else:
                    info_publisher.flag_publisher_.publish(Bool(data=False))
                
                trajectory_data = info_publisher.get_current_trajectory()
                
                # TODO: use current_pose from VSLAM to do MPC control
                if pose_data is not None:
                    # Convert PoseStamped to [x, y, z, qw, qx, qy, qz] list format
                    current_pose = [
                        pose_data.pose.position.x,
                        pose_data.pose.position.y, 
                        pose_data.pose.position.z,
                        pose_data.pose.orientation.w,
                        pose_data.pose.orientation.x,
                        pose_data.pose.orientation.y,
                        pose_data.pose.orientation.z
                    ]
                
                # TODO: merge current_trajectory with trajectory after it's done
                if trajectory_data is not None:
                    # Convert Path to [[x1,y1], [x2,y2], ..., [xn,yn]] list format
                    trajectory = [
                        [pose.pose.position.x, pose.pose.position.y] 
                        for pose in trajectory_data.poses
                    ]
                    if last_goal_position != trajectory[-1]:
                        last_goal_position = trajectory[-1]
                        continue_mpc = True
                    
                    trajectory = np.array(trajectory, dtype=np.float32)
                # if the robot is close to the last trajectory point, stop following
                last_point = trajectory[-1] if trajectory is not None else current_pose[:2]
                dist_to_last = np.linalg.norm(np.array(current_pose[:2]) - np.array(last_point[:2]))     
                if dist_to_last < 2.0:  # distance threshold can be adjusted
                    trajectory = None
                    continue_mpc = False
                # Increment image counter for statistics
                image_counter += 1
                
                # Print publishing statistics every 100 frames
                if image_counter % 100 == 0:
                    info_publisher.get_logger().info(f'📸 Published {image_counter} image pairs + IMU data via ROS2')
                        

            

            # Apply motion commands to observation space (use actual command based on policy state)
            obs["policy"][:, 9:13] = actual_cmd_tensor

    # Clean up UDP listener
    if udp_listener:
        udp_listener.stop()
        print("🛑 UDP listener stopped")
    
    # Clean up ROS2
    info_publisher.destroy_node()
    rclpy.shutdown()
    print("🛑 ROS2 publisher stopped")
    
    print(f"📸 Session completed! Total published: {image_counter} image pairs + IMU data via ROS2")
    print(f"    RGB topic: /camera/rgb/image_raw")
    print(f"    Depth topic: /camera/depth/image_raw")
    print(f"    IMU topic: /imu/data")
    print(f"    Root quaternion topic: /robot/root_quaternion")
    print(f"📍 Robot pose subscription: /robot/pose")
    print(f"🛤️ Trajectory subscription: /robot/trajectory")


if __name__ == "__main__":
    main()
    simulation_app.close()
