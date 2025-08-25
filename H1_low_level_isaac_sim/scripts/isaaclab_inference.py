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

import omni

import cv2
import numpy as np
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from isaaclab_assets import H1_CFG 

from isaacsim.core.utils.extensions import enable_extension
import omni.graph.core as og 


def _set_action_graph(env: ManagerBasedRLEnv):
    robot_asset = env.scene.articulations["robot"]
    robot_prim_path = str(robot_asset.cfg.prim_path).replace(".*", str(0))
    print(f"Roboter-Pfad für Action Graph gefunden: {robot_prim_path} with {robot_asset.num_joints} joints.")

    keys = og.Controller.Keys
    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": f"{robot_prim_path}/ActionGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                # Basics
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("RunSimFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                # ("RgbRenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                # ("DepthRenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

                # Publisher
                # ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                # ("RGBCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                # ("DepthCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                # Basics
                # ("ReadOdom.inputs:chassisPrim", robot_prim_path),
                # ("RgbRenderProd.inputs:cameraPrim", rgb_camera_path),
                # ("DepthRenderProd.inputs:cameraPrim", depth_camera_path),

                # Publisher
                # ("RGBCameraHelper.inputs:topicName", "rgb/image_raw"),
                # ("RGBCameraHelper.inputs:frameId", "d435_rgb_module_link"),
                # ("RGBCameraHelper.inputs:type", "rgb"),
                # ("DepthCameraHelper.inputs:topicName", "depth/image_raw"),
                # ("DepthCameraHelper.inputs:frameId", "d435_left_imager_link"),
                # ("DepthCameraHelper.inputs:type", "depth"),
            ],
            keys.CONNECT: [
                # Basics
                ("OnPlaybackTick.outputs:tick", "RunSimFrame.inputs:execIn"),
                # ("RunSimFrame.outputs:step", "RgbRenderProd.inputs:execIn"),
                # ("RunSimFrame.outputs:step", "DepthRenderProd.inputs:execIn"),

                # Publisher
                # ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                # ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                # ("RgbRenderProd.outputs:execOut", "RGBCameraHelper.inputs:execIn"),
                # ("RgbRenderProd.outputs:renderProductPath", "RGBCameraHelper.inputs:renderProductPath"),
                # ("DepthRenderProd.outputs:execOut", "DepthCameraHelper.inputs:execIn"),
                # ("DepthRenderProd.outputs:renderProductPath", "DepthCameraHelper.inputs:renderProductPath"),
            ],
        },
    ) 


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
    )
    
    env_cfg.scene.camera = CameraCfg(
        prim_path="/World/envs/env_0/Robot/torso_link/front_cam",
        update_period=0.02,
        width=640, height=480,
        data_types=["rgb", "depth"],   # "depth" or "distance_to_image_plane" both work
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, # aperture from tutorial https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/cameras/create-perspective-camera.html
            clipping_range=(0.1, 100), # depth range limit
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.7),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
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

    for _ in range(50): # 50 Schritte sind ein sicherer Puffer
        env.step(torch.zeros_like(env.action_manager.action))

    _set_action_graph(env)
    
    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)

            cam = env.scene["camera"]
            rgb: torch.Tensor = cam.data.output["rgb"]
            depth: torch.Tensor = cam.data.output["depth"]

            rgb0   = rgb[0].cpu().numpy()   if rgb.dim()   == 4 else rgb.cpu().numpy()
            depth0 = depth[0].cpu().numpy() if depth.dim() == 4 else depth.cpu().numpy()
            
            # Create camera_feed directory if it doesn't exist
            os.makedirs("camera_feed", exist_ok=True)
            
            # Save RGB image
            rgb_image = rgb0[..., :3]
            cv2.imwrite("camera_feed/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # Save depth data
            np.save("camera_feed/depth.npy", depth0.squeeze())
            
            # save depth as a grayscale image
            depth_vis = depth0.squeeze()
            # filter out invalid depth values
            depth_valid = depth_vis[(depth_vis > 0) & np.isfinite(depth_vis)]
            if depth_valid.size > 0:
                dmin, dmax = depth_valid.min(), depth_valid.max()
            else:
                dmin, dmax = 0, 1

            # from 0-1 to 0-255
            depth_norm = np.clip((depth_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # cv2.imwrite("camera_feed/depth.png", depth_uint8)

            # alternatively save depth as a color map, uncomment the following lines and comment the above cv2.imwrites
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            cv2.imwrite("camera_feed/depth.png", depth_color)
            
            # overwrite with given commands
            obs["policy"][:, 9:13] = torch.tensor([1.0, 0.0, 0.0, 0.2], device=args_cli.device)


if __name__ == "__main__":
    main()
    simulation_app.close()
