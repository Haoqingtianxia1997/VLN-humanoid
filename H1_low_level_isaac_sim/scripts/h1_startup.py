import os
import numpy as np
import time
import torch
import io

# isaac_lib_path = os.path.join(os.environ["VIRTUAL_ENV"], "Lib", "site-packages", "isaacsim", "exts", "isaacsim.ros2.bridge", "humble", "lib")
# os.environ["PATH"] += os.pathsep + isaac_lib_path
# os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# os.environ["ROS_DOMAIN_ID"] = "0"

CONFIG = {
        "headless": False,
        "hide_ui": None,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": True,
        "max_gpu_count": None,
        "sync_loads": True,
        "width": 1280,
        "height": 720,
        "window_width": 1440,
        "window_height": 900,
        "display_options": 3094,
        "subdiv_refinement_level": 0,
        "renderer": "RaytracedLighting",  # Can also be PathTracing
        "anti_aliasing": 3,
        "samples_per_pixel_per_frame": 64,
        "denoiser": True,
        "max_bounces": 4,
        "max_specular_transmission_bounces": 6,
        "max_volume_bounces": 4,
        "open_usd": None,
        "fast_shutdown": True,
        "profiler_backend": [],
        "create_new_stage": True,
        "extra_args": [
            "--/app/livestream/webrtc/enabled=true"
        ]
}

from isaacsim import SimulationApp
simulation_app = SimulationApp(CONFIG)

# Logging 
import carb
# import carb.settings
# carb.settings.get_settings().set("/app/enableDeveloperWarnings", False)
# carb.settings.get_settings().set("/app/scripting/ignoreWarningDialog", True)

# carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/verbose", False)
# carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/info", False)
# carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/warning", False)
# carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/error", False)
# carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/fatal", False)

# carb.settings.get_settings().set("/log/debugConsoleLevel", "Fatal")  # verbose"|"info"|"warning"|"error"|"fatal"
# carb.settings.get_settings().set("/log/enabled", False)
# carb.settings.get_settings().set("/log/outputStreamLevel", "Error")
# carb.settings.get_settings().set("/log/fileLogLevel", "Error")

# Isaaacsim imports
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api.world.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import omni.graph.core as og

enable_extension("isaacsim.ros2.bridge")
# enable_extension("isaacsim.ros2.tf_viewer")
# enable_extension("isaacsim.ros2.urdf")
enable_extension("isaacsim.core.nodes")

if __name__ == "__main__":
    policy_path = os.path.abspath("policy/policy.pt")
    policy = torch.jit.load(policy_path, map_location="cuda")
    print(policy)

    try:
        if World.instance():
            World.instance().clear_instance()

        world = World()
        world.scene.add_default_ground_plane(z_position=-1.0)
        robot_prim_path = "/World/H1"
        asset_path = get_assets_root_path() + "/Isaac/Robots/Unitree/H1/h1.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
        h1_robot = SingleArticulation(robot_prim_path, name="my_h1_robot")
        world.scene.add(h1_robot)

        # RGB Camera
        rgb_camera_path = f"{robot_prim_path}/d435_rgb_module_link/rgb_camera"
        rgb_camera = Camera(
            prim_path=rgb_camera_path, 
            name="rgb_camera_sensor",
            resolution=(1280, 720)
        )

        # Depth Camera
        depth_camera_path = f"{robot_prim_path}/d435_left_imager_link/depth_camera"
        depth_camera = Camera(
            prim_path=depth_camera_path, 
            name="depth_camera_sensor",
            resolution=(640, 480)
        )

        world.reset()
        rgb_camera.initialize()
        rgb_camera.add_motion_vectors_to_frame()
        depth_camera.initialize()
        depth_camera.add_motion_vectors_to_frame()


        joint_names = h1_robot.dof_names
        carb.log_info(f"Number of joints: {len(joint_names)} in {asset_path}")
        carb.log_info(f"Robot joints: {joint_names}")
        
        if len(joint_names) == 0:
            carb.log_error("No joints found in robot!")
            exit(1)

        

        keys = og.Controller.Keys
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": f"{robot_prim_path}/ActionGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    # Basics
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("RunSimFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("RgbRenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("DepthRenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),

                    # Publisher
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("RGBCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("DepthCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                    # Subscriber
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ],
                keys.SET_VALUES: [
                    # Basics
                    ("ReadOdom.inputs:chassisPrim", robot_prim_path),
                    ("RgbRenderProd.inputs:cameraPrim", rgb_camera_path),
                    ("DepthRenderProd.inputs:cameraPrim", depth_camera_path),

                    # Publisher
                    ("PublishTF.inputs:targetPrims", [robot_prim_path]),
                    ("PublishTF.inputs:topicName", "tf"),
                    ("PublishJointState.inputs:targetPrim", robot_prim_path),
                    ("PublishJointState.inputs:topicName", "joint_states"),
                    ("PublishOdom.inputs:topicName", "odom"),
                    ("RGBCameraHelper.inputs:topicName", "rgb/image_raw"),
                    ("RGBCameraHelper.inputs:frameId", "d435_rgb_module_link"),
                    ("RGBCameraHelper.inputs:type", "rgb"),
                    ("DepthCameraHelper.inputs:topicName", "depth/image_raw"),
                    ("DepthCameraHelper.inputs:frameId", "d435_left_imager_link"),
                    ("DepthCameraHelper.inputs:type", "depth"),

                    # Subscriber
                    ("ArticulationController.inputs:targetPrim", robot_prim_path),
                    ("SubscribeJointState.inputs:topicName", "joint_command"),
                ],
                keys.CONNECT: [
                    # Basics
                    ("OnPlaybackTick.outputs:tick", "ReadOdom.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "RunSimFrame.inputs:execIn"),
                    ("RunSimFrame.outputs:step", "RgbRenderProd.inputs:execIn"),
                    ("RunSimFrame.outputs:step", "DepthRenderProd.inputs:execIn"),

                    # Publisher
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                    ("ReadOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                    ("ReadOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                    ("ReadOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                    ("ReadOdom.outputs:position", "PublishOdom.inputs:position"),
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("RgbRenderProd.outputs:execOut", "RGBCameraHelper.inputs:execIn"),
                    ("RgbRenderProd.outputs:renderProductPath", "RGBCameraHelper.inputs:renderProductPath"),
                    ("DepthRenderProd.outputs:execOut", "DepthCameraHelper.inputs:execIn"),
                    ("DepthRenderProd.outputs:renderProductPath", "DepthCameraHelper.inputs:renderProductPath"),

                    # Subscriber
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
            },
        )
        
        carb.log_info(f"Action graph created successfully with nodes: {nodes}")

        world.play()

        carb.log_warn("Simulation running...")
        while simulation_app.is_running():
            world.step(render=True)

    except Exception as e:
        carb.log_error(f"Ein Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
