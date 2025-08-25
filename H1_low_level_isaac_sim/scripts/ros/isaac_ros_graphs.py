import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension





def setup_ros2_bridge_for_rgbd(env, topic_prefix="camera", frame_depth="camera_depth"):
    """Create Action Graph nodes to publish stereo RGB images (+ camera_info) via ROS2 bridge."""
    enable_extension("isaacsim.core.nodes")
    enable_extension("isaacsim.sensors.physics")
    enable_extension("isaacsim.ros2.bridge")

    robot_asset = env.scene.articulations["robot"]
    robot_link_prim = str(robot_asset.cfg.prim_path).replace(".*", str(0))

    torso_link_prim = f"{robot_link_prim}/torso_link"
    front_cam_prim = f"{torso_link_prim}/front_cam"

    keys = og.Controller.Keys
    graph_path = f"/World/ROS2Graph"

    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacClock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),

                # Renderer
                ("DepthRenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

                # Clock
                ("ClockPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),

                # Camera Publisher
                ("CameraInfoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("DepthCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                # ROS2 context settings
                ("ROS2Context.inputs:domain_id", 0),

                # Renderer
                ("DepthRenderProd.inputs:cameraPrim", front_cam_prim),

                # Camera Info
                ("CameraInfoHelper.inputs:topicName", f"{topic_prefix}/info"),
                ("CameraInfoHelper.inputs:frameId", frame_depth),
            ],
            keys.CONNECT: [
                # Renderer
                ("OnPlaybackTick.outputs:tick", "DepthRenderProd.inputs:execIn"),
                
                # Clock Publisher
                ("ROS2Context.outputs:context", "ClockPublisher.inputs:context"),
                ("OnPlaybackTick.outputs:tick", "ClockPublisher.inputs:execIn"),
                ("IsaacClock.outputs:simulationTime", "ClockPublisher.inputs:timeStamp"),

                # Camera Info Publisher
                ("ROS2Context.outputs:context", "CameraInfoHelper.inputs:context"),
                ("OnPlaybackTick.outputs:tick", "CameraInfoHelper.inputs:execIn"),
                ("DepthRenderProd.outputs:renderProductPath", "CameraInfoHelper.inputs:renderProductPath"),
            ],
        },
    )