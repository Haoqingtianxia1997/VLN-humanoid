import time
import rclpy
import socket
import time
import threading
import logging
import numpy as np
import cv2

from dataclasses import dataclass, field
from enum import Enum 
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo

try:    
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.as_posix())
    from src.VLM_agent.agent import VLM_agent
    from src.logger import setup_logging

    from src.execute.ros_communication import RosVlmNode
    from src.execute.target_pose_finder import (
        get_relative_target_pose_from_image, 
        transform_relative_to_map,
        add_margin_to_target
    )
    from src.VLM_agent.agent import VLM_agent
    logger = logging.getLogger(__name__)
except ImportError:
    from ros_communication import RosVlmNode
    from target_pose_finder import (
        get_relative_target_pose_from_image, 
        transform_relative_to_map,
        add_margin_to_target
    )



class Direction(Enum):
    LEFT       = "left"
    RIGHT      = "right"
    FORWARD    = "forward"
    BACKWARD   = "backward"


@dataclass
class PerceiveAction:
    target: str    = "Default Target"
    margin: float  = 2.0


@dataclass
class StopAction:
    duration: float = 1.0


@dataclass
class MoveAction:
    speed: float                          = 0.5
    distance: float                       = 0.0
    direction: Direction                  = Direction.FORWARD
    execution_time: float                 = field(init=False, repr=False)
    udp_cmd: tuple[float, float, float]   = field(init=False, repr=False)

    def __post_init__(self):
        self.speed = abs(self.speed)
        self.execution_time = 4 * self.distance / self.speed
        match self.direction:
            case Direction.FORWARD:
                self.udp_cmd = (self.speed, 0.0, 0.0)
            case Direction.BACKWARD:
                self.udp_cmd = (-self.speed, 0.0, 0.0)
            case Direction.LEFT:
                self.udp_cmd = (0.0, self.speed, 0.0)
            case Direction.RIGHT:
                self.udp_cmd = (0.0, -self.speed, 0.0)
            case _:
                self.udp_cmd = (0.0, 0.0, 0.0)


@dataclass
class TurnAction:
    speed: float                          = 0.5
    angle: float                          = 0.0
    direction: Direction                  = Direction.LEFT
    execution_time: float                 = field(init=False, repr=False)
    udp_cmd: tuple[float, float, float]   = field(init=False, repr=False)

    def __post_init__(self):
        self.angle = abs(self.angle)
        angle_radians = self.angle / 180 * 3.14
        self.execution_time = 1.5 * angle_radians / self.speed

        if self.direction == Direction.LEFT:
            self.udp_cmd = (0.0, 0.0, self.speed)
        elif self.direction == Direction.RIGHT:
            self.udp_cmd = (0.0, 0.0, -self.speed)
        else:
            self.udp_cmd = (0.0, 0.0, 0.0)


class ActionExecutor:
    def __init__(
            self, 
            address: str = "127.0.0.1", 
            port: int = 12345, 
            image_path: str = "images/rgb.png"
    ):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (address, port)
        self.image_path = image_path
        self.max_retries = 6
        self.logger = logging.getLogger('vln_humanoids')
        self.cmd = (0.0, 0.0, 0.0)
        rclpy.init()
        self.node = RosVlmNode()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin_thread.start()

    def execute_sequence(self, actions: list[TurnAction | MoveAction | PerceiveAction | StopAction]) -> bool:
        success = False
        for action in actions:
            self.logger.info(f"Executing -> {action}")
            match action:
                case PerceiveAction():
                    success = self.execute_preceive(action)
                case MoveAction() | TurnAction():
                    success = self.execute_move(action)
                case StopAction():
                    success = self.execute_stop(action)
                case _:
                    success = False

            if not success:
                self.logger.warning(f"Aborting action sequence due to failure at action {action}.")
                return False
        
            time.sleep(0.5)
        self.logger.info("Action sequence completed.")
        return True

    def execute_preceive(self, action: PerceiveAction):
        for _ in range(self.max_retries):
            rgb, depth, cam_info, robot_pose = self._wait_for_data()
            self._save_img(rgb)
            self._safe_depth(depth)
            success, image_target_point = VLM_agent(action.target, image_path=self.image_path)

            if success: 
                self._calc_and_publish_target_pose(image_target_point, depth, cam_info, robot_pose, action.margin)
                self.execute_stop(StopAction(duration=1.0))
                while self.node.get_path_execution_running():
                    time.sleep(1)
                return True

            if not self.execute_move(TurnAction(angle=60)):
                return False

        return False

    def execute_move(self, action: MoveAction | TurnAction) -> bool:
        self.send_udp_cmd(*action.udp_cmd)

        if self.cmd == (0.0, 0.0, 0.0):
            time.sleep(action.execution_time + 8.0)
        else:
            time.sleep(action.execution_time)
        self.cmd = action.udp_cmd
        self.send_udp_cmd(0.0, 0.0, 0.0)
        time.sleep(1)
        return True

    def execute_stop(self, action: StopAction) -> bool:
        self.send_udp_cmd(0.0, 0.0, 0.0)
        time.sleep(action.duration)
        return True

    def send_udp_cmd(self, vx, vy, wz):
        try:
            message = f"{vx} {vy} {wz}"
            data = message.encode('utf-8')
            self.sock.sendto(data, self.addr)
        except Exception as e:
            self.logger.error(f"Error sending UDP command: {e}")

    def shutdown(self):
        """Clean shutdown of ros spinners"""
        try:
            rclpy.shutdown()
        except Exception as e:
            self.logger.warning(f"Error during rclpy.shutdown(): {e}")
        try:
            self._spin_thread.join(timeout=1.0)
        except Exception:
            pass

    def _wait_for_data(self) -> tuple[np.ndarray, np.ndarray, CameraInfo, Pose]:
        while True:
            rgb, depth, cam_info, robot_pose = self.node.get_current_data()
            none_vals = [key for key, val in zip(
                ["rgb", "depth", "cam_info", "robot_pose"], [rgb, depth, cam_info, robot_pose]) if val is None]
            if none_vals:
                self.logger.info(f"Not all data available. Nones are {none_vals}. Waiting...")
                time.sleep(1)
            else:
                return rgb, depth, cam_info, robot_pose

    def _calc_and_publish_target_pose(self, image_target_point, depth, cam_info, robot_pose, margin) -> bool:
        rel_target_position = get_relative_target_pose_from_image(image_target_point, depth, cam_info, window_size=1)
        abs_target_pose = transform_relative_to_map(robot_pose, rel_target_position)
        margin_target_pose = add_margin_to_target(abs_target_pose, margin=margin)
        self.logger.debug(f"Robot pose: {robot_pose}")
        self.logger.debug(f"Relative target pose: {rel_target_position}")
        self.logger.debug(f"Absolute target pose: {abs_target_pose}")
        self.logger.debug(f"Margin target pose: {margin_target_pose}")
        self.node.publish_target_pose(margin_target_pose)
        return True

    def _save_img(self, rgb):
        if rgb is None:
            return
        img = rgb
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            img = img[..., :3]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.image_path, bgr)

    def _safe_depth(self, depth):
        depth_base = self.image_path.replace("rgb", "depth").replace(".png", "")
        if depth is None:
            return
        np.save(depth_base + ".npy", depth)
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth_mm = (depth_clean * 1000.0).astype(np.uint16)  # assumes depth in meters
        cv2.imwrite(depth_base + ".png", depth_mm)           # 16-bit png
        # visualization
        v = depth_mm.copy()
        v[v == 0] = 65535
        vmin = v.min() if v.size else 0
        vmax = v.max() if v.size else 1
        if vmin == vmax: vmax = vmin + 1
        vis = ((v.astype(np.float32) - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        vis_col = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imwrite(depth_base + "_vis.png", vis_col)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


def _parse_action(action_dict: dict) -> PerceiveAction | MoveAction | TurnAction | StopAction | None:
    """Parse a dictionary to an ActionClass object."""
    action_type = action_dict.get("type")
    if not action_type:
        logger.error("'type' missing in action dictionary.")
        return None

    params = action_dict.get("parameters", {})

    direction_map = {
        "forward": Direction.FORWARD,
        "backward": Direction.BACKWARD,
        "left": Direction.LEFT,
        "right": Direction.RIGHT,
    }
    parts = action_type.split('_', 1)
    main_action = parts[0]
    
    if main_action == "perceive":
        margin = params.get("margin", 0.0)
        return PerceiveAction(target=action_dict.get("target", "Default Target"), margin=margin)

    if main_action == "stop":
        return StopAction(duration=params.get("duration", 1.0))

    if main_action == "move":
        direction_str = parts[1] if len(parts) > 1 else None
        direction = direction_map.get(direction_str)
        if direction is None:
            print(f"Unknown direction for 'move': {direction_str}")
            return None
        
        return MoveAction(
            speed=params.get("speed", 1.0),
            distance=params.get("distance", 0.0),
            direction=direction
        )

    if main_action == "turn":
        direction_str = params.get("direction")
        direction = direction_map.get(direction_str)
        if direction not in [Direction.LEFT, Direction.RIGHT]:
            print(f"Unknown direction for 'turn': {direction_str}")
            return None
            
        return TurnAction(
            speed=params.get("speed", 0.3),
            angle=params.get("angle", 0.0),
            direction=direction
        )

    print(f"Unknown action type: {action_type}")
    return None


def parse_actions(actions_dict_list: list[dict]) -> list[PerceiveAction | MoveAction | TurnAction | StopAction]:
    return [_parse_action(action) for action in actions_dict_list]


if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging(level=logging.DEBUG, package_name='vln_humanoids')
    logger = logging.getLogger('vln_humanoids')

    actions_dict = [
        { "type": "perceive", "target": "plant", "parameters": { "margin": 0.0 } },
        { "type": "turn", "parameters": { "angle": 90, "direction": "left" } },
        { "type": "move_backward", "target": "", "parameters": { "distance": 1 } },
        { "type": "move_right", "parameters": { "distance": 0.5 } },
        { "type": "stop", "target": "TV", "parameters": {} },
    ]

    actions = parse_actions(actions_dict)
    with ActionExecutor(image_path="images/rgb.png", port=12345) as action_exec:
        action_exec.execute_sequence(actions)