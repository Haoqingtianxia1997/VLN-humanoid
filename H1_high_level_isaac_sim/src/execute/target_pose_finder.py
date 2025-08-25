import numpy as np
import logging

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Quaternion, Point
from sensor_msgs.msg import CameraInfo

logger = logging.getLogger(__name__)


def get_relative_target_pose_from_image(
        pixel_position: tuple[int, int],
        depth: np.ndarray, 
        cam_info: CameraInfo,
        window_size: int = 1,
        depth_scale: float | None = None
) -> Point | None:
    
    if depth_scale is None:
        depth_scale = 1.0

    x, y = pixel_position
    z = depth[y, x] * float(depth_scale) # direction forward
    K = cam_info.k
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]

    if fy == 0 or fx == 0:
        logger.warning("Camera intrinsic parameters fx or fy are zero, cannot compute relative position.")
        return None

    X = (x - cx) * z / fx # direction right
    Y = (y - cy) * z / fy # direction down
    return Point(x=z, y=-X, z=-Y)

def transform_relative_to_map(robot_pose: Pose, relative_point: Point) -> Pose:
    if robot_pose is None or relative_point is None:
        logger.warning("Robot Pose or Relative Point is None.")
        return None

    robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
    robot_quat = np.array([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
    relative_vec = np.array([relative_point.x, relative_point.y, relative_point.z])

    q_vec = robot_quat[:3]
    q_w = robot_quat[3]
    t = 2 * np.cross(q_vec, relative_vec)
    rotated_vec = relative_vec + q_w * t + np.cross(q_vec, t)

    absolute_pos = robot_pos + rotated_vec
    absolute_target_pose = Pose(
        position=Point(x=absolute_pos[0], y=absolute_pos[1], z=absolute_pos[2]),
        orientation=robot_pose.orientation
    )
    return absolute_target_pose

def add_margin_to_target(target_pose: Pose, margin:float = 0.5) -> Pose:
    if target_pose is None:
        logger.warning("Target Pose is None.")
        return None

    quat = np.array([target_pose.orientation.x, target_pose.orientation.y,
                     target_pose.orientation.z, target_pose.orientation.w])
    rot = R.from_quat(quat, scalar_first=True)
    margin_xyz = margin * np.diagonal(rot.as_matrix())
    target_margin_pose = Pose(
        position=Point(
            x=target_pose.position.x - margin_xyz[0],
            y=target_pose.position.y - margin_xyz[1],
            z=target_pose.position.z - margin_xyz[2]),
        orientation=target_pose.orientation
    )
    return target_margin_pose