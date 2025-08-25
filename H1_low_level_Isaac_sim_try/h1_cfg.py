# h1_cfg.py – H1 机器人与场景配置（含每关节力矩上限）
"""根据 MuJoCo XML 中的 `actuatorfrcrange`，逐关节设置 effort_limit_sim。"""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg  # 物理材质
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



__all__ = ["H1_CFG", "H1SceneCfg"]

# -----------------------------------------------
# 力矩上限（±）——同之前
EFFORT_LIMITS = {
    # 左腿
    "left_hip_yaw_joint":   200.0,
    "left_hip_roll_joint":  200.0,
    "left_hip_pitch_joint": 200.0,
    "left_knee_joint":      300.0,
    "left_ankle_joint":      40.0,
    # 右腿
    "right_hip_yaw_joint":   200.0,
    "right_hip_roll_joint":  200.0,
    "right_hip_pitch_joint": 200.0,
    "right_knee_joint":      300.0,
    "right_ankle_joint":      40.0,
}

# -----------------------------------------------
# 关节对应的 Kp / Kd（已经按 Isaac 顺序填好）
KP_KD = {
    "left_hip_yaw_joint":   (150.0, 2.0),
    "right_hip_yaw_joint":  (150.0, 2.0),
    "left_hip_roll_joint":  (150.0, 2.0),
    "right_hip_roll_joint": (150.0, 2.0),
    "left_hip_pitch_joint": (150.0, 2.0),
    "right_hip_pitch_joint":(150.0, 2.0),
    "left_knee_joint":      (200.0, 4.0),
    "right_knee_joint":     (200.0, 4.0),
    "left_ankle_joint":     ( 40.0, 2.0),
    "right_ankle_joint":    ( 40.0, 2.0),
}

# -----------------------------------------------
# 为每个关节生成 ImplicitActuatorCfg
_actuators_cfg = {
    name: ImplicitActuatorCfg(
        joint_names_expr=[name],          # 精确匹配该关节
        effort_limit_sim=EFFORT_LIMITS[name],
        velocity_limit_sim=100.0,
        stiffness=0.0 ,#KP_KD[name][0],         # Kp → stiffness
        damping=5,#KP_KD[name][1],           # Kd → damping
    )
    for name in EFFORT_LIMITS.keys()
}


H1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/H1",
    # spawn=sim_utils.UsdFileCfg(usd_path="../../h1_models/h1.usd"),
    spawn=sim_utils.UsdFileCfg(usd_path="h1_models/h1.usd"),
    actuators=_actuators_cfg,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)



# import os
# os.chdir("./usda_env/2n8kARJN3HM/")
class H1SceneCfg(InteractiveSceneCfg):
    """地面 + 光照 + H1（高摩擦系数 1.5）"""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=3,      # 起步/刹车更抓地
                dynamic_friction=3,     # 滑动摩擦
                restitution=0.0,          # 不弹跳
            ),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=5000),
    )

    H1 = H1_CFG

    office = AssetBaseCfg(
    prim_path="/World/Office",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
    ),
)

    # myroom_mesh = AssetBaseCfg(
    #     prim_path="/World/MyRoomMesh",   # 你希望在USD里的路径（一般放/World下）
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="2n8kARJN3HM.usda"
    #     ),
    # )