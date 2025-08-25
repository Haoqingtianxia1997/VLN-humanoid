# # h1_env_isaac.py – H1IsaacEnv 环境封装（适配 Isaac Lab 0.40.x）
# """SimulationContext + InteractiveScene 的轻量封装"""

# import torch
# from isaaclab.sim import SimulationContext, SimulationCfg
# from isaaclab.scene import InteractiveScene

# from h1_cfg import H1SceneCfg

# __all__ = ["H1IsaacEnv"]


# class H1IsaacEnv:
#     """H1 在 Isaac Lab 中的最小可运行环境"""

#     def __init__(self, num_envs: int = 1, device: str = "cpu") -> None:
#         # 物理上下文
#         self.sim = SimulationContext(SimulationCfg(device=device))
#         self.sim.set_camera_view([4, 0, 2], [0, 0, 1])

#         # 场景
#         self.scene = InteractiveScene(H1SceneCfg(num_envs, env_spacing=3.0))
#         self.sim.reset()
#         self.scene.write_data_to_sim()

#         # 时间信息
#         self.dt = self.sim.get_physics_dt()
#         self.time = 0.0
#         self.step_count = 0

#     # ------------------------------------------------------------------ API --
#     def reset(self) -> None:
#         """复位环境"""
#         self.scene.reset()
#         self.time = 0.0
#         self.step_count = 0

#     def step(self, actions: torch.Tensor) -> None:
#         """执行一步物理仿真
#         Parameters
#         ----------
#         actions : torch.Tensor, shape (num_envs, num_dofs)
#             关节力矩
#         """
#         self.scene["H1"].set_joint_effort_target(actions)

#         self.scene.write_data_to_sim()
#         self.sim.step()
#         self.scene.update(self.dt)

#         self.time += self.dt
#         self.step_count += 1

#     # ---------------------------------------------------------------- clean --
#     def close(self) -> None:
#         """关闭 Omniverse 应用"""
#         if getattr(self.sim, "app", None) is not None:
#             self.sim.app.close()










# h1_env_isaac.py – 完整 H1Env（Isaac Lab 0.40.x 版）
"""Isaac 复现 Mujoco H1Env 逻辑：
- 500 Hz 物理步 (dt=0.002)
- 50 Hz 控制 (control_decimation=10)
- 41 维观测 / 10 维动作协议
- reset / step / render / close / set_cmd / get_cmd / get_joint_names
"""

import math
import time
from typing import Optional

import numpy as np
import torch
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene

from h1_cfg import H1SceneCfg

__all__ = ["H1Env"]

# ---------- 常量 ----------
SIM_DT = 0.002
CONTROL_DECIMATION = 10

DEFAULT_Q_M = np.array([
    0.0, 0.0, -0.1, 0.3, -0.2,
    0.0, 0.0, -0.1, 0.3, -0.2,
], dtype=np.float32)

# 原始（MuJoCo 顺）
KPS_M = np.array([150, 150, 150, 200, 40,
                  150, 150, 150, 200, 40], dtype=np.float32)
KDS_M = np.array([2, 2, 2, 4, 2,
                  2, 2, 2, 4, 2], dtype=np.float32)

# ------- MuJoCo→Isaac 索引映射 --------
IDX_M2I = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9], dtype=int)
IDX_I2M = np.argsort(IDX_M2I)


# 按 Isaac 顺序重排得到真正用的增益
KPS_I = KPS_M[IDX_M2I]
KDS_I = KDS_M[IDX_M2I] 

DEFAULT_Q_I = DEFAULT_Q_M[IDX_M2I] 
print("默认关节位置（Isaac 顺）:", DEFAULT_Q_I)

CMD_SCALE     = np.array([4.0, 4.0, 0.5], dtype=np.float32)
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25
ACTION_SCALE  = 0.25

OBS_DIM = 41
MAX_EPISODE_STEPS = 1000
MAX_EPISODE_TIME  = 30.0  # 秒


# ---------- 辅助 ----------
def _gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


# ---------- 环境 ----------
class H1Env:
    def __init__(self, device: str = "cpu") -> None:
        # Sim & Scene
        self.sim = SimulationContext(SimulationCfg(device=device))
        self.sim.set_camera_view([4, 0, 2], [0, 0, 1])

        self.scene = InteractiveScene(H1SceneCfg(1, env_spacing=3.0))
        self.sim.reset()
        self.scene.write_data_to_sim()

        self.h1 = self.scene["H1"]
        self.dt = SIM_DT

        # 状态缓存
        self.cmd      = np.array([0.08, 0.0, 0.0], dtype=np.float32)
        self.obs      = np.zeros(OBS_DIM, dtype=np.float32)
        self.action   = np.zeros(10, dtype=np.float32)
        self.target_q = DEFAULT_Q_I.copy()

        # 计数
        self.step_counter      = 0
        self.episode_steps     = 0
        self.episode_start_time: Optional[float] = None
        self._x0y0      = np.zeros(2)
        self._start_yaw = 0.0


    # ---------------- reset ----------------
    def reset(self):
        # ---------- 根姿态 ----------
        root_pose = torch.tensor([[0, 0, 1.2 , 1, 0, 0, 0]], device=self.sim.device)
        self.h1.write_root_pose_to_sim(root_pose)
        self.h1.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.sim.device))

        # ---------- 直接写关节状态 ----------
        with torch.no_grad():
            self.h1.data.joint_pos[0].copy_(torch.from_numpy(DEFAULT_Q_I).to(self.sim.device))
            self.h1.data.joint_vel[0].zero_()

        # ---------- 刷新 & 步进 ----------
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

        # ---------- 计数 ----------
        self.step_counter = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        return self._build_obs()

    # ---------------- step ----------------
    def step(self, action: np.ndarray, cmd: Optional[np.ndarray] = None):
        if cmd is not None:
            self.set_cmd(cmd)

        q_des_m = action.copy()     # MuJoCo 顺
        q_des_i = q_des_m[IDX_M2I]                                  # → Isaac 顺
        # self.target_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.target_q = q_des_i * ACTION_SCALE + DEFAULT_Q_I
        self.action   = action.copy()    

        for _ in range(CONTROL_DECIMATION):
            # root_pose = torch.tensor([[0, 0, 2.0, 1, 0, 0, 0]], device=self.sim.device)
            # self.h1.write_root_pose_to_sim(root_pose)
            # self.h1.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.sim.device))

            tau = self._compute_pd_torque(self.target_q)
            self.h1.set_joint_effort_target(torch.from_numpy(tau).unsqueeze(0))

            # self.h1.set_joint_position_target(torch.from_numpy(self.target_q).unsqueeze(0))
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)
            self.step_counter += 1

        obs = self._build_obs()
        reward = 0.0
        done = self._check_done()
        truncated = self._check_truncate()
        info = {}

        self.episode_steps += 1
        return obs, reward, done, truncated, info

    # ---------------- 其他公开 ----------------
    def render(self): pass  # Omniverse 窗口自动更新

    def _compute_pd_torque(self, target_q: np.ndarray):
        q  = self.h1.data.joint_pos[0].cpu().numpy()
        dq = self.h1.data.joint_vel[0].cpu().numpy()
        return KPS_I * (target_q - q) - KDS_I * dq
    
    def close(self):
        """安全关闭 Omniverse 应用（各版本兼容）"""
        if getattr(self.sim, "app", None):
            if hasattr(self.sim.app, "close"):
                self.sim.app.close()
            elif hasattr(self.sim.app, "quit"):
                self.sim.app.quit()
            # 若两者都没有，说明当前版本会在脚本结束时自动析构，无需手动处理


    def set_cmd(self, new_cmd: np.ndarray): self.cmd = new_cmd.astype(np.float32)
    def get_cmd(self): return self.cmd.copy()
    def get_joint_names(self): return self.h1.data.joint_names.copy()



    def _build_obs(self):
        root_ang_vel = self.h1.data.root_ang_vel_b[0].cpu().numpy()
        print("根角速度（b）:", root_ang_vel)
        root_quat_wxyz    = self.h1.data.root_quat_w[0].cpu().numpy()

        print("根四元数（wxyz）:", root_quat_wxyz)
        root_quat_xyzw = np.array([root_quat_wxyz[1],root_quat_wxyz[2],root_quat_wxyz[3],root_quat_wxyz[0]]) 
        print("根四元数（xyzw）:", root_quat_xyzw)
        joint_pos_i = self.h1.data.joint_pos[0].cpu().numpy()         # Isaac 顺
        joint_vel_i = self.h1.data.joint_vel[0].cpu().numpy()
        joint_pos_m = joint_pos_i[IDX_I2M]
        joint_vel_m = joint_vel_i[IDX_I2M]
        # print("关节位置（Isaac 顺）:", joint_pos_i)
        # print("关节位置（MuJoCo 顺）:", joint_pos_m)
        print("关节速度（MuJoCo 顺）:", joint_vel_m)
        


        obs = self.obs
        obs[0:3]   = root_ang_vel * ANG_VEL_SCALE
        obs[3:6] = _gravity_orientation(root_quat_xyzw)
        obs[6:9]   = self.cmd * CMD_SCALE
        obs[9:19]  = (joint_pos_m - DEFAULT_Q_M) * DOF_POS_SCALE
        obs[19:29] = joint_vel_m * DOF_VEL_SCALE
        obs[29:39] = self.action
        phase = (self.step_counter * self.dt) % 0.8 / 0.8
        obs[39] = math.sin(2 * math.pi * phase)
        obs[40] = math.cos(2 * math.pi * phase)
        return obs.copy()

    def _check_done(self):
        return bool(self.h1.data.root_pos_w[0, 2].item() < 0.5)

    def _check_truncate(self):
        if self.episode_start_time is None:
            return False
        elapsed = time.time() - self.episode_start_time
        return self.episode_steps >= MAX_EPISODE_STEPS or elapsed >= MAX_EPISODE_TIME















