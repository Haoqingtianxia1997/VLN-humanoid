#!/usr/bin/env python
"""
Unitree-H1 平地策略 (h1_flat.pt) —— 速度指令行走示例
支持 Isaac-Lab 0.4.x / 0.5.x，CPU 或 GPU 均可。
"""

import argparse, time, torch, sys
from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="h1_flat.pt",
                    help="TorchScript 权重路径")
parser.add_argument("--vx", type=float, default=0.3, help="前向速度 m/s")
parser.add_argument("--vy", type=float, default=0.0, help="侧向速度 m/s")
parser.add_argument("--wz", type=float, default=0.0, help="角速度 rad/s")
AppLauncher.add_app_launcher_args(parser)       # 自动加 --device
cli = parser.parse_args()

# ---------------- 启动 Omniverse ----------------
simulation_app = AppLauncher(cli).app           # ☆ 必须最先

# ---------------- 动态 import 环境 cfg ----------
try:
    # Isaac-Lab ≥0.5
    from isaaclab.envs.unitree.h1.flat_env_cfg import UnitreeH1FlatEnvCfg
except ModuleNotFoundError:
    # Isaac-Lab 0.4.x
    from isaaclab.envs.unitree_h1.flat_env_cfg import UnitreeH1FlatEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import clamp_tensor as _dummy  # 仅检测旧函数是否存在
del _dummy                                              # 不再使用

# ---------------- 创建环境 ----------------
cfg = UnitreeH1FlatEnvCfg()
cfg.scene.num_envs = 1

# 可选：提高地面摩擦
if hasattr(cfg.scene.ground.spawn, "physics_material"):
    mat = cfg.scene.ground.spawn.physics_material
elif hasattr(cfg.scene.ground.spawn, "physx_material"):
    mat = cfg.scene.ground.spawn.physx_material
else:
    mat = None
if mat:
    mat.static_friction = 1.5
    mat.dynamic_friction = 1.5
    mat.restitution = 0.0

env = ManagerBasedRLEnv(cfg=cfg, render_mode="human", device=cli.device)

# ---------------- 加载策略 ----------------
policy = torch.jit.load(cli.policy, map_location=cli.device)
obs = env.reset()

# 固定速度指令 (vx, vy, wz)
cmd = torch.tensor([[cli.vx, cli.vy, cli.wz]], device=cli.device)

# ---------------- 主循环 ----------------
while simulation_app.is_running():
    obs[:, 6:9] = cmd                     # 官方策略使用 obs[6:9] 读取指令
    with torch.no_grad():
        act = policy(obs)                 # 输出 (1,10)
    act = torch.clamp(act, -1.0, 1.0)     # 限幅

    obs, _, done, _ = env.step(act)
    if done.any():
        obs = env.reset()

    time.sleep(env.cfg.sim.dt)            # 500 Hz ≈ 0.002 s

env.close()
