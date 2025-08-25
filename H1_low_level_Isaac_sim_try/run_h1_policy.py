# # run_h1_policy.py – 运行 H1 PD 控制策略（适配 Isaac Lab 0.40.x）
# """CLI: 启动 Omniverse → 创建环境 → 简易 PD 控制"""

# import argparse
# import math

# import torch
# from isaaclab.app import AppLauncher


# def main() -> None:
#     # ---------------- CLI & App 启动 ----------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_envs", type=int, default=1)
#     AppLauncher.add_app_launcher_args(parser)
#     cli_args = parser.parse_args()

#     # 必须先启动 AppLauncher，确保 omni.* 依赖注入
#     simulation_app = AppLauncher(cli_args).app

#     # 之后才导入依赖 isaaclab.sim 的模块
#     from h1_env_isaac import H1IsaacEnv  # noqa: E402

#     # ---------------- 创建环境 ----------------------
#     env = H1IsaacEnv(num_envs=cli_args.num_envs, device=cli_args.device)

#     h1 = env.scene["H1"]
#     dof_names = h1.data.joint_names
#     num_dofs = len(dof_names)

#     # PD 增益
#     kp = torch.ones(num_dofs, device=env.sim.device) * 200.0
#     kd = torch.ones(num_dofs, device=env.sim.device) * 10.0
#     q0 = h1.data.default_joint_pos.clone()

#     print("机器人关节名列表:\n", dof_names)

#     # ---------------- 主循环 -----------------------
#     while simulation_app.is_running():
#         # 每 3 秒复位
#         if env.step_count % int(3.0 / env.dt) == 0:
#             env.scene.reset()
#             print("[INFO] reset H1")

#         # ① 期望角
#         q_des_scalar = 0.25 * math.sin(2 * math.pi * 0.5 * env.time)
#         q_des = q0 + q_des_scalar

#         # ② 当前状态
#         q = h1.data.joint_pos[0]
#         dq = h1.data.joint_vel[0]

#         # ③ 力矩
#         tau = kp * (q_des - q) - kd * dq

#         # ④ 步进
#         env.step(tau.unsqueeze(0))

#     # ---------------- 结束 -------------------------
#     env.close()


# if __name__ == "__main__":
#     main()











# run_h1_policy.py – 极简 Isaac 运行脚本
"""启动 Omniverse → 创建 H1Env → 策略推理 → env.step()"""

import argparse
import time

import numpy as np
import torch
from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
# parser.add_argument("--policy", default="../../motion.pt")
parser.add_argument("--policy", default="motion.pt")
AppLauncher.add_app_launcher_args(parser)          # 添加 --device
args = parser.parse_args()

# ---------------- 启动 Omniverse ----------------
simulation_app = AppLauncher(args).app             # ☆ 放在任何 isaaclab.sim import 之前

# ---------------- 导入 env ----------------
from h1_env_isaac import H1Env                     # noqa: E402

# ---------------- 创建环境 & 网络 ----------------
env = H1Env(device=args.device)
print("关节名列表:")
for i, n in enumerate(env.get_joint_names()):
    print(f"{i}: {n}")

policy = torch.jit.load(args.policy)
env.set_cmd(np.array([0.04, 0.0, 0.0], dtype=np.float32))

# UDP 指令（若模块缺失则忽略）
try:
    from udp_cmd import UDPCmdListener             # noqa: F401
    try:
        UDPCmdListener(env, port=5555)
    except OSError as e:
        print(f"[WARN] UDP 端口被占用: {e}. 跳过指令监听。")
except ModuleNotFoundError:
    pass

obs = env.reset()
print("Isaac obs0:", env._build_obs()[:20])
print("Isaac order :", env.get_joint_names())


# ---------------- 主循环 ----------------
while simulation_app.is_running():
    with torch.no_grad():
        act = policy(torch.from_numpy(obs)[None].float()).squeeze(0).cpu().numpy()
    # act = np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0], dtype=np.float32)
    cmd = env.get_cmd()
    obs, reward, done, truncated, info = env.step(act, cmd)
    if done or truncated:
        obs = env.reset()
    time.sleep(env.dt)

env.close()
