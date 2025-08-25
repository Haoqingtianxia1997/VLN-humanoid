# 评价 ppo 模型
# import torch
# import os
# import time
# import gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from env.h1_gym_wrapper import H1GymWrapper  # ← 封装过的 H1StandaloneEnv

# # 路径配置
# MODEL_XML_PATH = "h1_models/h1.xml"  # ← 修改为你的 MuJoCo XML 路径
# SAVE_DIR = "models"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ─────────────────────────────────────────────
# # Action Mask Wrapper：只学习 q*
# # ─────────────────────────────────────────────
# class MaskQWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.nu = 10
#         self.kp_array = np.array([150, 150, 150, 200, 40,
#                                   150, 150, 150, 200, 40], dtype=np.float32)
#         self.kd_array = np.array([2, 2, 2, 4, 2,
#                                   2, 2, 2, 4, 2], dtype=np.float32)

#         self.action_space = gym.spaces.Box(
#             low=-0.3 * np.ones(self.nu),
#             high=0.3 * np.ones(self.nu),
#             dtype=np.float32
#         )

#     def action(self, act_q):
#         q_des = act_q
#         full = np.zeros(self.nu * 5, dtype=np.float32)
#         full[0::5] = q_des
#         full[1::5] = 0.0
#         full[2::5] = 0.0
#         full[3::5] = self.kp_array
#         full[4::5] = self.kd_array
#         return full


# def make_env():
#     base = H1GymWrapper(model_path=MODEL_XML_PATH, frame_skip=10)
#     return base


# env = make_env()


# CHECKPOINT_PATH = os.path.join(SAVE_DIR, "ppo_h1_walk.zip")
# model = PPO.load(CHECKPOINT_PATH, env=env, device="cuda")

# def evaluate(model, seconds=30):
#     env = make_env()
#     obs, _ = env.reset()
    
#     env.render()
#     t_end = time.time() + seconds
#     while time.time() < t_end:
#         act_q, _ = model.predict(obs, deterministic=True)  # 10 维
#         # print(f"Act Q : {act_q}")
#         obs, reward, terminated, truncated, _ = env.step(act_q)
#         # print(f"Obs 1 : {obs}")
#         done = terminated or truncated
#         if done:
#             obs, _ = env.reset()
#         time.sleep(0.002)
#     env.close()

# print("▶️  回放 30 秒 …")
# evaluate(model)














# 原始版本评价motion.pt模型的代码如下：
# import time
# import torch
# import mujoco
# import mujoco.viewer
# import numpy as np
# def get_gravity_orientation(quaternion):
#     qw, qx, qy, qz = quaternion
#     gravity_orientation = np.zeros(3)
#     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
#     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
#     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
#     return gravity_orientation


# def pd_control(target_q, q, kp, target_dq, dq, kd):
#     return (target_q - q) * kp + (target_dq - dq) * kd


# def main():
#     # === 手动配置 ===
#     policy_path = "motion.pt"       # 改成你自己的路径
#     xml_path = "h1_models/h1.xml"          # 改成你自己的路径

#     simulation_duration = 120.0
#     simulation_dt = 0.002
#     control_decimation = 10

#     kps = np.array([150, 150, 150, 200, 40,
#                     150, 150, 150, 200, 40], dtype=np.float32)
#     kds = np.array([2, 2, 2, 4, 2,
#                     2, 2, 2, 4, 2], dtype=np.float32)

#     default_angles = np.array([0, 0.0, -0.1, 0.3, -0.2,
#                                0, 0.0, -0.1, 0.3, -0.2], dtype=np.float32)

#     ang_vel_scale = 0.25
#     dof_pos_scale = 1.0
#     dof_vel_scale = 0.05
#     action_scale = 0.25
#     cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)

#     num_actions = 10
#     num_obs = 41
#     cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

#     # === 加载模型和数据 ===
#     m = mujoco.MjModel.from_xml_path(xml_path)
#     d = mujoco.MjData(m)
#     m.opt.timestep = simulation_dt

#     # === 加载策略 ===
#     policy = torch.jit.load(policy_path)

#     obs = np.zeros(num_obs, dtype=np.float32)
#     action = np.zeros(num_actions, dtype=np.float32)
#     target_dof_pos = default_angles.copy()
#     counter = 0

#     # === 启动模拟器 ===
#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         start = time.time()
#         while viewer.is_running() and time.time() - start < simulation_duration:
#             step_start = time.time()

#             # === PD 控制 ===
#             tau = pd_control(
#                 target_dof_pos, d.qpos[7:], kps,
#                 np.zeros_like(kds), d.qvel[6:], kds
#             )
#             d.ctrl[:] = tau

#             mujoco.mj_step(m, d)

#             # === 控制频率更新策略 ===
#             counter += 1
#             if counter % control_decimation == 0:
#                 quat = d.qpos[3:7]
#                 omega = d.qvel[3:6]
#                 qj = d.qpos[7:]
#                 dqj = d.qvel[6:]

#                 qj = (qj - default_angles) * dof_pos_scale
#                 dqj = dqj * dof_vel_scale
#                 omega = omega * ang_vel_scale
#                 gravity = get_gravity_orientation(quat)

#                 phase = (counter * simulation_dt) % 0.8 / 0.8
#                 sin_phase = np.sin(2 * np.pi * phase)
#                 cos_phase = np.cos(2 * np.pi * phase)

#                 obs[:3] = omega
#                 obs[3:6] = gravity
#                 obs[6:9] = cmd * cmd_scale
#                 obs[9:9 + num_actions] = qj
#                 obs[9 + num_actions:9 + 2 * num_actions] = dqj
#                 obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
#                 obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = [sin_phase, cos_phase]

#                 obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
#                 action = policy(obs_tensor).detach().cpu().numpy().squeeze()
#                 target_dof_pos = action * action_scale + default_angles

#             viewer.sync()

#             time_until_next_step = m.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)
# if __name__ == "__main__":
#     main()















# # 利用H1env 评价motion.pt模型
# import time
# import torch
# import numpy as np
# from env.h1_env import H1Env  # ← 确保这个路径指向你封装好的 H1Env


# def main():
#     # === 配置路径 ===
#     policy_path = "motion.pt"
#     model_path = "h1_models/h1.xml"
#     cmd = np.array([0.04, 0, 0.0], dtype=np.float32)  # 控制命令
#     # === 初始化环境和策略 ===
#     env = H1Env(model_path=model_path, simulation_dt=0.002, control_decimation=10)
#     policy = torch.jit.load(policy_path)

#     obs = env.reset()       # 会自动加扰动，和训练一致
#     env.render()            # 启动 GUI 可视化

#     start_time = time.time()
#     while time.time() - start_time < 120.0:
#         # 推理动作（保持和训练一致）
#         obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
#         action = policy(obs_tensor).detach().cpu().numpy().squeeze()
#         # 环境步进（自动 PD 控制）
#         obs, reward, terminated, truncated, _ = env.step(action, cmd)

#         done = terminated or truncated
#         # if done:
#         #     obs = env.reset()
#         # 控制仿真速度
#         time.sleep(env.dt)

#     env.close()


# if __name__ == "__main__":
#     main()












# 用python脚本控制 H1 走路（UDP 命令源）：
import time
import torch
import numpy as np
from env.h1_env import H1Env  # ← 确保这个路径指向你封装好的 H1Env
from udp_cmd import UDPCmdListener

def main():
    model_path  = "h1_models/h1.xml"
    policy_path = "motion.pt"

    env    = H1Env(model_path, simulation_dt=0.002, control_decimation=10)
    joint_names = env.get_joint_names()
    print("所有关节名称：")
    for i, name in enumerate(joint_names):
        print(f"{i}: {name}")

    env.set_cmd(np.array([0.04, 0.0, 0.0], dtype=np.float32))
    policy = torch.jit.load(policy_path)

    # ------------ 开 UDP 背景线程 ------------
    UDPCmdListener(env, port=5556)

    obs = env.reset()
    print("MJ obs0:", env.get_obs()[:20])  # 前 20 维即可
    print("MJ order :", env.get_joint_names())
    env.render()

    while True:                                  # 无限跑；Ctrl-C 退出
        with torch.no_grad():                                 # ① 关闭梯度
            act = policy(torch.from_numpy(obs)[None].float()) \
                    .squeeze(0).cpu().numpy()         
        cmd  = env.get_cmd()                     # 当前指令（随时可能被线程更新）
        obs, *_ = env.step(act, cmd)
        time.sleep(env.dt)
        # env.render()

main()
















# # 命令行版本：从 stdin 读入命令
# import time
# import torch
# import numpy as np
# from env.h1_env import H1Env  # ← 确保这个路径指向你封装好的 H1Env
# from udp_cmd import UDPCmdListener
# import threading, sys, numpy as np, time, torch

# def main():
#     model_path  = "h1_models/h1.xml"
#     policy_path = "motion.pt"

#     env    = H1Env(model_path, simulation_dt=0.002, control_decimation=10)
#     env.set_cmd(np.array([0.04, 0.0, 0.0], np.float32))      # 默认指令
#     policy = torch.jit.load(policy_path)

#     # ---------- ① 后台线程：从 stdin 读 cmd ----------
#     def stdin_listener():
#         print("\n请输入 vx vy wz (空格分隔)，回车立即生效；例如 0.1 0.0 0.3")
#         for line in sys.stdin:
#             try:
#                 vx, vy, wz = map(float, line.strip().split())
#                 env.set_cmd(np.array([vx, vy, wz], np.float32))
#                 print(f"新 cmd -> vx={vx:.3f}  vy={vy:.3f}  wz={wz:.3f}")
#             except ValueError:
#                 print("⚠️  格式错误，请重新输入 例如：0.08 -0.02 0.5")

#     threading.Thread(target=stdin_listener, daemon=True).start()

#     # ---------- ② 启动仿真 ----------
#     obs = env.reset()
#     env.render()

#     while True:                                    # Ctrl-C 终止
#         with torch.no_grad():
#             act = policy(torch.from_numpy(obs)[None].float()) \
#                    .squeeze(0).cpu().numpy()

#         cmd = env.get_cmd()                        # 随时可能被改
#         obs, *_ = env.step(act, cmd)
#         time.sleep(env.dt)


# main()