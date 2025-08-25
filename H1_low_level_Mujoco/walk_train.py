import torch
import os
import time
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.h1_gym_wrapper import H1GymWrapper  # ← 封装过的 H1StandaloneEnv
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
# 路径配置
MODEL_XML_PATH = "h1_models/h1.xml"  # ← 修改为你的 MuJoCo XML 路径
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Action Mask Wrapper：只学习 q*
# ─────────────────────────────────────────────
class MaskQWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.nu = 10
        self.kp_array = np.array([150, 150, 150, 200, 40,
                                  150, 150, 150, 200, 40], dtype=np.float32)
        self.kd_array = np.array([2, 2, 2, 4, 2,
                                  2, 2, 2, 4, 2], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-0.3 * np.ones(self.nu),
            high=0.3 * np.ones(self.nu),
            dtype=np.float32
        )

    def action(self, act_q):
        q_des = act_q
        full = np.zeros(self.nu * 5, dtype=np.float32)
        full[0::5] = q_des
        full[1::5] = 0.0
        full[2::5] = 0.0
        full[3::5] = self.kp_array
        full[4::5] = self.kd_array
        return full


def make_env():
    base = H1GymWrapper(model_path=MODEL_XML_PATH, frame_skip=10, sleep_per_step=0.0)
    return MaskQWrapper(base)

# 向量化环境
env = DummyVecEnv([make_env])

# 打开第一个环境的渲染
h1_env = env.envs[0]
h1_env.render()
h1_env.sim.sleep_sec = 0.002  # 设置刷新频率




# PPO Agent 配置
policy_kwargs = dict(net_arch=[256, 256, 128])  # 第四步代码
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device="cuda"
)

class BCPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(41, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

bc_model = BCPolicyNet()
bc_model.load_state_dict(torch.load("action_clone/bc_policy.pt"))
bc_model.eval()

ppo_actor = model.policy.mlp_extractor.policy_net



with torch.no_grad():
    for ppo_param, bc_param in zip(ppo_actor.parameters(), bc_model.parameters()):
        ppo_param.copy_(bc_param)

print("✅ PPO actor initialized from behavior cloning policy")


# checkpoint 配置
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "ppo_h1_walk.zip")
TOTAL_STEPS = 1_000_000_000
SAVE_INTERVAL = 10_000

# 如果存在旧模型，加载继续训练
if os.path.exists(CHECKPOINT_PATH):
    print(f"▶️  加载已有模型: {CHECKPOINT_PATH}")
    model = PPO.load(CHECKPOINT_PATH, env=env, device="cuda")

# 初始化策略 log_std
with torch.no_grad():
    model.policy.log_std[:] = torch.log(torch.full_like(model.policy.log_std, 1))

trained = model.num_timesteps
while trained < TOTAL_STEPS:
    chunk = min(SAVE_INTERVAL, TOTAL_STEPS - trained)
    model.learn(total_timesteps=chunk, reset_num_timesteps=False)
    trained += chunk
    model.save(CHECKPOINT_PATH)



