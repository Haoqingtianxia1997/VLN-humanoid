import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import torch
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lstm_policy.custom_policy import MotionLSTMPolicy
from env.h1_gym_wrapper import H1GymWrapper
import gym


# ───── 设置环境 ─────
MODEL_XML_PATH = "h1_models/h1.xml"
motion_pt_path = "motion.pt"

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
        full = np.zeros(self.nu * 5, dtype=np.float32)
        full[0::5] = act_q
        full[3::5] = self.kp_array
        full[4::5] = self.kd_array
        return full

def make_env():
    env = H1GymWrapper(model_path=MODEL_XML_PATH, frame_skip=10, sleep_per_step=0.0)
    return MaskQWrapper(env)

env = DummyVecEnv([make_env])


# ───── 创建模型 ─────
print("Using cuda device")
# … 省略导入与环境封装 …

model = RecurrentPPO(
    policy = MotionLSTMPolicy,
    env    = env,
    device = "cuda",
    verbose= 1,
    n_steps=128,
    batch_size=64,
    learning_rate=3e-4,
)

# ------- 迁移 motion.pt 权重 --------
motion = torch.jit.load("motion.pt")
sd     = motion.state_dict()          # ← 关键：拿到真正的权重字典

with torch.no_grad():
    p = model.policy
    p.actor_net[0].weight.copy_(sd["actor.0.weight"])
    p.actor_net[0].bias .copy_(sd["actor.0.bias"])
    p.actor_net[2].weight.copy_(sd["actor.2.weight"])
    p.actor_net[2].bias .copy_(sd["actor.2.bias"])

    p.lstm_actor.weight_ih_l0.copy_(sd["memory.weight_ih_l0"])
    p.lstm_actor.weight_hh_l0.copy_(sd["memory.weight_hh_l0"])
    p.lstm_actor.bias_ih_l0 .copy_(sd["memory.bias_ih_l0"])
    p.lstm_actor.bias_hh_l0 .copy_(sd["memory.bias_hh_l0"])

print("✅ weights loaded, start RL finetune")
model.learn(total_timesteps=200_000)
model.save("models/recurrent_ppo_from_motion")
