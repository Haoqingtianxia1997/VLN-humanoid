import gym
import numpy as np
from env.h1_env import H1Env

class H1GymWrapper(gym.Env):
    def __init__(self, model_path, frame_skip=10, sleep_per_step=0.0):
        super().__init__()
        self.sim = H1Env(model_path, simulation_dt=0.002, control_decimation=frame_skip)
        self.sim.sleep_sec = sleep_per_step

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32  # 实际上只学 q*
        )

    def reset(self, seed=None, options=None):
        obs = self.sim.reset()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.sim.step(action)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
