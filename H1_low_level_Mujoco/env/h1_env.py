import numpy as np
import mujoco
import mujoco.viewer
import time
import torch
from scipy.spatial.transform import Rotation as R

class H1Env:
    def __init__(self, model_path, simulation_dt=0.002, control_decimation=10):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = simulation_dt

        self.dt = simulation_dt
        self.control_decimation = control_decimation
        self.viewer = None

        self.nu = self.model.nu
        self.dof_ids = self.model.actuator_trnid[:, 0]

        # config
        self.default_q = np.array([0, 0.0, -0.1, 0.3, -0.2,
                                   0, 0.0, -0.1, 0.3, -0.2], dtype=np.float32)
        self.kps = np.array([150, 150, 150, 200, 40, 150, 150, 150, 200, 40], dtype=np.float32)
        self.kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 4, 2], dtype=np.float32)

        self.cmd = np.array([0.08, 0.0, 0.0], dtype=np.float32)
        self.cmd_scale = np.array([4.0, 4.0, 0.5], dtype=np.float32)

        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25
        self.action_scale = 0.25

        self.num_actions = self.nu
        self.num_obs = 41
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_q = self.default_q.copy()
        self.step_counter = 0

        self._x0y0 = np.zeros(2)
        self._q0 = self.default_q.copy()
        self.total_steps = 0
        self.policy = torch.jit.load("motion.pt")
        self.max_episode_steps = 1000  # 最大步数，训练时为 1000 步
        self.max_episode_time  = 30      # 秒
        self.episode_steps = 0
        self.episode_start_time = None

    def reset(self, seed=None, options=None):
        # ========== Mujoco 状态清零 ==========
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0
        self.data.qpos[2] = 2.0                   # 建议保持训练时的 1.1
        self.data.qvel[:] = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()


        yaw_offset = np.deg2rad(0)         # ← 想让它一开局朝向 +45° 就写 45
        quat_yaw   = R.from_euler('x', yaw_offset).as_quat()   # [x y z w]
        self.data.qpos[3:7] = quat_yaw      # 写进 base 四元数
        # ========== 生成随机扰动 ==========
        
        noise = np.random.normal(loc=0.0, scale=0.1, size=self.default_q.shape)
        q_init = self.default_q #+ noise
        q_init = np.clip(q_init, -0.3, 0.3)        # 与动作空间一致的安全范围

        self.data.qpos[7:] = q_init                # 加上扰动的初始姿态
        self.target_q = q_init.copy()              # 目标角同步为当前角
        mujoco.mj_step(self.model, self.data)
        # ========== 计数器 / 参考状态重置 ==========
        self.step_counter = 0
        self._x0y0 = self.data.qpos[0:2].copy()
        self._q0   = self.data.qpos[self.dof_ids].copy()
        self._start_yaw = self.get_base_yaw() * -1 # 获取初始 yaw 角度
        info = {}                                  # Gymnasium 需要 info

        return self.get_obs()



    def step(self, action= None , cmd =None ):#，obs=None
        # obs = self.get_obs()
        # obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        # print(f"Obs : {obs}")  # 调试输出
        # self.target_action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        # print(f"Target action: {self.target_action}")  # 调试输出
        # print(f"Action : {action}")  # 调试输出
        # action = self.target_action  # 使用策略网络输出的动作
        # self.action = action  # 使用策略网络输出的动作  

        self.cmd = cmd if cmd is not None else self.cmd  # 保持原有命令

        #
        vx,vy,wz = self.get_episode_average_velocity()
        # print(f"Episode average velocity - vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}")  # 调试输出 
        # 获取当前速度信息
        # vx, vy, wz = self.get_velocity_info()
        # print(f"Velocity info - vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}")  # 调试输出


        if action is not None:
            if action.shape[0] == self.nu * 5:
                q_des = action[0::5]  # 提取每组的第一个（q*）
            elif action.shape[0] == self.nu:
                q_des = action  # 原始 10 维动作，直接用
            else:
                raise ValueError(f"Invalid action shape: {action.shape}")
        
            # noise = np.random.normal(loc=0.0, scale=0.15, size=self.default_q.shape)

            self.target_q = q_des * self.action_scale + self.default_q # np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) #
                      
        self.action = q_des


        for _ in range(self.control_decimation):

            # self.data.qpos[2] = 2.0                   # 建议保持训练时的 1.1
            # self.data.qvel[:] = 0

            tau = self._compute_pd_torque(self.target_q)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            self.step_counter += 1          # ← 计数器移到这里
            if self.viewer:
                self.viewer.sync()



        obs = self.get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        # ---------- truncate 逻辑 ----------
        self.episode_steps += 1
        elapsed_time = time.time() - self.episode_start_time
        truncated = (
            self.episode_steps >= self.max_episode_steps
            or elapsed_time    >= self.max_episode_time
        )
        self.total_steps += 1
        return obs, reward, done, truncated, info

    def get_obs(self):
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        # print(f"q: {q}, dq: {dq}, quat: {quat}, omega: {omega}")  # 调试输出
        gravity = self._gravity_orientation(quat)
        phase = (self.step_counter * self.dt) % 0.8 / 0.8
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs = self.obs
        obs[:3] = omega * self.ang_vel_scale
        obs[3:6] = gravity
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9:19] = (q - self.default_q) * self.dof_pos_scale
        obs[19:29] = dq * self.dof_vel_scale
        obs[29:39] = self.action
        obs[39:41] = [sin_phase, cos_phase]
        return obs.copy()

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _compute_pd_torque(self, target_q):
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        return self.kps * (target_q - q) - self.kds * dq

    def _gravity_orientation(self, quat):
        qw, qx, qy, qz = quat
        g = np.zeros(3)
        g[0] = 2 * (-qz * qx + qw * qy)
        g[1] = -2 * (qz * qy + qw * qx)
        g[2] = 1 - 2 * (qw * qw + qz * qz)
        return g


    def _compute_reward(self):
        # """
        # 多阶段“稳站”奖励函数
        # - 第一阶段(0~20%)：只学高度、姿态、活着
        # - 第二阶段(20%~100%)：逐渐加入速度、能耗、关节偏差惩罚
        # """
        # # ------------- 基本状态 -------------
        # roll, pitch, _ = self._quat_to_euler(self.data.xquat[0])
        # z      = self.data.qpos[2]
        # v_xy   = self.data.qvel[0:2]
        # xy_disp = self.data.qpos[0:2] - self._x0y0

        # q      = self.data.qpos[self.dof_ids]
        # qd     = self.data.qvel[self.dof_ids]
        # tau    = self.data.ctrl

        # # ------------- 课程调度比例 -------------
        # #   0   →  1   对应 0 → 200_000 环境步
        # prog = np.clip(self.total_steps / 200_000.0, 0.0, 1.0)

        # # ------------- 主要奖励 -------------
        # z_ref = 1.05 + 0.05 * prog                            # 高度目标逐渐抬高
        # r_h   = np.exp(-15.0 * (z - z_ref) ** 2)              # 0~1
        # r_ori = np.cos(roll) * np.cos(pitch)                  # 0~1 (≈1 表示直立)
        # r_live = 0.5 * (1.0 - prog) + 0.1                     # 0.5→0.1

        # # ------------- 惩罚项（权重随 prog 递增） -------------
        # # 横向位移（想让它站在原地）
        # p_disp  = 0.5 * 2.0 * np.sum(xy_disp ** 2)

        # # COM 横向速度，阈值 0.05 m/s 以内不惩罚
        # v_mag   = np.linalg.norm(v_xy)
        # p_comv  = 0.05 * max(0.0, v_mag - 0.05) ** 2          # 平滑 ReLU²

        # # 关节速度、能耗、姿态偏差（课程调度）
        # w_qd   = 0.002 * prog
        # w_tau  = 0.001 * prog
        # w_qdev = 0.20  * prog

        # p_vel  = w_qd   * np.sum(qd  ** 2)
        # p_tau  = w_tau  * np.sum(tau ** 2)
        # p_qdev = w_qdev * np.sum((q - self._q0) ** 2)

        # # ------------- 合成 & 软裁剪 -------------
        # reward_raw = (
        #     r_h + r_ori + r_live
        #     - (p_disp + p_comv + p_vel + p_tau + p_qdev)
        # )
        # # print(f"Reward raw: {reward_raw:.4f} | " )
        # # 使用 tanh 进行软裁剪，输出范围 (-1, 1)
        # reward = float(np.tanh(reward_raw / 5.0))
        # return reward
        return 0.0  # 这里返回 0.0 作为占位符，实际奖励函数需要根据具体任务实现


    def _check_done(self):

        too_low = self.data.qpos[2] < 0.6
        return bool(too_low)


    
    def get_velocity_info(self):
        vx = self.data.qvel[0]   # x 方向线速度
        vy = self.data.qvel[1]   # y 方向线速度
        wz = self.data.qvel[5]   # 绕 z 轴的角速度
        return vx, vy, wz
    
    def get_episode_average_velocity(self):
        current_xy = self.data.qpos[0:2]
        current_yaw = self.get_base_yaw()  # 获取当前 yaw 角度
        elapsed_time = time.time() - self.episode_start_time

        if elapsed_time <= 1e-6:
            return np.array([0.0, 0.0, 0.0])  # 防止除以 0

        delta_xy = current_xy - self._x0y0
        delta_yaw = current_yaw - self._start_yaw
     

        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi

        vx_avg = delta_xy[0] / elapsed_time
        vy_avg = delta_xy[1] / elapsed_time
        wz_avg = delta_yaw / elapsed_time

        return vx_avg, vy_avg, wz_avg
    
   

    def get_base_yaw(self):
        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        quat = self.data.xquat[pelvis]
        r = R.from_quat(quat)  
        euler = r.as_euler('xyz', degrees=False)  # 返回 [roll, pitch, yaw]
        return euler[0] * -1 + 3.1415926 # yaw
    

    def set_cmd(self, new_cmd: np.ndarray):
        self.cmd = new_cmd.astype(np.float32)

    def get_cmd(self):
        return self.cmd.copy()
    
    def get_base_state(self):
        """世界坐标系下的 (x, y, yaw)"""
        x, y = self.data.qpos[0:2]
        yaw   = self.get_base_yaw()
        return np.array([x, y, yaw], dtype=np.float32)
    
    def get_joint_names(self):
        """返回所有关节的名称列表"""
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(self.model.njnt)]