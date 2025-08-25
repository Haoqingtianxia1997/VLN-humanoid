"""
SimpleWBC — 极简 Whole-Body PD 控制器
------------------------------------
• 对每个 actuator 施加 PD 力矩：τ = Kp(q*-q) − Kd q̇
• default desired_q 为 0，可在外部脚本随时修改:
      env.wbc.desired_q[joint_index] = target_angle
"""

from __future__ import annotations
import numpy as np
import mujoco


class SimpleWBC:
    def __init__(self, model: mujoco.MjModel):
        self.model = model

        # --- PD 增益（可按关节分配不同值）
        self.kp = np.full(model.nu, 25.0)   # 刚性低一点更稳
        self.kd = np.full(model.nu, 3.0)

        # --- 目标关节角 (rad)，可外部修改
        self.desired_q = np.zeros(model.nu)

        # --- 做一个查表：actuator -> qpos index
        self._dof_addr = np.array(
            [model.actuator_trnid[i][0] for i in range(model.nu)], dtype=int
        )

    # ------------------------------------------------------------------
    def compute_torque(self, data: mujoco.MjData, action=None) -> np.ndarray:
        """
        返回长度 == model.nu 的关节力矩。
        action 参数目前未使用，留作拓展 (vx, vy, yaw 等)。
        """
        q  = data.qpos[self._dof_addr]   # 当前角度
        qd = data.qvel[self._dof_addr]   # 当前角速度

        torque = self.kp * (self.desired_q - q) - self.kd * qd

        # 安全限幅，避免抖动
        np.clip(torque, -1.0, 1.0, out=torque)
        return torque
