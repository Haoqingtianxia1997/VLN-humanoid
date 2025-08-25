# h1_demo.py – 适配 Isaac Lab 0.40.x
import argparse, math, torch
from isaaclab.app import AppLauncher

# ---------- CLI & 启动 ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
cli_args = parser.parse_args()
simulation_app = AppLauncher(cli_args).app        # 其他 import 之前！

# ---------- Isaac Lab import ---------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim      import SimulationContext, SimulationCfg
from isaaclab.scene    import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets   import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ---------- H1 机器人描述 -------------------------------------------------
H1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/H1",
    spawn      = sim_utils.UsdFileCfg(
        usd_path="h1_models/h1.usd"   
        # usd_path="h1_models/h1_total.usd"
  
    ),
    actuators={
        # 一定写成「列表」，别写成字符串或 None
        "all": ImplicitActuatorCfg(
            joint_names_expr=[r".*"],     # ← 关键！一定要是 list[str]
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=2_000.0,
            damping=50.0,
        )
    },
    # ★ 不给 joint_pos ——> 用 USD 默认（全部 0）
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0)            # 站在 1 m 高空
    ),
)

# ---------- 场景 ----------------------------------------------------------
class H1SceneCfg(InteractiveSceneCfg):
    # ground & light
    ground     = AssetBaseCfg(
        prim_path="/World/Ground",                       
        spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",                        
        spawn=sim_utils.DomeLightCfg(intensity=5000)
    )
    H1         = H1_CFG

# ---------- 主循环 --------------------------------------------------------
def sim_loop(sim: SimulationContext, scene: InteractiveScene):
    sim_dt, sim_time, step = sim.get_physics_dt(), 0.0, 0

    # 预设一个简易 PD——完全在 Python 侧算力矩
    dof_names = scene["H1"].data.joint_names
    num_dofs  = len(dof_names)

    kp = torch.ones(num_dofs, device=sim.device) * 200.0
    kd = torch.ones(num_dofs, device=sim.device) * 10.0
    q0 = scene["H1"].data.default_joint_pos.clone()  # 初始角

    joint_names = scene["H1"].data.joint_names  # list[str]
    print("机器人关节名列表:\n", joint_names)

    while simulation_app.is_running():
        if step % int(3.0 / sim_dt) == 0:
            scene.reset()
            print("[INFO] reset H1")

        # ---- ① 生成期望关节角 ----
        q_des = q0 + 0.25 * math.sin(2 * math.pi * 0.5 * sim_time)

        # ---- ② 读取当前状态 ----
        q  = scene["H1"].data.joint_pos[0]   # (DOF,)
        dq = scene["H1"].data.joint_vel[0]

        # ---- ③ 计算力矩 ----
        tau = kp * (q_des - q) - kd * dq     # (DOF,)

        # ---- ④ 写入力矩目标 ----
        scene["H1"].set_joint_effort_target(tau.unsqueeze(0))  # shape (1, DOF)

        # ---- ⑤ 物理步进 ----
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        step += 1

# position control
# def sim_loop(sim: SimulationContext, scene: InteractiveScene):
#     sim_dt, sim_time, step = sim.get_physics_dt(), 0.0, 0
#     default_q = scene["H1"].data.default_joint_pos.clone()

#     while simulation_app.is_running():
#         # 每 3 秒复位
#         if step % int(3.0 / sim_dt) == 0:
#             scene.reset()
#             print("[INFO] reset H1")

#         # 让每个关节做正弦摆动  🔽 这里改了一行
#         q_des = default_q + 0.25 * math.sin(2 * math.pi * 0.5 * sim_time)

#         scene["H1"].set_joint_position_target(q_des)

#         # 写入 & 物理步进
#         scene.write_data_to_sim()
#         sim.step()
#         scene.update(sim_dt)


#         q_tensor = scene["H1"].data.joint_pos       # shape: (num_envs, num_dofs)
#         q_numpy  = q_tensor.cpu().numpy()           # 如果你想转成 numpy

#         # 单环境最常见的写法
#         q = q_tensor[0]            # torch.Tensor, size = DOF
#         print("当前关节角:", q.tolist())
#         # 计数
#         sim_time += sim_dt
#         step += 1

# ---------- main ----------------------------------------------------------
def main():
    sim = SimulationContext(SimulationCfg(device=cli_args.device))
    sim.set_camera_view([4, 0, 2], [0, 0, 1])

    scene = InteractiveScene(H1SceneCfg(cli_args.num_envs, env_spacing=3.0))
    sim.reset(); scene.write_data_to_sim()
    print("[INFO] scene ready, starting sim ...")

    sim_loop(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()





