import time, torch, numpy as np, mujoco, mujoco.viewer
from env.h1_env import H1Env
from mpc_follow_trajectory import generate_track, mpc_control

# ---------- 参数 ----------
SIM_DT, CTRL_DEC, MPC_DT = 0.002, 10, 0.02
MPC_STEPS = int(round(MPC_DT / (SIM_DT*CTRL_DEC)))

# ---------- 环境 ----------
env   = H1Env("h1_models/h1.xml", simulation_dt=SIM_DT, control_decimation=CTRL_DEC)
obs   = env.reset()
viewer = env.render()          # 官方 GUI
policy = torch.jit.load("motion.pt")

# ---------- 轨迹 ----------

trajectory =  generate_track()

def draw_vector(vr, start_xy, vec_xy, color=(0,1,0,1), width=0.05, z=0.06):
    """在 viewer.user_scn 里画一支从 start_xy 指向 start_xy+vec_xy 的箭"""
    with vr.lock():
        scn = vr.user_scn
        if scn.ngeom >= scn.maxgeom:
            print("⚠️ user_scn 已满，无法再添加箭头")
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=[width, 0, 0],                 # radius
            mat=np.eye(3).flatten(),
            pos=[0, 0, 0],                      # 位置稍后由 mjv_connector 设置
            rgba=color
        )
        p0 = np.array([start_xy[0],            start_xy[1],            z])
        p1 = np.array([start_xy[0]+vec_xy[0],  start_xy[1]+vec_xy[1],  z])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
        scn.ngeom += 1
    vr.sync()


def reset_aligned(env, trajectory, viewer=None):
    """
    reset 环境，并把 base 对齐到轨迹起点；若传入 viewer，则同步画切线箭头
    """
    idx =0
    obs = env.reset()

    # ① 把位置钉到起点
    env.data.qpos[0] = trajectory[idx, 0]   # x
    env.data.qpos[1] = trajectory[idx+1, 1]   # y

    # ② 计算切线方向并写入 yaw（假定在 qpos[3]）
    dx, dy = (trajectory[idx+1] - trajectory[idx]) * 100
    print(f"dx={dx:.3f}, dy={dy:.3f}")
    yaw = np.arctan2(dy, dx)
    qw = np.cos(yaw / 2)
    qx = 0.0
    qy = 0.0
    qz = np.sin(yaw / 2)

    env.data.qpos[3:7] = (qw, qx, qy, qz)     # ← 注意顺序

    # ③ 清零速度 & forward
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


    # draw_vector(viewer, trajectory[idx], (dx, dy),
    #             color=(1, 0.1, 0.1, 1), width=0.05, z=0.06)

    # ⑤ 重新获取 obs（某些环境 reset 后会缓存）
    if hasattr(env, 'get_obs'):
        obs = env.get_obs()
    elif hasattr(env, '_get_obs'):
        obs = env._get_obs()
    return obs



# ---------- 画轨迹（串珠小球） ----------
def draw_track(vr, xy_seq, radius=0.1, stride=2):
    with vr.lock():                       # ① 拿锁
        scn = vr.user_scn
        scn.ngeom = 0                     # ② 清空之前可能的几何
        for xy in xy_seq[::stride]:
            if scn.ngeom >= scn.maxgeom:  # 防止越界
                break
            mujoco.mjv_initGeom(          # ③ 正规初始化
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=[xy[0], xy[1], 0.03],
                mat=np.eye(3).flatten(),  # 单位朝向矩阵
                rgba=[1, 0.1, 0.1, 1]
            )
            scn.ngeom += 1
    vr.sync()   

viewer = env.render()      
draw_track(viewer, trajectory)                            # ① 先打开 GUI
obs = reset_aligned(env, trajectory, viewer)          # ② 把 viewer 传进去

viewer.sync()

# ---------- 主循环 ----------
step_cnt   = 0


while True:
    ...
    try:
        viewer.sync()
    except Exception:
        break   # 窗口被关会抛异常；抓住后跳出                    

    if step_cnt % MPC_STEPS == 0:
        state    = env.get_base_state()
        print(f"Step {step_cnt}, state={state}")
        cmd_vel  = mpc_control(state, trajectory)      # m/s
        # print(f"Step {step_cnt}, cmd_vel={cmd_vel}")
        # -------- 把真速度归一化到 [-1,1] -------
    # 肌肉策略出 action
    act = policy(torch.from_numpy(obs).unsqueeze(0).float()).detach().cpu().numpy().squeeze()

    obs, _, done, trunc, _ = env.step(act, cmd=cmd_vel)

    # -------- 打印是否摔倒 --------
    if done:                                   # <<< ③
        print(f"Terminated? done={done}, truncated={trunc}  @step {step_cnt}")
        obs = reset_aligned(env, trajectory)            ### <<< 朝向重置
        step_cnt = 0
        continue

    viewer.sync()
    time.sleep(env.dt)
    step_cnt += 1
