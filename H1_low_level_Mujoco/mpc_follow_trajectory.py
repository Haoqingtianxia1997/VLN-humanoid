import numpy as np  # 导入数值计算库
import casadi as ca  # 导入CasADi优化库
import matplotlib.pyplot as plt  # 导入绘图库
import matplotlib  # 设置matplotlib相关配置
from matplotlib import animation  # 动画功能
from matplotlib.patches import FancyArrowPatch  # 用于画箭头
import time  # 用于计时

# 设置支持中文字体显示
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# MPC参数
N = 5   # 预测步长
dt = 0.05     # 控制时间步长

# 代价函数权重
Q_x = 1000     # 位置x误差权重
Q_y = 1000     # 位置y误差权重
Q_theta = 1000000  # 朝向误差权重
R_vx = 0.1     # x方向速度控制惩罚
R_vy = 100    # y方向速度控制惩罚（抑制侧滑）
R_w = 0.00000001    # 角速度控制惩罚

def vehicle_dynamics(state, control):  # 车辆动力学模型
    x, y, theta = state  # 状态变量：位置和朝向
    vx, vy, w = control  # 控制变量：x/y方向速度与角速度
    dx = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
    dy = (vx * np.sin(theta) + vy * np.cos(theta)) * dt
    dtheta = w * dt
    return np.array([x + dx, y + dy, theta + dtheta])

def mpc_control(state, trajectory):  # MPC控制器
    opti = ca.Opti()  # 创建优化器
    distances = np.linalg.norm(trajectory[:, :2] - state[:2], axis=1)  # 当前点与轨迹距离
    nearest_idx = np.argmin(distances)  # 找到最近点索引
    ref_points = trajectory[nearest_idx:nearest_idx + N + 1]  # 获取预测点

    if len(ref_points) < N + 1:  # 如果不够则补齐
        pad = np.tile(ref_points[-1], (N + 1 - len(ref_points), 1))
        ref_points = np.vstack((ref_points, pad))

    dx_array = ref_points[1:, 0] - ref_points[:-1, 0]  # 差分计算方向
    dy_array = ref_points[1:, 1] - ref_points[:-1, 1]
    theta_refs = np.unwrap(np.arctan2(dy_array, dx_array))  # 计算并展开参考朝向

    X = opti.variable(3, N + 1)  # 状态变量
    U = opti.variable(3, N)      # 控制变量
    opti.subject_to(X[:, 0] == state)  # 初始状态约束

    cost = 0  # 初始化代价
    for k in range(N):
        x_ref, y_ref = ref_points[k, :2]  # 当前参考点位置
        theta_ref = theta_refs[k]         # 当前参考点朝向
        theta_err = ca.atan2(ca.sin(X[2, k] - theta_ref), ca.cos(X[2, k] - theta_ref))  # 处理朝向跳变

        # 构建代价函数
        cost += Q_x * (X[0, k] - x_ref) ** 2
        cost += Q_y * (X[1, k] - y_ref) ** 2
        cost += Q_theta * theta_err ** 2
        cost += R_vx * U[0, k] ** 2 + R_vy * U[1, k] ** 2 + R_w * U[2, k] ** 2

        # 系统动力学约束
        theta_k = X[2, k]
        dx = (U[0, k] * ca.cos(theta_k) - U[1, k] * ca.sin(theta_k)) * dt
        dy = (U[0, k] * ca.sin(theta_k) + U[1, k] * ca.cos(theta_k)) * dt
        dtheta = U[2, k] * dt
        opti.subject_to(X[:, k + 1] == X[:, k] + ca.vertcat(dx, dy, dtheta))

    opti.minimize(cost)  # 设置优化目标

    # 控制输入约束
    opti.subject_to(opti.bounded(-0.5, U[0, :], 0.5))
    opti.subject_to(opti.bounded(-0.2, U[1, :], 0.2))
    opti.subject_to(opti.bounded(-0.7, U[2, :], 0.7))
    opti.subject_to(U[0, :] >= 0)  # 禁止后退

    opts = {'ipopt.print_level': 0, 'print_time': 0}  # 求解器选项
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()  # 尝试求解
        return sol.value(U[:, 0])  # 返回第一步控制量
    except:
        print("MPC求解失败，使用默认控制")
        return np.array([0.5, 0.0, 0.0])  # 默认前进

def generate_track():  # 生成测试轨迹
    t = np.linspace(0, 2 * np.pi, 300)
    x = 5 * np.cos(t) + 1.5 * np.sin(2 * t)
    y = 5 * np.sin(t) + 1.0 * np.cos(3 * t)
    return np.column_stack((x, y))

def calculate_tracking_error(path_followed, trajectory):  # 误差评估
    from scipy.interpolate import interp1d
    path_length = len(path_followed)
    traj_length = len(trajectory)
    interp_x = interp1d(np.linspace(0, 1, traj_length), trajectory[:, 0], kind='linear')
    interp_y = interp1d(np.linspace(0, 1, traj_length), trajectory[:, 1], kind='linear')
    resampled_x = interp_x(np.linspace(0, 1, path_length))
    resampled_y = interp_y(np.linspace(0, 1, path_length))
    resampled_traj = np.column_stack((resampled_x, resampled_y))
    errors = np.linalg.norm(path_followed[:, :2] - resampled_traj, axis=1)
    return errors

def simulate_mpc(trajectory):  # 仿真主函数
    path = []
    computation_times = []
    theta0 = np.arctan2(trajectory[1, 1] - trajectory[0, 1], trajectory[1, 0] - trajectory[0, 0])
    state = np.array([trajectory[0, 0], trajectory[0, 1], theta0])
    path.append(state.copy())
    for _ in range(1500):
        start_time = time.time()
        u = mpc_control(state, trajectory)
        state = vehicle_dynamics(state, u)
        path.append(state.copy())
        computation_times.append(time.time() - start_time)
    return np.array(path), computation_times

def draw_robot(ax, state, size=0.4):  # 绘制机器人（方块+箭头）
    x, y, theta = state
    half = size / 2
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = corners @ rot.T + np.array([x, y])
    box = ax.plot(rotated[:, 0], rotated[:, 1], 'k-')[0]
    arrow_dx = size * np.cos(theta)
    arrow_dy = size * np.sin(theta)
    arrow = FancyArrowPatch((x, y), (x + arrow_dx, y + arrow_dy), color='r', arrowstyle='->', mutation_scale=15)
    ax.add_patch(arrow)
    return box, arrow

def animate_simulation(trajectory, path):  # 动画
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='参考轨迹')
    box, arrow = draw_robot(ax, path[0])
    ax.set_xlim(np.min(trajectory[:, 0]) - 1, np.max(trajectory[:, 0]) + 1)
    ax.set_ylim(np.min(trajectory[:, 1]) - 1, np.max(trajectory[:, 1]) + 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    def update(i):  # 更新每一帧
        state = path[i]
        x, y, theta = state
        half = 0.2
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = corners @ rot.T + np.array([x, y])
        box.set_data(rotated[:, 0], rotated[:, 1])
        arrow_dx = 0.4 * np.cos(theta)
        arrow_dy = 0.4 * np.sin(theta)
        arrow.set_positions((x, y), (x + arrow_dx, y + arrow_dy))
        return box, arrow

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=dt * 1000, blit=False)
    plt.title('MPC 路径跟踪动画（带朝向箭头）')
    plt.show()

def main():  # 主程序
    trajectory = generate_track()  # 生成轨迹
    print(f"\n轨迹总长度: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.2f} m")
    path, computation_times = simulate_mpc(trajectory)  # 跟踪路径
    errors = calculate_tracking_error(path, trajectory)  # 误差统计
    total_sim_time = len(path) * dt
    print(f"仿真跑一圈所需时间: {total_sim_time:.2f} s")
    print(f"\n计算时间统计:")
    print(f"平均计算时间: {np.mean(computation_times)*1000:.2f} ms")
    print(f"最大计算时间: {np.max(computation_times)*1000:.2f} ms")
    print(f"最小计算时间: {np.min(computation_times)*1000:.2f} ms")
    print(f"总计算时间: {np.sum(computation_times):.2f} s")
    print(f"\n跟踪误差统计:")
    print(f"平均跟踪误差: {np.mean(errors):.4f} m")
    print(f"最大跟踪误差: {np.max(errors):.4f} m")

    plt.figure(figsize=(12, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='参考轨迹')
    plt.plot(path[:, 0], path[:, 1], 'r--', label='MPC路径')
    plt.grid(True)
    plt.legend()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.title('路径跟踪结果')
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(computation_times, label='MPC计算时间')
    plt.grid(True)
    plt.xlabel('步骤')
    plt.ylabel('时间 (s)')
    plt.title('每步MPC计算耗时')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(errors, label='跟踪误差')
    plt.grid(True)
    plt.xlabel('步骤')
    plt.ylabel('误差 (m)')
    plt.title('跟踪误差随时间变化')
    plt.legend()
    plt.show()

    animate_simulation(trajectory, path)  # 播放动画

if __name__ == '__main__':  # 入口函数
    main()







