import matplotlib.pyplot as plt
import numpy as np


def plot_mpc_path(trajectory: np.ndarray, ocp_trajectories: np.ndarray = None, project: bool = False):
    if trajectory.shape[1] != 3:
        raise ValueError("positions must be a Nx3 array for 3D plotting.") 
    
    ocp_contains_values = ocp_trajectories is not None and len(ocp_trajectories.shape) == 3 \
            and ocp_trajectories.shape[2] == 3

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    if ocp_contains_values:
        last_idx = ocp_trajectories.shape[0] - 1
        for i, ocp_traj in enumerate(ocp_trajectories):
            ax.plot(ocp_traj[:,0], ocp_traj[:,1], ocp_traj[:,2],
                    label='OCP Trajectories' if last_idx == i else None, 
                    linestyle='--', color='orange', linewidth=1)
            
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label='SRBM MPC Path', linewidth=4)
            
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    if project and ocp_contains_values:
        for ocp_traj in ocp_trajectories:
            ax.scatter(xs=xlim[0], ys=ocp_traj[:,1], zs=ocp_traj[:,2], alpha=0.1, color='green', s=3)
            ax.scatter(xs=ocp_traj[:,0], ys=ylim[1], zs=ocp_traj[:,2], alpha=0.1, color='green', s=3)
            ax.scatter(xs=ocp_traj[:,0], ys=ocp_traj[:,1], zs=zlim[0], alpha=0.1, color='green', s=3)
    
    if project:
        for traj in trajectory:
            ax.scatter(xs=xlim[0], ys=traj[1], zs=traj[2], alpha=0.8, color='red', s=7)
            ax.scatter(xs=traj[0], ys=ylim[1], zs=traj[2], alpha=0.8, color='red', s=7)
            ax.scatter(xs=traj[0], ys=traj[1], zs=zlim[0], alpha=0.8, color='red', s=7)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title("MPC-Controlled SRBM Path")
    ax.legend()
    plt.tight_layout()
    plt.show()