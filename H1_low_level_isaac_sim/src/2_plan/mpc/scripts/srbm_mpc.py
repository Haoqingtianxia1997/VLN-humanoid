import numpy as np
import scipy.linalg as spl


def setup_weights():
    # Cost weights
    Q_pos = 1e1 * np.eye(3)
    Q_vel = 1e0 * np.eye(3)
    Q_omega = 1e0 * np.eye(3)
    Q_quat = 1e1 * np.eye(4)
    R_u = 1e-1 * np.eye(6)
    Q = spl.block_diag(Q_pos, Q_quat, Q_vel, Q_omega)

    Q_pos_e = 1e2 * np.eye(3)
    Q_vel_e = 1e0 * np.eye(3)
    Q_omega_e = 1e0 * np.eye(3)
    Q_quat_e = 1e1 * np.eye(4)
    Q_e = spl.block_diag(Q_pos_e, Q_quat_e, Q_vel_e, Q_omega_e)
    return Q, R_u, Q_e

# %%
if __name__ == "__main__":
    import numpy as np
    import scipy.linalg as spl
    from srbm_scripts import *
    import argparse

    parser = argparse.ArgumentParser(description='Generate ACADOS solver code.')
    parser.add_argument('--models_path', type=str, required=True, help='The path to the urdf file.')
    parser.add_argument('--gen_code_path', type=str, required=True, help='The base path for robot properties and output.')
    args = parser.parse_args()
    
    N_horizon = 50
    dt = 0.02

    # Cost weights
    Q, R_u, Q_e = setup_weights()

    g = 9.81
    m, com, com_inertia, contact_properties = get_h1_robot_inertial_properties(args.models_path)
    com_contact_frame = np.array([contact_properties['left_ankle_link']['com_in_link_frame'],
                            contact_properties['right_ankle_link']['com_in_link_frame']])
    
    ocp_solver = setup_acados_ocp_solver(N_horizon, dt, com_inertia, com_contact_frame, m, Q, R_u, Q_e, gen_code_path=args.gen_code_path)
    sim_solver = setup_acados_sim_solver(com_inertia, com_contact_frame, m, dt, gen_code_path=args.gen_code_path)
        
    x0 = np.array([
        0.0, 0.0, 1.0,    # Position
        1.0, 0.0, 0.0, 0.0,  # Quaternion (identity)
        0.0, 0.0, 0.0,    # Linear velocity
        0.0, 0.0, 0.0     # Angular velocity
    ])
    y_ref = np.array([
        0.8, 0.3, 1.0,    # Target position
        1.0, 0.0, 0.0, 0.0,  # Target quaternion (identity)
        0.0, 0.0, 0.0,    # Target linear velocity
        0.0, 0.0, 0.0,    # Target angular velocity
        0.0, 0.0, m*g/2,    # Target force left
        0.0, 0.0, m*g/2     # Target force right
    ])
    x_traj, *_ = solve_traj(
        x0, 
        y_ref, 
        ocp_solver=ocp_solver, 
        sim_solver=sim_solver,
        N_sim=1,
        print_ocp=False
    )
    print(f"Number of nan states: {np.count_nonzero(np.isnan(x_traj[:,2,0]))} / {x_traj.shape[0]}")