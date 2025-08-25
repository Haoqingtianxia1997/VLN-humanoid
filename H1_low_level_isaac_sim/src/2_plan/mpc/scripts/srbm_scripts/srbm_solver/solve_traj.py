import numpy as np

from acados_template import AcadosOcpSolver, AcadosSimSolver

def solve_traj(
        init_state: np.ndarray,
        y_ref: np.ndarray, 
        ocp_solver: AcadosOcpSolver, 
        sim_solver: AcadosSimSolver = None,
        N_sim: int = 1,
        print_ocp: bool = True,
) -> tuple[list, list]:
    N_horizon = ocp_solver.acados_ocp.solver_options.N_horizon
    nx = ocp_solver.acados_ocp.model.x.shape[0]  # Number of states
    nu = ocp_solver.acados_ocp.model.u.shape[0]  # Number of controls

    x_traj = np.full((N_sim, N_horizon + 1, nx), np.nan)
    u_traj = np.full((N_sim, N_horizon, nu), np.nan)
    x_sim = np.full((N_sim + 1, nx), np.nan)
    x_current = init_state.copy()
    x_traj[0, 0] = init_state.copy()

    if sim_solver is not None:
        sim_solver.set('x', init_state.copy())
        x_sim[0] = init_state.copy()
        
        param_values = ocp_solver.get(0, 'p')
        for k in range(N_horizon + 1):
            sim_solver.set('p', param_values)

    # Set references
    for k in range(N_horizon):
        ocp_solver.set(k, "yref", y_ref)
    ocp_solver.set(N_horizon, "yref", y_ref[:nx])

    for i in range(N_sim):
        # set initial state
        ocp_solver.set(0, "lbx", x_current)
        ocp_solver.set(0, "ubx", x_current)

        # Solve MPC
        status = ocp_solver.solve()
        if status != 0:
            print(f"Solver failed at stage {i} with status {status}")
            return x_traj, u_traj, x_sim
        
        # get solver states and inputs
        for k in range(N_horizon + 1):
            xk = ocp_solver.get(k, "x")
            x_traj[i, k] = xk
        for k in range(N_horizon):
            uk = ocp_solver.get(k, "u")
            u_traj[i, k] = uk

        # use simulation results
        if sim_solver is not None:
            sim_solver.set('u', u_traj[i, 0])
            sim_solver.set('x', x_current.copy())
            sim_solver.solve()
            x_current = sim_solver.get('x')
            x_sim[i + 1] = x_current.copy()
        else:
            x_current = x_traj[i, 1]

        if i < (N_sim - 1):
            x_traj[i + 1, 0] = x_current.copy()

        if print_ocp:
            print("States:")
            for k, xk in enumerate(x_traj[i]):
                value_strings = [f"{xkj:.3g}" for xkj in xk]
                print(f"Stage {k}: {value_strings}")

            print("Inputs:")
            for k, uk in enumerate(u_traj[i]):
                value_strings = [f"{uk_j:.3g}" for uk_j in uk]
                print(f"Stage {k}: {value_strings}")
    return x_traj, u_traj, x_sim
