import os
import numpy as np
import scipy.linalg

from acados_template import AcadosOcp, AcadosOcpSolver, builders, AcadosSim, AcadosSimSolver
from .srbm_model import srbm_model


def setup_acados_ocp_solver(
        N_horizon: int,
        dt: float,
        com_inertia: np.ndarray,
        com_contact_frame: np.ndarray,
        mass: float,
        Q: np.ndarray,
        R: np.ndarray,
        Q_e: np.ndarray = None,
        g: float = 9.81,
        epsilon: float = 1e-4,
        mu: float = 0.7,    
        gen_code_path: str = ""
) -> AcadosOcpSolver:

    # Define OCP
    ocp = AcadosOcp()
    ocp.model = srbm_model(com_inertia, mass, g, epsilon, mu)  # Identity inertia for simplicity

    nx = ocp.model.x.shape[0]  # Number of states
    nu = ocp.model.u.shape[0]  # Number of controls
    ny = nx + nu

    # Initial state setup
    x0 = np.zeros(nx)
    x0[2] = 1.0 # Initial height
    x0[3] = 1.0 # Initial quaternion (w=1, x=y=z=0)
    ocp.constraints.x0 = x0

    # Reference control input
    u_ref = np.zeros(nu)
    u_ref[2] = u_ref[5] = mass * g / 2  # Force in z-direction
    
    # Stage Cost setup l = 0.5 * || (Vx@x + Vu@u - yref) ||^2_W
    ocp.cost.cost_type = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.yref = np.concatenate([x0, u_ref])

    # Terminal Cost setup l = 0.5 * || (Vx@x - yref) ||^2_We
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = Q_e if Q_e is not None else Q
    ocp.cost.yref_e = x0

    # Equality Constraints 
    # Lower bounds: Fx, Fy can be negative; Fz must be >= 0
    # Upper bounds: All forces must be <= F_max
    F_max = mass * g
    ocp.constraints.lbu = np.array([-F_max, -F_max, 0.0, -F_max, -F_max, 0.0])
    ocp.constraints.ubu = np.array([F_max, F_max, F_max, F_max, F_max, F_max])
    ocp.constraints.idxbu = np.arange(nu)

    # Inequality Constraints [quat, friction1, friction2]
    #### Friction can be changed to soft constraint ####
    epsilon = 1e-4
    ocp.constraints.lh = np.array([-epsilon, 0, 0]) 
    ocp.constraints.uh = np.array([epsilon, 1e6, 1e6]) # practically infinite for friction constraints

    # Parameters for the model
    ocp.parameter_values = com_contact_frame.flatten()

    # Solver options
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = dt * N_horizon
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP" 
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 4
    ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.nlp_solver_max_iter = 200 # Default is often 50 or 100
    # ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.regularize_method = 'PROJECT_REDUC_HESS'  # 'MIRROR' or 'PROJECT_REDUC_HESS' are also options
    if gen_code_path:
        ocp.code_export_directory = gen_code_path

    cm_builder = builders.ocp_get_default_cmake_builder()
    cm_builder.options_on = ['BUILD_ACADOS_OCP_SOLVER_LIB']
    return AcadosOcpSolver(ocp, json_file='srbm_ocp.json', cmake_builder=cm_builder)


def setup_acados_sim_solver(
        com_inertia: np.ndarray,
        com_contact_frame: np.ndarray,
        mass: float, 
        dt: float, 
        g: float = 9.81,
        epsilon: float = 1e-4,
        mu: float = 0.7, 
        gen_code_path: str = ""
) -> AcadosSimSolver:
    sim = AcadosSim()
    sim.model = srbm_model(com_inertia, mass, g, epsilon, mu)
    sim.parameter_values = com_contact_frame.flatten()

    # Set simulation options
    sim.solver_options.T = dt  # We want to simulate one step at a time
    sim.solver_options.integrator_type = 'IRK' # Use the same robust integrator
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    if gen_code_path:
        sim.code_export_directory = gen_code_path

    # Create the solver
    sim_solver = AcadosSimSolver(sim, json_file='srbm_sim.json')
    return sim_solver