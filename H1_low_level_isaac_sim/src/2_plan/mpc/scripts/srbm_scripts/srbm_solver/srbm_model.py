import casadi as cs
import numpy as np

from casadi import MX, DM
from acados_template import AcadosModel

from .helper import quat_mul


def srbm_model(
        inertia: np.ndarray, 
        mass: float, 
        g: float = 9.81,
        epsilon: float = 1e-4,
        mu: float = 0.7,
) -> AcadosModel:
    # Inertia matrix (3x3) flattened to 9x1 for CasADi
    I = DM(inertia)  # Ensure Inertia is a CasADi DM for compatibility
    I_inv = cs.inv(I)  # Inverse inertia matrix (3x3)
    gravity = cs.vertcat(0.0, 0.0, -g)  # Gravity vector (3x1)
    # r_c1_inertial = DM(rc_inertial[0])  # Center of mass for contact 1 in inertial frame
    # r_c2_inertial = DM(rc_inertial[1])  # Center of mass for contact 2 in inertial frame

    # States
    px, py, pz = MX.sym('px'), MX.sym('py'), MX.sym('pz') # CoM position
    qw, qx, qy, qz = MX.sym('qw'), MX.sym('qx'), MX.sym('qy'), MX.sym('qz') # Orientation (Quaternion)
    vx, vy, vz = MX.sym('vx'), MX.sym('vy'), MX.sym('vz') # CoM linear velocity
    wx, wy, wz = MX.sym('wx'), MX.sym('wy'), MX.sym('wz') # Angular velocity (body frame)
    x = cs.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)

    # Control Forces at contacts
    F_c1_inertial = MX.sym('F_c1_inertial', 3)
    F_c2_inertial = MX.sym('F_c2_inertial', 3)

    # State derivatives
    xdot_sym = MX.sym('xdot', x.shape[0])

    # Parameters
    r_c1_inertial = MX.sym('r_c1_inertial', 3) # First contact point in inertial frame
    r_c2_inertial = MX.sym('r_c2_inertial', 3) # Second contact point in inertial frame

    # Extract parameters
    p = cs.vertcat(px, py, pz)
    q = cs.vertcat(qw, qx, qy, qz)
    v = cs.vertcat(vx, vy, vz)
    omega_body = cs.vertcat(wx, wy, wz)
    
    # Quaternions to rotation matrix (from body to inertial)
    R_body_to_inertial = cs.vertcat(
        cs.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
        cs.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
        cs.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
    )
    R_inertial_to_body = R_body_to_inertial.T # Transpose is inverse for rotation matrices

    r_c1_body = R_inertial_to_body @ (r_c1_inertial - p)
    r_c2_body = R_inertial_to_body @ (r_c2_inertial - p)
    F_c1_body = R_inertial_to_body @ F_c1_inertial
    F_c2_body = R_inertial_to_body @ F_c2_inertial

    sum_tau_body = cs.cross(r_c1_body, F_c1_body) + cs.cross(r_c2_body, F_c2_body)

    # Dynamics equations
    p_dot = cs.vertcat(vx, vy, vz)
    q_dot = 0.5 * quat_mul(q, cs.vertcat(0, omega_body))
    v_dot = (1/mass) * (F_c1_inertial + F_c2_inertial) + gravity
    omega_dot = I_inv @ (sum_tau_body - cs.cross(omega_body, I @ omega_body))
    xdot = cs.vertcat(p_dot, q_dot, v_dot, omega_dot)

    # Constraints Friction cone 0 >= (mu^2 * Fz^2) - (Fx^2 + Fy^2) + epsilon
    friction_cone1 = (mu**2) * F_c1_inertial[2]**2 - (F_c1_inertial[0]**2 + F_c1_inertial[1]**2) + epsilon
    friction_cone2 = (mu**2) * F_c2_inertial[2]**2 - (F_c2_inertial[0]**2 + F_c2_inertial[1]**2) + epsilon

    model = AcadosModel()
    model.name = 'srbm_robot'
    model.x = x
    model.xdot = xdot_sym
    model.u = cs.vertcat(F_c1_inertial, F_c2_inertial)
    model.f_expl_expr = xdot 
    model.f_impl_expr = xdot_sym - xdot
    model.con_h_expr = cs.vertcat(
        cs.sumsqr(q) - 1.0,  # Quaternion normalization
        friction_cone1,
        friction_cone2
    )
    model.p = cs.vertcat(r_c1_inertial, r_c2_inertial)
    
    return model


if __name__ == "__main__":
    inertia = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    rc_inertial = np.array([[0.3, 0.2, 0.0], [-0.3, 0.2, 0.0]])  # Center of mass in inertial frame
    mass = 1.0
    model = srbm_model(inertia, mass)