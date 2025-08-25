import casadi as cs

def quat_mul(q: cs.MX, r: cs.MX) -> cs.DM:
    """
    Hamilton product of two quaternions q and r.
    q, r: 4x1 CasADi MX vectors [w, x, y, z]
    returns: 4x1 quaternion result of q * r
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    return cs.vertcat(
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw
    )