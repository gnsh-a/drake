"""
Rigid body dynamics for a single free-floating body with quaternion coordinates.

State conventions (matching Drake's QuaternionFloatingMobilizer):
  q = [qw, qx, qy, qz, px, py, pz]   (7 positions)
  v = [omega_x, omega_y, omega_z, vx, vy, vz]  (6 velocities)

Quaternion is scalar-first: [w, x, y, z].
Angular velocity is in world frame.
"""

import numpy as np


def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def skew(v):
    """3-vector to 3x3 skew-symmetric matrix."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def mass_matrix(q, I_B, m):
    """6x6 mass matrix in world frame for a body with COM at origin."""
    R = quat_to_rotmat(q[:4])
    I_W = R @ I_B @ R.T
    M = np.zeros((6, 6))
    M[:3, :3] = I_W
    M[3:, 3:] = m * np.eye(3)
    return M


def bias_forces(q, v, I_B):
    """Gyroscopic bias term C(q,v)*v for a single free body."""
    R = quat_to_rotmat(q[:4])
    I_W = R @ I_B @ R.T
    omega = v[:3]
    return np.concatenate([np.cross(omega, I_W @ omega), np.zeros(3)])


def gravity_forces(m, g):
    """Generalized gravity forces for a free body with COM at origin."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, -m * g])


def free_motion_velocity(A, v, tau_g, k_bias, dt):
    """Free-motion velocity: v_star = v + dt * A^{-1} @ (tau_g - k_bias)."""
    return v + dt * np.linalg.solve(A, tau_g - k_bias)


def compute_dynamics(q, v, I_B, m, g, dt):
    """Compute mass matrix and free-motion velocity for one step.

    Returns (M, v_star).
    """
    M = mass_matrix(q, I_B, m)
    k_bias = bias_forces(q, v, I_B)
    tau_g = gravity_forces(m, g)
    v_star = free_motion_velocity(M, v, tau_g, k_bias, dt)
    return M, v_star


def integrate_state(q, v_next, dt):
    """Integrate positions forward by dt using v_next.

    Uses Drake's exact MapVelocityToQDot via Q matrix.
    No quaternion renormalization (matches Drake).
    """
    qw, qx, qy, qz = q[:4]
    Q = np.array([[-qx, -qy, -qz],
                  [ qw,  qz, -qy],
                  [-qz,  qw,  qx],
                  [ qy, -qx,  qw]])

    q_next = np.empty(7)
    q_next[:4] = q[:4] + dt * 0.5 * Q @ v_next[:3]
    q_next[4:] = q[4:] + dt * v_next[3:6]
    return q_next
