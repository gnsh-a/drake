"""
Collision detection, contact frame, Jacobian, and contact property combination.

Implements sphere-halfspace point contact matching Drake's exact algorithms.
"""

import numpy as np


def skew(v):
    """3-vector to 3x3 skew-symmetric matrix."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def make_from_one_vector(u, axis_index=2):
    """Construct rotation matrix with column `axis_index` = u.

    Matches Drake's RotationMatrix::MakeFromOneVector exactly.
    """
    u = u / np.linalg.norm(u)

    i = np.argmin(np.abs(u))
    j = (i + 1) % 3
    k = (j + 1) % 3

    mag = np.sqrt(1.0 - u[i] ** 2)
    r = 1.0 / mag
    s = -r * u[i]

    v = np.zeros(3)
    v[i] = 0.0
    v[j] = -r * u[k]
    v[k] = r * u[j]

    w = np.zeros(3)
    w[i] = mag
    w[j] = s * u[j]
    w[k] = s * u[k]

    R = np.zeros((3, 3))
    R[:, axis_index] = u
    R[:, (axis_index + 1) % 3] = v
    R[:, (axis_index + 2) % 3] = w
    return R


def sphere_halfspace(q, r, R_WH, p_WH):
    """Detect contact between a sphere and a half-space.

    Returns None if no contact, otherwise dict with:
        depth, p_WCa, p_WCb, nhat_BA_W, phi0
    """
    p_WS = q[4:7]
    p_HS = R_WH.T @ (p_WS - p_WH)
    depth = r - p_HS[2]

    if depth <= 0.0:
        return None

    # Drake convention: A = halfspace, B = sphere.
    # nhat_BA_W points from B (sphere) into A (halfspace).
    nhat_BA_W = -(R_WH @ np.array([0.0, 0.0, 1.0]))
    p_WCa = R_WH @ np.array([p_HS[0], p_HS[1], 0.0]) + p_WH
    p_WCb = p_WS + r * nhat_BA_W

    return {
        "depth": depth,
        "p_WCa": p_WCa,
        "p_WCb": p_WCb,
        "nhat_BA_W": nhat_BA_W,
        "phi0": -depth,
    }


def hertz_contact_point(p_WCa, p_WCb, k_A, k_B):
    """Compliance-weighted contact point (Hertz theory)."""
    denom = k_A + k_B
    if denom == 0.0:
        return 0.5 * (p_WCa + p_WCb)
    return (k_A * p_WCa + k_B * p_WCb) / denom


def estimate_contact_parameters(m_max, g, penetration_allowance):
    """Estimate point contact stiffness and dissipation.

    Matches Drake's EstimatePointContactParameters().

    Returns (geometry_k, dissipation).
    """
    combined_k = m_max * g / penetration_allowance
    omega = np.sqrt(combined_k / m_max)
    time_scale = 1.0 / omega
    geometry_k = 2.0 * combined_k
    dissipation = time_scale / penetration_allowance
    return geometry_k, dissipation


def combine_contact_properties(k_A, k_B, d_A, d_B, mu_A, mu_B,
                                tau_A=0.1, tau_B=0.1):
    """Combine contact properties for a pair of geometries.

    Returns (k, d, mu, tau).
    """
    # Stiffness: harmonic mean
    denom = k_A + k_B
    if denom == 0.0:
        k = 0.0
    elif k_A == np.inf:
        k = k_B
    elif k_B == np.inf:
        k = k_A
    else:
        k = k_A * k_B / denom

    # Hunt-Crossley dissipation: stiffness-weighted average
    if denom == 0.0:
        d = 0.0
    elif k_A == np.inf:
        d = d_B
    elif k_B == np.inf:
        d = d_A
    else:
        d = (k_B * d_A + k_A * d_B) / denom

    # Friction: harmonic mean
    mu_denom = mu_A + mu_B
    mu = 0.0 if mu_denom == 0.0 else 2.0 * mu_A * mu_B / mu_denom

    # Dissipation time constant: sum
    tau = tau_A + tau_B

    return k, d, mu, tau


def contact_frame(nhat_BA_W):
    """Build contact frame R_WC with z-axis = -nhat_BA_W."""
    return make_from_one_vector(-nhat_BA_W, axis_index=2)


def detect_and_build(q, r, R_WH, p_WH, geometry_k):
    """Detect contact and build contact frame + Jacobian.

    Returns dict with (J, phi0, p_WC, R_WC) or None if no contact.
    """
    cd = sphere_halfspace(q, r, R_WH, p_WH)
    if cd is None:
        return None

    p_WC = hertz_contact_point(cd["p_WCa"], cd["p_WCb"], geometry_k, geometry_k)
    R_WC = contact_frame(cd["nhat_BA_W"])
    J = contact_jacobian(q, p_WC, R_WC)

    return {"J": J, "phi0": cd["phi0"], "p_WC": p_WC, "R_WC": R_WC}


def contact_jacobian(q, p_WC, R_WC):
    """Contact Jacobian: J = R_WC^T @ [-skew(p_BC_W), I3].

    For body B (free body) with body A = world (anchored).
    """
    p_BC_W = p_WC - q[4:7]
    J_WBc_W = np.zeros((3, 6))
    J_WBc_W[:, :3] = -skew(p_BC_W)
    J_WBc_W[:, 3:] = np.eye(3)
    return R_WC.T @ J_WBc_W
