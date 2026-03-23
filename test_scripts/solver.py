"""
SAP (Semi-Analytic Primal) contact solver.

Implements the friction cone constraint with Newton iteration and exact line
search, matching Drake's SapSolver + SapFrictionConeConstraint.

Reference: Castro et al., 2021. "An unconstrained convex formulation of
compliant contact." IEEE T-RO, 2022.
"""

import numpy as np

# Contact modes
MODE_STICTION = 0
MODE_SLIDING = 1
MODE_NO_CONTACT = 2


def soft_norm(x, eps=1e-7):
    """Soft norm: sqrt(||x||^2 + eps^2)."""
    return np.sqrt(np.dot(x, x) + eps * eps)


def setup_regularization(J, A, k, tau, dt, mu, phi0, beta=1.0, sigma=1e-3):
    """Compute all SAP regularization parameters and bias.

    Merges delassus_diagonal + calc_regularization + calc_bias.

    Returns (R, v_hat, mu_hat, mu_tilde).
        R: 3-vector [Rt, Rt, Rn]
        v_hat: 3-vector bias [0, 0, -phi0/(dt+tau)]
        mu_hat, mu_tilde: derived friction parameters
    """
    # Delassus diagonal: ||W||_F / ni (Drake's exact formula)
    A_diag_inv = 1.0 / np.diag(A)
    W = J @ np.diag(A_diag_inv) @ J.T
    ni = J.shape[0]
    w = np.linalg.norm(W, 'fro') / ni

    # Regularization
    beta_factor = beta * beta / (4.0 * np.pi * np.pi)
    Rn = max(beta_factor * w, 1.0 / (dt * k * (dt + tau)))
    Rt = sigma * w

    R = np.array([Rt, Rt, Rn])
    v_hat = np.array([0.0, 0.0, -phi0 / (dt + tau)])
    mu_hat = mu * Rt / Rn
    mu_tilde = mu * np.sqrt(Rt / Rn)

    return R, v_hat, mu_hat, mu_tilde


def project_impulse(vc, v_hat, R, mu, mu_hat, mu_tilde):
    """Project constraint velocity to impulse via SAP friction cone.

    Returns (gamma, mode, yt, yn, yr).
    """
    y = (v_hat - vc) / R
    yt = y[:2]
    yn = y[2]
    yr = soft_norm(yt)

    if yr <= mu * yn:
        return y.copy(), MODE_STICTION, yt, yn, yr

    if -mu_hat * yr < yn < yr / mu:
        gn = (yn + mu_hat * yr) / (1.0 + mu_tilde * mu_tilde)
        gt = mu * gn * yt / yr
        return np.array([gt[0], gt[1], gn]), MODE_SLIDING, yt, yn, yr

    return np.zeros(3), MODE_NO_CONTACT, yt, yn, yr


def constraint_hessian(mode, yt, yn, yr, gn, R, mu, mu_hat, mu_tilde):
    """Constraint Hessian G (3x3): dP/dy @ R^{-1}."""
    if mode == MODE_NO_CONTACT:
        return np.zeros((3, 3))

    R_inv = np.diag(1.0 / R)

    if mode == MODE_STICTION:
        return R_inv.copy()

    # Sliding mode
    t_hat = yt / yr
    P = np.outer(t_hat, t_hat)
    Pperp = np.eye(2) - P
    factor = 1.0 / (1.0 + mu_tilde * mu_tilde)

    dP_dy = np.zeros((3, 3))
    dP_dy[:2, :2] = mu * (gn / yr * Pperp + mu_hat * factor * P)
    dP_dy[0, 2] = mu * factor * t_hat[0]
    dP_dy[1, 2] = mu * factor * t_hat[1]
    dP_dy[2, 0] = mu_hat * factor * t_hat[0]
    dP_dy[2, 1] = mu_hat * factor * t_hat[1]
    dP_dy[2, 2] = factor

    return dP_dy @ R_inv


def convergence_check(grad, A, v, J, gamma, abs_tol=1e-14, rel_tol=1e-6):
    """Check SAP convergence using scaled momentum residual."""
    D_tilde = 1.0 / np.sqrt(np.diag(A))
    residual = np.linalg.norm(D_tilde * grad)
    scale = max(np.linalg.norm(D_tilde * (A @ v)),
                np.linalg.norm(D_tilde * (J.T @ gamma)))
    return residual <= abs_tol + rel_tol * scale


def _eval_line_search(alpha, v, dv, v_star, A, J, v_hat, R, mu, mu_hat,
                       mu_tilde, dp, dvc):
    """Evaluate dl/dalpha and d2l/dalpha2 at given alpha."""
    v_alpha = v + alpha * dv
    vc_alpha = J @ v_alpha

    gamma_alpha, mode, yt, yn, yr = project_impulse(
        vc_alpha, v_hat, R, mu, mu_hat, mu_tilde)

    G_alpha = constraint_hessian(mode, yt, yn, yr, gamma_alpha[2],
                                  R, mu, mu_hat, mu_tilde)

    dl = dp @ (v_alpha - v_star) - dvc @ gamma_alpha
    d2l = dp @ dv + dvc @ G_alpha @ dvc

    return dl, d2l


def exact_line_search(v, dv, v_star, A, J, v_hat, R, mu, mu_hat, mu_tilde,
                       alpha_max=1.5, max_iter=100, tol=1e-8):
    """Exact line search: find alpha minimizing l(v + alpha * dv).

    Newton's method with bisection fallback on dl/dalpha = 0.
    """
    dp = A @ dv
    dvc = J @ dv

    dl0, _ = _eval_line_search(
        0.0, v, dv, v_star, A, J, v_hat, R, mu, mu_hat, mu_tilde, dp, dvc)

    if dl0 >= 0.0:
        return 0.0

    scaled_tol = tol * abs(dl0)
    a_lo, a_hi = 0.0, alpha_max
    alpha = 0.5 * alpha_max

    for _ in range(max_iter):
        dl, d2l = _eval_line_search(
            alpha, v, dv, v_star, A, J, v_hat, R, mu, mu_hat, mu_tilde,
            dp, dvc)

        if abs(dl) < scaled_tol:
            return alpha

        if dl < 0.0:
            a_lo = alpha
        else:
            a_hi = alpha

        if d2l > 0.0:
            alpha_newton = alpha - dl / d2l
            if a_lo < alpha_newton < a_hi:
                alpha = alpha_newton
                continue

        alpha = 0.5 * (a_lo + a_hi)

    return alpha


def sap_solve(A, v_star, J, phi0, k, tau, mu, v0, dt,
              beta=1.0, sigma=1e-3, max_iter=100):
    """Full SAP solver: Newton iteration with exact line search.

    Returns v_next (6-vector solved velocity).
    """
    R, v_hat, mu_hat, mu_tilde = setup_regularization(
        J, A, k, tau, dt, mu, phi0, beta, sigma)

    v = v0.copy()

    for iteration in range(max_iter):
        vc = J @ v
        gamma, mode, yt, yn, yr = project_impulse(
            vc, v_hat, R, mu, mu_hat, mu_tilde)

        grad = A @ (v - v_star) - J.T @ gamma

        if convergence_check(grad, A, v, J, gamma):
            break

        G = constraint_hessian(mode, yt, yn, yr, gamma[2],
                                R, mu, mu_hat, mu_tilde)
        H = A + J.T @ G @ J
        dv = -np.linalg.solve(H, grad)

        alpha = exact_line_search(v, dv, v_star, A, J, v_hat, R, mu,
                                   mu_hat, mu_tilde)
        v = v + alpha * dv

    return v
