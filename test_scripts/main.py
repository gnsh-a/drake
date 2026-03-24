"""
Standalone SAP contact simulation: sphere on inclined plane.

Matches Drake's inclined_plane_with_body demo with identical physics.
No Drake dependency at runtime.

Author: Ganesh Arivoli <arivoli@wisc.edu>
"""

import csv
import numpy as np
import dynamics
import contact
import solver


# --- Scene Parameters (matching Drake demo) ---
MASS = 0.1              # kg
RADIUS = 0.25           # m
GRAVITY = 9.8           # m/s^2
INCLINE_DEG = 15.0      # degrees
MU_STATIC = 0.3
MU_KINETIC = 0.3 # not used
DT = 1.0e-3             # s
PENETRATION_ALLOWANCE = 1.0e-5  # m
SIM_TIME = 2.0          # s
import os
CSV_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output",
    f"sim_r{RADIUS}_inc{INCLINE_DEG}.csv")


def run():
    theta = np.radians(INCLINE_DEG)

    # Body-frame inertia (solid sphere, constant)
    I_B = (2.0 / 5.0 * MASS * RADIUS**2) * np.eye(3)

    # Inclined plane pose (half-space, rotation about Y)
    R_WH = np.array([
        [ np.cos(theta), 0.0, np.sin(theta)],
        [           0.0, 1.0,           0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    p_WH = np.zeros(3)

    # Contact properties (computed once)
    geometry_k, dissipation = contact.estimate_contact_parameters(
        MASS, GRAVITY, PENETRATION_ALLOWANCE)
    k_comb, d_comb, mu_comb, tau_comb = contact.combine_contact_properties(
        geometry_k, geometry_k, dissipation, dissipation, MU_STATIC, MU_STATIC)

    # Initial state
    q = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.2])
    v = np.zeros(6)

    num_steps = int(SIM_TIME / DT)
    print(f"Running {num_steps} steps at dt={DT}s ...")
    print(f"Contact: k={k_comb:.0f} N/m, d={d_comb:.1f} s/m, "
          f"mu={mu_comb:.2f}, tau={tau_comb:.2f} s")
    print(f"Initial: p=({q[4]:.3f}, {q[5]:.3f}, {q[6]:.3f})\n")

    # CSV logging
    header = ["t", "qw", "qx", "qy", "qz", "px", "py", "pz",
              "wx", "wy", "wz", "vx", "vy", "vz", "contact"]
    rows = []

    for step in range(num_steps):
        # Drake uses A = M + dt*D (joint damping); D=0 for free body.
        M, v_star = dynamics.compute_dynamics(q, v, I_B, MASS, GRAVITY, DT)

        # Detect contact and assemble contact data if present
        cd = contact.detect_and_build(q, RADIUS, R_WH, p_WH, geometry_k)

        if cd is not None:
            # Solve the SAP contact problem for constrained velocity
            v_next = solver.sap_solve(M, v_star, cd["J"], cd["phi0"],
                                     k_comb, tau_comb, mu_comb, v, DT)
        else:
            # No contact: use unconstrained velocity
            v_next = v_star

        # Integrate the new state given the velocity
        q = dynamics.integrate_state(q, v_next, DT)
        v = v_next

        # Current simulation time
        t = (step + 1) * DT
        # Log state and contact status for each timestep
        rows.append([t, *q, *v, int(cd is not None)])

        # Print status every 100 steps
        if (step + 1) % 100 == 0:
            speed = np.linalg.norm(v[3:6])
            omega_mag = np.linalg.norm(v[0:3])
            print(f"t={t:.3f}s  p=({q[4]:+.4f}, {q[5]:+.4f}, {q[6]:+.4f})  "
                  f"|v|={speed:.4f} m/s  |w|={omega_mag:.2f} rad/s  "
                  f"contact={cd is not None}")

    # Write CSV
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} steps to {CSV_FILE}")
    print(f"Final: p=({q[4]:.6f}, {q[5]:.6f}, {q[6]:.6f})")
    print(f"Final: v=({v[3]:.6f}, {v[4]:.6f}, {v[5]:.6f})")
    print(f"Run: python3 visualize.py")


if __name__ == "__main__":
    run()
