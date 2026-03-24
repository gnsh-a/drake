"""
Animated visualization of SAP contact simulation data.

Reads sim_data.csv and shows:
  - Top: 2D side view (x-z) with sphere rolling on incline
  - Bottom-left: |v| vs time
  - Bottom-right: |omega| vs time

Author: Ganesh Arivoli <arivoli@wisc.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import os
import glob
import re

SKIP = 10  # show every Nth frame


def _find_latest_csv():
    """Find the most recent sim CSV and extract RADIUS and INCLINE_DEG from its name."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    pattern = os.path.join(output_dir, "sim_r*_inc*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sim CSV files found in {output_dir}")
    latest = max(files, key=os.path.getmtime)
    m = re.search(r"sim_r([\d.]+)_inc([\d.]+)\.csv", os.path.basename(latest))
    if not m:
        raise ValueError(f"Cannot parse radius/incline from {latest}")
    return latest, float(m.group(1)), float(m.group(2))


CSV_FILE, RADIUS, INCLINE_DEG = _find_latest_csv()


def load_data():
    data = np.genfromtxt(CSV_FILE, delimiter=",", skip_header=1)
    t = data[:, 0]
    px, pz = data[:, 5], data[:, 7]
    wx, wy, wz = data[:, 8], data[:, 9], data[:, 10]
    vx, vy, vz = data[:, 11], data[:, 12], data[:, 13]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    return t, px, pz, speed, omega_mag


def main():
    t, px, pz, speed, omega_mag = load_data()

    # Subsample
    idx = np.arange(0, len(t), SKIP)
    t_s, px_s, pz_s = t[idx], px[idx], pz[idx]
    speed_s, omega_s = speed[idx], omega_mag[idx]

    # Incline line
    x_range = np.array([px_s.min() - 0.5, px_s.max() + 0.5])
    z_incline = -np.tan(np.radians(INCLINE_DEG)) * x_range

    # Figure layout
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_scene = fig.add_subplot(gs[0, :])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_omega = fig.add_subplot(gs[1, 1])

    # Scene setup
    ax_scene.set_xlim(x_range[0], x_range[1])
    z_min = min(pz_s.min(), z_incline.min()) - 0.3
    z_max = max(pz_s.max(), z_incline.max()) + 0.3
    ax_scene.set_ylim(z_min, z_max)
    ax_scene.set_aspect("equal")
    ax_scene.set_xlabel("x (m)")
    ax_scene.set_ylabel("z (m)")
    ax_scene.set_title("Sphere on Inclined Plane")
    ax_scene.plot(x_range, z_incline, "k-", lw=2, label="incline")

    sphere = Circle((px_s[0], pz_s[0]), RADIUS, fill=False, ec="blue", lw=2)
    ax_scene.add_patch(sphere)
    marker_line, = ax_scene.plot([], [], "b-", lw=1.5)
    time_text = ax_scene.text(0.02, 0.95, "", transform=ax_scene.transAxes,
                               fontsize=11, verticalalignment="top")

    # Precompute cumulative rotation angle from omega_mag
    dt_sub = t_s[1] - t_s[0] if len(t_s) > 1 else 0.0
    cumulative_angle = np.cumsum(omega_s) * dt_sub

    # Trajectory trace
    trace_line, = ax_scene.plot([], [], "b-", lw=0.5, alpha=0.3)

    # Velocity plot
    ax_vel.set_xlim(0, t_s[-1])
    ax_vel.set_ylim(0, speed.max() * 1.1 + 0.01)
    ax_vel.set_xlabel("t (s)")
    ax_vel.set_ylabel("|v| (m/s)")
    ax_vel.set_title("Speed")
    vel_line, = ax_vel.plot([], [], "g-", lw=1.5)

    # Angular velocity plot
    ax_omega.set_xlim(0, t_s[-1])
    ax_omega.set_ylim(0, omega_mag.max() * 1.1 + 0.01)
    ax_omega.set_xlabel("t (s)")
    ax_omega.set_ylabel("|w| (rad/s)")
    ax_omega.set_title("Angular Velocity")
    omega_line, = ax_omega.plot([], [], "r-", lw=1.5)

    def init():
        sphere.center = (px_s[0], pz_s[0])
        marker_line.set_data([], [])
        time_text.set_text("")
        trace_line.set_data([], [])
        vel_line.set_data([], [])
        omega_line.set_data([], [])
        return (sphere, marker_line, time_text, trace_line, vel_line, omega_line)

    def update(frame):
        i = frame
        cx, cz = px_s[i], pz_s[i]
        sphere.center = (cx, cz)

        # Diameter line rotating with the body
        angle = cumulative_angle[i]
        dx = RADIUS * np.cos(angle)
        dz = RADIUS * np.sin(angle)
        marker_line.set_data([cx - dx, cx + dx], [cz - dz, cz + dz])

        time_text.set_text(f"t = {t_s[i]:.3f} s")

        # Trajectory trace
        trace_line.set_data(px_s[:i+1], pz_s[:i+1])

        # Live plots
        vel_line.set_data(t_s[:i+1], speed_s[:i+1])
        omega_line.set_data(t_s[:i+1], omega_s[:i+1])

        return (sphere, marker_line, time_text, trace_line, vel_line, omega_line)

    anim = FuncAnimation(fig, update, frames=len(idx), init_func=init,
                          interval=20, blit=True, repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
