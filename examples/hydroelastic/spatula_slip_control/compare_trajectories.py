"""Runs and plots coarse/fine/finer tet and fine/finer voxel trajectories."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spatula-slip-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402, I001


_CSV_FIELDS = (
    "representation",
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "left_finger_m",
    "right_finger_m",
    "spatula_contacts",
    "left_contact_force_x_N",
    "left_contact_force_y_N",
    "left_contact_force_z_N",
    "left_handle_axis_torque_Nm",
    "right_contact_force_x_N",
    "right_contact_force_y_N",
    "right_contact_force_z_N",
    "right_handle_axis_torque_Nm",
)
_HANDLE_AXIS_B = (0.0, -math.sin(2.0), math.cos(2.0))
_TARGET = "//examples/hydroelastic/spatula_slip_control:spatula_slip_control"
_BINARY = (
    "bazel-bin/examples/hydroelastic/spatula_slip_control/spatula_slip_control"
)
_DEFAULT_OUTPUT = "examples/hydroelastic/spatula_slip_control/out"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-sec", type=float, default=30.0)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(_DEFAULT_OUTPUT)
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing CSVs instead of building and running simulations.",
    )
    args = parser.parse_args()
    if args.simulation_sec < 0.0:
        parser.error("--simulation-sec must be non-negative")
    if args.time_step <= 0.0:
        parser.error("--time-step must be positive")
    steps = round(args.simulation_sec / args.time_step)
    if not math.isclose(
        steps * args.time_step,
        args.simulation_sec,
        rel_tol=0.0,
        abs_tol=1.0e-12 * max(1.0, args.simulation_sec),
    ):
        parser.error("--simulation-sec must be an integer number of time steps")
    return args


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _absolute(path, root):
    return path if path.is_absolute() else root / path


def _build(root):
    subprocess.run(
        [
            "bazel",
            "--batch",
            "--output_user_root=/tmp/bazel-output",
            "build",
            "--experimental_collect_system_network_usage=false",
            _TARGET,
        ],
        cwd=root,
        check=True,
    )
    binary = root / _BINARY
    if not binary.is_file():
        raise RuntimeError(f"Built binary is missing: {binary}")
    return binary


def _run_case(
    args,
    root,
    binary,
    representation,
    csv_path,
    *,
    label,
    tet_resolution_hint_scale=1.0,
    voxel_width_scale=1.0,
):
    command = [
        str(binary),
        f"--hydroelastic_representation={representation}",
        f"--tet_resolution_hint_scale={tet_resolution_hint_scale}",
        f"--voxel_sdf_width_scale={voxel_width_scale}",
        f"--simulation_sec={args.simulation_sec}",
        f"--mbp_discrete_update_period={args.time_step}",
        "--gripper_force=1.5",
        "--amplitude=5.0",
        "--duty_cycle=0.5",
        "--period=3.0",
        "--stiction_tolerance=1e-4",
        "--contact_model=hydroelastic",
        "--contact_surface_representation=polygon",
        "--contact_approximation=lagged",
        "--realtime_rate=0.0",
        "--visualize=false",
        f"--trajectory_csv={csv_path}",
    ]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            f"{label} simulation failed with code {result.returncode}"
        )
    print(f"{label}: {csv_path}")


def _read_csv(path, expected_representation):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
            raise RuntimeError(f"{path} has an unexpected CSV header")
        rows = []
        for raw in reader:
            if raw["representation"] != expected_representation:
                raise RuntimeError(
                    f"{path} contains representation {raw['representation']!r}"
                )
            row = {"representation": raw["representation"]}
            for field in _CSV_FIELDS[1:]:
                value = float(raw[field])
                if not math.isfinite(value):
                    raise RuntimeError(f"{path} contains non-finite {field}")
                row[field] = value
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{path} contains no samples")
    return rows


def _validate_alignment(args, cases):
    expected_samples = round(args.simulation_sec / args.time_step) + 1
    for label, rows in cases.items():
        if len(rows) != expected_samples:
            raise RuntimeError(
                f"{label} has {len(rows)} samples; expected {expected_samples}"
            )
        for index, row in enumerate(rows):
            expected_time = index * args.time_step
            if not math.isclose(
                row["time_s"],
                expected_time,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise RuntimeError(
                    f"{label} sample {index} is at {row['time_s']}, "
                    f"expected {expected_time}"
                )


def _quaternion(row):
    q = (row["qw"], row["qx"], row["qy"], row["qz"])
    norm = math.sqrt(sum(value * value for value in q))
    if norm == 0.0:
        raise RuntimeError("trajectory contains a zero quaternion")
    return tuple(value / norm for value in q)


def _multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def _rotate(q, vector):
    rotated = _multiply(_multiply(q, (0.0, *vector)), _conjugate(q))
    return rotated[1:]


def _handle_axis_swing(rows):
    initial_axis = _rotate(_quaternion(rows[0]), _HANDLE_AXIS_B)
    result = []
    for row in rows:
        current_axis = _rotate(_quaternion(row), _HANDLE_AXIS_B)
        cosine = sum(
            a * b for a, b in zip(initial_axis, current_axis, strict=True)
        )
        result.append(math.acos(max(-1.0, min(1.0, cosine))))
    return result


def _force_norm(row, side):
    return math.sqrt(
        row[f"{side}_contact_force_x_N"] ** 2
        + row[f"{side}_contact_force_y_N"] ** 2
        + row[f"{side}_contact_force_z_N"] ** 2
    )


def _plot_finger_force(axis, time, cases, side):
    for label, rows, linestyle in cases:
        axis.plot(
            time,
            [_force_norm(row, side) for row in rows],
            linestyle=linestyle,
            label=label,
        )
    axis.set_ylabel("force [N]")
    axis.set_title(f"{side.capitalize()}-finger contact force")
    axis.legend()


def _plot_finger_torque(axis, time, cases, side):
    torque_field = f"{side}_handle_axis_torque_Nm"
    for label, rows, linestyle in cases:
        axis.plot(
            time,
            [row[torque_field] for row in rows],
            linestyle=linestyle,
            label=label,
        )
    axis.set_ylabel("handle-axis torque [N m]")
    axis.set_title(f"{side.capitalize()}-finger handle-axis torque")
    axis.legend()


def _plot(
    args,
    tet_rows,
    tet_fine_rows,
    tet_finer_rows,
    fine_rows,
    finer_rows,
    output_path,
):
    time = [row["time_s"] for row in tet_rows]
    cases = (
        ("tet", tet_rows, "-"),
        ("tet fine", tet_fine_rows, "-."),
        ("tet finer", tet_finer_rows, (0, (5, 1))),
        ("voxel fine", fine_rows, "--"),
        ("voxel finer", finer_rows, ":"),
    )

    figure = plt.figure(figsize=(15, 14))
    grid = figure.add_gridspec(4, 2)
    position_axis = figure.add_subplot(grid[0, 0])
    swing_axis = figure.add_subplot(grid[0, 1])
    finger_axis = figure.add_subplot(grid[1, :])
    left_force_axis = figure.add_subplot(grid[2, 0])
    right_force_axis = figure.add_subplot(grid[2, 1])
    left_torque_axis = figure.add_subplot(grid[3, 0])
    right_torque_axis = figure.add_subplot(grid[3, 1])
    axes = (
        position_axis,
        swing_axis,
        finger_axis,
        left_force_axis,
        right_force_axis,
        left_torque_axis,
        right_torque_axis,
    )

    colors = {"x_m": "tab:red", "y_m": "tab:green", "z_m": "tab:blue"}
    for field, color in colors.items():
        coordinate = field[0]
        for label, rows, linestyle in cases:
            position_axis.plot(
                time,
                [row[field] for row in rows],
                color=color,
                linestyle=linestyle,
                label=f"{label} {coordinate}",
            )
    position_axis.set_ylabel("position [m]")
    position_axis.set_title("Spatula position")
    position_axis.legend(ncol=3, fontsize="small")

    for label, rows, linestyle in cases:
        swing_axis.plot(
            time,
            _handle_axis_swing(rows),
            linestyle=linestyle,
            label=label,
        )
    swing_axis.set_ylabel("handle-axis swing [rad]")
    swing_axis.set_title("Spatula handle-axis swing")
    swing_axis.legend()

    for field, color, side in (
        ("left_finger_m", "tab:purple", "left"),
        ("right_finger_m", "tab:orange", "right"),
    ):
        for label, rows, linestyle in cases:
            finger_axis.plot(
                time,
                [row[field] for row in rows],
                color=color,
                linestyle=linestyle,
                label=f"{label} {side}",
            )
    finger_axis.set_ylabel("finger position [m]")
    finger_axis.set_title("Finger positions")
    finger_axis.legend(ncol=5)

    _plot_finger_force(left_force_axis, time, cases, "left")
    _plot_finger_force(right_force_axis, time, cases, "right")
    _plot_finger_torque(left_torque_axis, time, cases, "left")
    _plot_finger_torque(right_torque_axis, time, cases, "right")
    left_torque_axis.set_xlabel("time [s]")
    right_torque_axis.set_xlabel("time [s]")

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Spatula coarse/fine/finer tet vs fine/finer voxel-SDF trajectory\n"
        f"duration={args.simulation_sec:g} s, dt={args.time_step:g} s"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = _parse_args()
    root = _repo_root()
    output_dir = _absolute(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tet_csv = output_dir / "spatula_tet_trajectory.csv"
    tet_fine_csv = output_dir / "spatula_tet_fine_trajectory.csv"
    tet_finer_csv = output_dir / "spatula_tet_finer_trajectory.csv"
    fine_csv = output_dir / "spatula_voxel_sdf_fine_trajectory.csv"
    finer_csv = output_dir / "spatula_voxel_sdf_finer_trajectory.csv"
    plot_path = output_dir / "spatula_trajectory_comparison.png"

    if not args.plot_only:
        binary = _build(root)
        _run_case(args, root, binary, "tet", tet_csv, label="tet")
        _run_case(
            args,
            root,
            binary,
            "tet",
            tet_fine_csv,
            label="tet fine",
            tet_resolution_hint_scale=0.5,
        )
        _run_case(
            args,
            root,
            binary,
            "tet",
            tet_finer_csv,
            label="tet finer",
            tet_resolution_hint_scale=0.25,
        )
        _run_case(
            args,
            root,
            binary,
            "voxel_sdf",
            fine_csv,
            label="voxel_sdf fine",
            voxel_width_scale=0.5,
        )
        _run_case(
            args,
            root,
            binary,
            "voxel_sdf",
            finer_csv,
            label="voxel_sdf finer",
            voxel_width_scale=0.25,
        )

    tet_rows = _read_csv(tet_csv, "tet")
    tet_fine_rows = _read_csv(tet_fine_csv, "tet")
    tet_finer_rows = _read_csv(tet_finer_csv, "tet")
    fine_rows = _read_csv(fine_csv, "voxel_sdf")
    finer_rows = _read_csv(finer_csv, "voxel_sdf")
    _validate_alignment(
        args,
        {
            "tet": tet_rows,
            "tet fine": tet_fine_rows,
            "tet finer": tet_finer_rows,
            "voxel_sdf fine": fine_rows,
            "voxel_sdf finer": finer_rows,
        },
    )
    _plot(
        args,
        tet_rows,
        tet_fine_rows,
        tet_finer_rows,
        fine_rows,
        finer_rows,
        plot_path,
    )
    print(f"samples per representation: {len(tet_rows)}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
