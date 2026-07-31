"""Runs the tet vs affine-SDF disk comparison and writes an HTML report.

Runs the production tet/affine sweep (for SAP-iteration and trajectory-error
comparison) plus a dedicated affine-SDF convergence sweep (grid and
time-step refinement), then renders tools/sdf_disk_farkas/generate_report.py
into a single HTML report. Each simulation case also writes companion
<stem>_sap_stats.csv and <stem>_tamsi_stats.csv solver-statistics files.
"""

import argparse
from pathlib import Path
import subprocess
import sys

_DEFAULT_VOXEL_SIZES = (0.0025, 0.00125, 0.000625)

# Convergence sweep: a fixed 0.2 s window (the disk's full slide+spin-to-rest
# motion completes by ~0.125 s) sampled at 2 kHz, independent of the
# production sweep's dt=1 ms / 3 s settings.
_CONV_NUM_FRAMES = 400
_CONV_FPS = 2000.0
_CONV_GRID_SIZES = (0.005, 0.0025, 0.00125, 0.000625, 0.0003125)
_CONV_GRID_TIME_STEP = 1.25e-4
_CONV_DT_GRID_SIZE = 0.0025
_CONV_TIME_STEPS = (2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/hydro_compare/out/sdf_disk_farkas"),
    )
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument(
        "--voxel-sizes",
        type=float,
        nargs="+",
        default=_DEFAULT_VOXEL_SIZES,
        help="target voxel sizes in meters, ordered from coarse to fine",
    )
    parser.add_argument(
        "--contact-approximation", choices=("tamsi", "lagged"), default="lagged"
    )
    args = parser.parse_args()
    if args.time_step <= 0.0:
        parser.error("--time-step must be positive")
    if len(args.voxel_sizes) < 3:
        parser.error("--voxel-sizes requires at least three resolutions")
    if any(value <= 0.0 for value in args.voxel_sizes):
        parser.error("--voxel-sizes entries must be positive")
    if any(
        coarse <= fine
        for coarse, fine in zip(args.voxel_sizes, args.voxel_sizes[1:])
    ):
        parser.error("--voxel-sizes must be strictly coarse-to-fine")
    return args


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _absolute(path, root):
    return path if path.is_absolute() else root / path


def _size_slug(value, scale, suffix):
    number = f"{value * scale:g}".replace(".", "p")
    return f"{number}{suffix}"


def _case_stem(args, representation, voxel_size, time_step):
    h_slug = _size_slug(voxel_size, 1e3, "mm")
    dt_slug = _size_slug(time_step, 1e3, "ms")
    return (
        f"{args.contact_approximation}_{representation}_h{h_slug}_dt{dt_slug}"
    )


def _conv_grid_stem(args, voxel_size):
    h_slug = _size_slug(voxel_size, 1e3, "mm")
    return f"conv_grid_{args.contact_approximation}_h{h_slug}"


def _conv_dt_stem(args, time_step):
    dt_slug = _size_slug(time_step, 1e3, "ms")
    return f"conv_dt_{args.contact_approximation}_dt{dt_slug}"


def _build_runner(root):
    command = [
        "bazel",
        "--batch",
        "--output_user_root=/tmp/bazel-output",
        "build",
        "--experimental_collect_system_network_usage=false",
        "//tools/sdf_disk_farkas:disk_contact_reference",
    ]
    subprocess.run(command, cwd=root, check=True)
    return root / "bazel-bin/tools/sdf_disk_farkas/disk_contact_reference"


def _run_case(
    args,
    root,
    output_dir,
    binary,
    representation,
    voxel_size,
    time_step,
    stem,
    num_frames,
    fps,
    write_surface=False,
):
    command = [
        str(binary),
        "--scene",
        "tools/sdf_disk_farkas/disk_plane.yaml",
        "--representation",
        representation,
        "--sdf-target-voxel-size",
        str(voxel_size),
        "--output",
        str(output_dir / f"{stem}.csv"),
        "--num-frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--time-step",
        str(time_step),
        "--contact-approximation",
        args.contact_approximation,
        "--quiet",
    ]
    if write_surface:
        command.extend(
            ["--surface-output", str(output_dir / f"{stem}_surface.vtk")]
        )
    subprocess.run(command, cwd=root, check=True)
    return output_dir / f"{stem}.csv"


def main():
    args = _parse_args()
    root = _repo_root()
    output_dir = _absolute(args.output_dir, root)
    # Deliberately do not clear this directory; it may contain other local
    # experiment artifacts that must survive a rerun.
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_size = args.voxel_sizes[0]
    fine_size = args.voxel_sizes[-1]

    binary = _build_runner(root)

    tet_csv = _run_case(
        args,
        root,
        output_dir,
        binary,
        "tet",
        coarse_size,
        args.time_step,
        _case_stem(args, "tet", coarse_size, args.time_step),
        args.num_frames,
        args.fps,
    )
    affine_runs = [
        (
            voxel_size,
            args.time_step,
            _run_case(
                args,
                root,
                output_dir,
                binary,
                "voxel_sdf",
                voxel_size,
                args.time_step,
                _case_stem(args, "voxel_sdf", voxel_size, args.time_step),
                args.num_frames,
                args.fps,
                write_surface=(voxel_size == fine_size),
            ),
        )
        for voxel_size in args.voxel_sizes
    ]

    print("Running affine-SDF convergence sweep...")
    conv_grid_runs = [
        (
            voxel_size,
            _run_case(
                args,
                root,
                output_dir,
                binary,
                "voxel_sdf",
                voxel_size,
                _CONV_GRID_TIME_STEP,
                _conv_grid_stem(args, voxel_size),
                _CONV_NUM_FRAMES,
                _CONV_FPS,
            ),
        )
        for voxel_size in _CONV_GRID_SIZES
    ]
    conv_dt_runs = [
        (
            time_step,
            _run_case(
                args,
                root,
                output_dir,
                binary,
                "voxel_sdf",
                _CONV_DT_GRID_SIZE,
                time_step,
                _conv_dt_stem(args, time_step),
                _CONV_NUM_FRAMES,
                _CONV_FPS,
            ),
        )
        for time_step in _CONV_TIME_STEPS
    ]

    report_path = output_dir / "disk_farkas_affine_sdf.html"
    report_command = [
        sys.executable,
        str(root / "tools/sdf_disk_farkas/generate_report.py"),
        "--tet-csv",
        str(tet_csv),
        "--tet-target-voxel-size",
        str(coarse_size),
        "--tet-time-step",
        str(args.time_step),
        "--output",
        str(report_path),
    ]
    for voxel_size, time_step, csv_path in affine_runs:
        report_command.extend(
            ["--affine-run", f"{voxel_size},{time_step},{csv_path}"]
        )
    for voxel_size, csv_path in conv_grid_runs:
        report_command.extend(["--conv-grid-run", f"{voxel_size},{csv_path}"])
    for time_step, csv_path in conv_dt_runs:
        report_command.extend(["--conv-dt-run", f"{time_step},{csv_path}"])
    subprocess.run(report_command, cwd=root, check=True)

    print(f"Report: {report_path}")
    surface_name = _case_stem(args, "voxel_sdf", fine_size, args.time_step)
    print(
        "Surface viewer: uv run --with pyvista python "
        "tools/sdf_disk_farkas/view_surface.py --surface "
        f"{output_dir / (surface_name + '_surface.vtk')}"
    )


if __name__ == "__main__":
    main()
