"""Runs the tet and affine-SDF disk simulation sweep and writes CSV output.

Each case writes a trajectory CSV plus companion <stem>_sap_stats.csv and
<stem>_tamsi_stats.csv solver-statistics files. Analyze the CSVs directly; no
report is generated.
"""

import argparse
from pathlib import Path
import subprocess

_DEFAULT_VOXEL_SIZES = (0.0025, 0.00125, 0.000625)


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
):
    stem = _case_stem(args, representation, voxel_size, time_step)
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
        "--surface-output",
        str(output_dir / f"{stem}_surface.vtk"),
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--time-step",
        str(time_step),
        "--contact-approximation",
        args.contact_approximation,
        "--quiet",
    ]
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
    written = [
        _run_case(
            args, root, output_dir, binary, "tet", coarse_size, args.time_step
        )
    ]
    for voxel_size in args.voxel_sizes:
        written.append(
            _run_case(
                args,
                root,
                output_dir,
                binary,
                "voxel_sdf",
                voxel_size,
                args.time_step,
            )
        )
    written.append(
        _run_case(
            args,
            root,
            output_dir,
            binary,
            "voxel_sdf",
            fine_size,
            args.time_step / 2.0,
        )
    )

    print(f"Wrote {len(written)} trajectory CSVs to {output_dir}:")
    for csv_path in written:
        print(f"  {csv_path.name} (+ _sap_stats.csv, _tamsi_stats.csv)")
    surface_name = _case_stem(args, "voxel_sdf", fine_size, args.time_step)
    print(
        "Surface viewer: uv run --with pyvista python "
        "tools/sdf_disk_farkas/view_surface.py --surface "
        f"{output_dir / (surface_name + '_surface.vtk')}"
    )


if __name__ == "__main__":
    main()
