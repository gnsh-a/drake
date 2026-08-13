#!/usr/bin/env python3
"""Runs the free-disk grid ladder serially and writes trajectory CSVs.

This driver is only convenience infrastructure for the demo. It intentionally
does not analyze convergence or generate a report. Voxel grids are sized by the
0.2 m box, reach roughly 40 GB at the 0.15625 mm cap, and therefore must never
run concurrently.
"""

import argparse
import pathlib
import subprocess
import sys
import time


REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
DEFAULT_RUNGS_MM = (5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625)
FINEST_ALLOWED_MM = 0.15625


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _rung_label(h_mm: float) -> str:
    return f"{h_mm:g}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=_workspace_root()
        / "bazel-bin/tools/voxel_sdf_experiments/disk_farkas/disk_farkas",
        help="Path to the already-built disk_farkas binary.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_workspace_root()
        / "tools/voxel_sdf_experiments/out/disk_farkas_ladder",
    )
    parser.add_argument(
        "--rungs_mm",
        default=",".join(f"{value:g}" for value in DEFAULT_RUNGS_MM),
        help="Comma-separated grid sizes in mm. Values finer than 0.15625 mm "
        "are rejected because the box-sized voxel grids exceed the safe cap.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    try:
        rungs = tuple(float(value) for value in args.rungs_mm.split(","))
    except ValueError as error:
        parser.error(f"invalid --rungs_mm: {error}")
    if not rungs or not all(value > 0.0 for value in rungs):
        parser.error("--rungs_mm values must be positive")
    if min(rungs) < FINEST_ALLOWED_MM:
        parser.error(
            f"finest rung {min(rungs):g} mm exceeds the "
            f"{FINEST_ALLOWED_MM:g} mm safety cap"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        (representation, h_mm)
        for representation in args.representations
        for h_mm in rungs
    ]
    print(
        f"Running {len(runs)} disk simulations SERIALly; "
        f"rungs={rungs} mm, output={args.output_dir}"
    )
    for index, (representation, h_mm) in enumerate(runs, start=1):
        stem = f"{representation}__h_{_rung_label(h_mm)}mm"
        output = args.output_dir / f"{stem}.csv"
        command = [
            str(binary),
            f"--representation={representation}",
            f"--resolution={h_mm / 1000.0:.17g}",
            f"--output={output}",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(
            f"PASS {index:2d}/{len(runs)} {representation:14s} "
            f"h={h_mm:8g} mm {time.monotonic() - started:8.1f} s "
            f"-> {output.name}"
        )
        if completed.stdout.strip():
            print(f"  {completed.stdout.strip()}")
    print(f"Completed all {len(runs)} serial runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
