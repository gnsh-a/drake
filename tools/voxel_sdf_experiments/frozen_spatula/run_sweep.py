#!/usr/bin/env python3
"""Runs the frozen gripper-spatula resolution-scale sweep.

The default matrix matches the legacy study's three resolution scales. It
records one snapshot at the demo pose and a first-touch-referenced penetration
sweep from 0 through 6 mm for each scale. Every subprocess computes all three
representations against one shared FineTetReference and writes a three-row CSV.
"""

import argparse
import concurrent.futures
import csv
import pathlib
import subprocess
import sys


DEFAULT_SCALES = (1.0, 0.5, 0.25)
DEFAULT_MAX_PENETRATION_MM = 6.0
DEFAULT_PENETRATION_STEP_MM = 0.25


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _parse_positive_list(text: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(piece) for piece in text.split(","))
    except ValueError as error:
        raise ValueError(f"invalid {name}: {error}") from error
    if not values or not all(value > 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")
    return values


def _penetrations(maximum_mm: float, step_mm: float) -> tuple[float, ...]:
    if maximum_mm < 0.0 or step_mm <= 0.0:
        raise ValueError("penetration maximum must be nonnegative and step positive")
    count = int(maximum_mm / step_mm + 1.0e-12)
    values = [index * step_mm for index in range(count + 1)]
    if not values or abs(values[-1] - maximum_mm) > 1.0e-12:
        values.append(maximum_mm)
    return tuple(values)


def _run_one(
    binary: pathlib.Path,
    output_dir: pathlib.Path,
    pose: str,
    penetration_mm: float,
    scale: float,
    fine_hint_m: float,
    meshes: bool,
) -> tuple[pathlib.Path, str]:
    stem = (
        f"{pose}__delta_{_number_label(penetration_mm)}mm__"
        f"scale_{_number_label(scale)}"
    )
    output = output_dir / "cases" / f"{stem}.csv"
    command = [
        str(binary),
        f"--pose={pose}",
        f"--penetration={penetration_mm / 1000.0:.17g}",
        f"--resolution_scale={scale:.17g}",
        f"--fine_tet_resolution_hint={fine_hint_m:.17g}",
        f"--output={output}",
    ]
    if meshes:
        command.append(f"--mesh_output_dir={output_dir / 'meshes' / stem}")
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return output, completed.stdout.strip()


def _write_summary(output_dir: pathlib.Path, paths: list[pathlib.Path]) -> None:
    rows = []
    fieldnames = None
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise RuntimeError(f"CSV schema mismatch in {path}")
            file_rows = list(reader)
        if len(file_rows) != 3:
            raise RuntimeError(f"{path} has {len(file_rows)} rows; expected 3")
        rows.extend(file_rows)
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=_workspace_root()
        / "bazel-bin/tools/voxel_sdf_experiments/frozen_spatula/frozen_spatula",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_workspace_root()
        / "tools/voxel_sdf_experiments/out/frozen_spatula_sweep",
    )
    parser.add_argument(
        "--resolution-scales",
        default=",".join(f"{value:g}" for value in DEFAULT_SCALES),
    )
    parser.add_argument(
        "--max-penetration-mm", type=float, default=DEFAULT_MAX_PENETRATION_MM
    )
    parser.add_argument(
        "--penetration-step-mm", type=float, default=DEFAULT_PENETRATION_STEP_MM
    )
    parser.add_argument(
        "--fine-tet-resolution-hint",
        type=float,
        default=4.8828125e-6,
        help="FineTetReference hint in meters, shared by every case.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent cases; default 1 limits fine-reference peak memory.",
    )
    parser.add_argument("--meshes", action="store_true")
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    try:
        scales = _parse_positive_list(
            args.resolution_scales, "resolution scale"
        )
        penetration_mm = _penetrations(
            args.max_penetration_mm, args.penetration_step_mm
        )
    except ValueError as error:
        parser.error(str(error))
    if args.fine_tet_resolution_hint <= 0.0:
        parser.error("--fine-tet-resolution-hint must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    (args.output_dir / "cases").mkdir(parents=True, exist_ok=True)

    runs = [
        ("demo", 0.0, scale) for scale in scales
    ] + [
        ("first_touch", delta_mm, scale)
        for scale in scales
        for delta_mm in penetration_mm
    ]
    print(
        f"Running {len(runs)} frozen cases with {args.jobs} workers; "
        f"each case contains all three representations"
    )
    outputs = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_run = {
            pool.submit(
                _run_one,
                binary,
                args.output_dir,
                pose,
                delta_mm,
                scale,
                args.fine_tet_resolution_hint,
                args.meshes,
            ): (pose, delta_mm, scale)
            for pose, delta_mm, scale in runs
        }
        for future in concurrent.futures.as_completed(future_to_run):
            pose, delta_mm, scale = future_to_run[future]
            try:
                output, message = future.result()
                outputs.append(output)
                print(
                    f"PASS {pose:11s} delta={delta_mm:5g} mm "
                    f"scale={scale:g} -> {output.name}"
                )
                if message:
                    print(f"  {message}")
            except subprocess.CalledProcessError as error:
                failures.append((pose, delta_mm, scale))
                print(error.stdout, file=sys.stderr)
    if failures:
        print(f"{len(failures)} of {len(runs)} cases failed", file=sys.stderr)
        return 1
    _write_summary(args.output_dir, outputs)
    print(f"Completed {len(runs)} cases; wrote {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
