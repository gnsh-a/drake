#!/usr/bin/env python3
"""Runs the Phase 1 frozen contact-surface convergence ladder."""

import argparse
import concurrent.futures
import pathlib
import subprocess
import sys


SCENES = ("sphere_sphere", "sphere_box")
REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
ALL_RUNGS_MM = (10.0, 5.0, 2.5, 1.25)


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _rung_label(h_mm: float) -> str:
    return f"{h_mm:g}".replace(".", "p")


def _run_one(
    binary: pathlib.Path,
    output_dir: pathlib.Path,
    scene: str,
    representation: str,
    h_mm: float,
    meshes: bool,
) -> tuple[pathlib.Path, str]:
    h_m = h_mm / 1000.0
    stem = f"{scene}__{representation}__h_{_rung_label(h_mm)}mm"
    output = output_dir / f"{stem}.csv"
    command = [
        str(binary),
        f"--scene={scene}",
        f"--representation={representation}",
        "--penetration=0.0199",
        f"--voxel_width={h_m:.17g}",
        f"--tet_resolution_hint={h_m:.17g}",
        f"--output={output}",
    ]
    if meshes:
        command.append(f"--mesh_output={output_dir / 'meshes' / stem}.vtk")
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return output, completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=_workspace_root()
        / "bazel-bin/tools/voxel_sdf_experiments/frozen_surface/frozen_surface",
        help="Path to the already-built frozen_surface binary.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_workspace_root()
        / "tools/voxel_sdf_experiments/out/frozen_surface_ladder",
    )
    parser.add_argument(
        "--rungs",
        choices=("endpoints", "all"),
        default="all",
        help="Run only 10 and 1.25 mm, or the complete four-rung ladder.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Maximum concurrent frozen-query subprocesses.",
    )
    parser.add_argument(
        "--meshes",
        action="store_true",
        help="Also write each contact surface as VTK POLYDATA under "
        "OUTPUT_DIR/meshes, for ParaView or PyVista.",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.meshes:
        (args.output_dir / "meshes").mkdir(parents=True, exist_ok=True)

    rungs = (
        (ALL_RUNGS_MM[0], ALL_RUNGS_MM[-1])
        if args.rungs == "endpoints"
        else ALL_RUNGS_MM
    )
    runs = [
        (scene, representation, h_mm)
        for scene in SCENES
        for representation in REPRESENTATIONS
        for h_mm in rungs
    ]
    print(
        f"Running {len(runs)} frozen queries with {args.jobs} workers; "
        f"output={args.output_dir}"
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_run = {
            pool.submit(
                _run_one,
                binary,
                args.output_dir,
                scene,
                representation,
                h_mm,
                args.meshes,
            ): (scene, representation, h_mm)
            for scene, representation, h_mm in runs
        }
        for future in concurrent.futures.as_completed(future_to_run):
            scene, representation, h_mm = future_to_run[future]
            try:
                output, message = future.result()
                print(
                    f"PASS {scene:13s} {representation:14s} "
                    f"h={h_mm:5g} mm -> {output.name}"
                )
                if message:
                    print(f"  {message}")
            except subprocess.CalledProcessError as error:
                failures.append((scene, representation, h_mm, error))
                print(
                    f"FAIL {scene} {representation} h={h_mm:g} mm\n"
                    f"{error.stdout}",
                    file=sys.stderr,
                )

    if failures:
        print(f"{len(failures)} of {len(runs)} runs failed", file=sys.stderr)
        return 1
    print(f"Completed all {len(runs)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
