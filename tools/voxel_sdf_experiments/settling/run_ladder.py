#!/usr/bin/env python3
"""Runs the Phase 2 free-body settling convergence ladder."""

import argparse
import concurrent.futures
import csv
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
    mass: float | None,
    trajectories: bool = False,
    trajectory_stride: int = 1,
    duration_periods: float | None = None,
) -> tuple[pathlib.Path, dict[str, str], str]:
    h_m = h_mm / 1000.0
    stem = f"{scene}__{representation}__h_{_rung_label(h_mm)}mm"
    output = output_dir / f"{stem}.csv"
    command = [
        str(binary),
        f"--scene={scene}",
        f"--representation={representation}",
        f"--voxel_width={h_m:.17g}",
        f"--tet_resolution_hint={h_m:.17g}",
        f"--output={output}",
    ]
    if mass is not None:
        command.append(f"--mass={mass:.17g}")
    if meshes:
        command.append(f"--mesh_output={output_dir / 'meshes' / stem}.vtk")
    if trajectories:
        command.append(
            f"--trajectory={output_dir / 'trajectories' / stem}.csv"
        )
        command.append(f"--trajectory_stride={trajectory_stride}")
    if duration_periods is not None:
        command.append(f"--duration_periods={duration_periods:.17g}")
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    return output, row, completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=_workspace_root()
        / "bazel-bin/tools/voxel_sdf_experiments/settling/settling",
        help="Path to the already-built settling binary.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_workspace_root()
        / "tools/voxel_sdf_experiments/out/settling_ladder",
    )
    parser.add_argument(
        "--rungs",
        choices=("endpoints", "coarse", "all"),
        default="all",
        help="endpoints runs 10 and 1.25 mm; coarse runs 10, 5 and 2.5 mm; all "
        "runs the complete four-rung ladder. Cost climbs roughly 4-8x per "
        "halving of h -- a single 1.25 mm run takes hours, so coarse is the "
        "right choice for a quick check and all is an overnight job.",
    )
    parser.add_argument(
        "--rungs_mm",
        type=str,
        default="",
        help="Comma-separated rung list in mm, overriding --rungs.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Maximum concurrent settling subprocesses.",
    )
    parser.add_argument(
        "--meshes",
        action="store_true",
        help="Also write each settled surface under OUTPUT_DIR/meshes.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        help="Optional common mass in kilograms, applied to every scene. By "
        "default each scene derives its own mass from the closed form, so both "
        "settle at the same off-boundary equilibrium penetration; a single "
        "common mass cannot do that, since the two scenes carry different "
        "loads at equal penetration.",
    )
    parser.add_argument(
        "--scene",
        choices=("all",) + SCENES,
        default="all",
        help="Run both scenes or filter to one scene.",
    )
    parser.add_argument(
        "--trajectories",
        action="store_true",
        help="Also write a per-step trajectory under OUTPUT_DIR/trajectories. "
        "Needed to compare transients against a reference; the closed form "
        "pins only the final penetration, not the path taken to it.",
    )
    parser.add_argument(
        "--trajectory_stride",
        type=int,
        default=10,
        help="Keep every Nth trajectory sample. A trajectory makes the surface "
        "and connected-component pass run at every written step instead of "
        "only inside the settled window, so this is what keeps a trajectory "
        "sweep affordable. Every run in one study must share this value.",
    )
    parser.add_argument(
        "--reference_mm",
        type=float,
        default=0.0,
        help="Also run tetrahedra at this resolution, once per scene, as the "
        "trajectory reference. The transient has no closed form, so a fine tet "
        "run stands in for one. 0 disables it. This rung is expensive: cost "
        "climbs 4-8x per halving.",
    )
    parser.add_argument(
        "--duration_periods",
        type=float,
        default=None,
        help="Simulated duration in natural periods. The body stops moving by "
        "about three, so the binary's default of 15 is conservative. Note that "
        "marching cubes loses contact at 9.3 periods on sphere_sphere, so a "
        "value below that hides the fall-through.",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.mass is not None and args.mass <= 0.0:
        parser.error("--mass must be positive")
    if args.rungs_mm:
        try:
            rungs = tuple(float(value) for value in args.rungs_mm.split(","))
        except ValueError as error:
            parser.error(f"invalid --rungs_mm: {error}")
        if not rungs or not all(h > 0.0 for h in rungs):
            parser.error("--rungs_mm values must be positive")
    elif args.rungs == "endpoints":
        rungs = (ALL_RUNGS_MM[0], ALL_RUNGS_MM[-1])
    elif args.rungs == "coarse":
        rungs = ALL_RUNGS_MM[:-1]
    else:
        rungs = ALL_RUNGS_MM
    scenes = SCENES if args.scene == "all" else (args.scene,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.meshes:
        (args.output_dir / "meshes").mkdir(parents=True, exist_ok=True)
    runs = [
        (scene, representation, h_mm)
        for scene in scenes
        for representation in REPRESENTATIONS
        for h_mm in rungs
    ]
    if args.reference_mm > 0.0:
        # One reference per scene, not per representation: it is the stand-in
        # for the closed form, so all three are compared against the same one.
        runs += [(scene, "tet", args.reference_mm) for scene in scenes
                 if (scene, "tet", args.reference_mm) not in runs]
    mass_description = (
        "scene defaults" if args.mass is None else f"{args.mass} kg"
    )
    print(
        f"Running {len(runs)} settling simulations with {args.jobs} workers; "
        f"mass={mass_description}, rungs={rungs} mm, output={args.output_dir}"
    )

    failures = []
    rows: dict[tuple[str, str, float], dict[str, str]] = {}
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
                args.mass,
                args.trajectories,
                args.trajectory_stride,
                args.duration_periods,
            ): (scene, representation, h_mm)
            for scene, representation, h_mm in runs
        }
        for future in concurrent.futures.as_completed(future_to_run):
            scene, representation, h_mm = future_to_run[future]
            try:
                output, row, message = future.result()
                rows[(scene, representation, h_mm)] = row
                # A run that lost contact still exits zero and still writes a
                # row, so the process status alone would report it as fine. Call
                # it out here: over a long unattended ladder this is the line
                # that says which rows are physically meaningless.
                settled = row.get("settled", "").strip().lower() == "true"
                status = "PASS" if settled else "UNSETTLED"
                print(
                    f"{status:9s} {scene:13s} {representation:14s} "
                    f"h={h_mm:5g} mm m_hat="
                    f"{float(row['elements_across_patch']):7.3f} "
                    f"penetration_error_m="
                    f"{float(row['penetration_error_m']):+.6e} "
                    f"-> {output.name}"
                )
                if message:
                    print(f"  {message}")
            except (
                subprocess.CalledProcessError,
                OSError,
                StopIteration,
            ) as error:
                failures.append((scene, representation, h_mm, error))
                output = getattr(error, "stdout", str(error))
                print(
                    f"FAIL {scene} {representation} h={h_mm:g} mm\n{output}",
                    file=sys.stderr,
                )

    if rows:
        summary = args.output_dir / "summary.csv"
        ordered_rows = [rows[run] for run in runs if run in rows]
        with summary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ordered_rows[0].keys())
            writer.writeheader()
            writer.writerows(ordered_rows)
        print(f"Wrote {len(ordered_rows)} rows to {summary}")
    if failures:
        print(f"{len(failures)} of {len(runs)} runs failed", file=sys.stderr)
        return 1
    unsettled = [
        run
        for run, row in rows.items()
        if row.get("settled", "").strip().lower() != "true"
    ]
    print(f"Completed all {len(runs)} runs.")
    if unsettled:
        # Not an error: losing contact is itself a result, and the row is kept
        # so the convergence fit can decide what to do with it.
        print(f"{len(unsettled)} of {len(runs)} did not settle:")
        for scene, representation, h_mm in sorted(unsettled):
            print(f"  {scene} {representation} h={h_mm:g} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
