#!/usr/bin/env python3
"""Runs the free-disk sweeps serially and writes one trajectory CSV per case.

Three sweeps, matching the three studies the benchmark supports:

  grid   grid size at a fixed time step -- spatial convergence
  dt     time step at a fixed grid      -- temporal convergence
  eps0   initial slip-to-spin ratio     -- invariance of the terminal ratio

The dt sweep pins the settle phase at a single time step for every case, so
all post-kick trajectories start from an identical settled state and the sweep
measures integration error alone rather than a different starting condition per
case.

Concurrency is bounded by memory, not by cores. Voxel grids are sized by the
ground box rather than by the 12 mm disk, so their footprint grows like h^-3:
measured 0.12 GB at 1.25 mm and 0.70 GB at 0.625 mm, which fits
0.032 + 0.162 * h_mm^-3 GB and predicts 42 GB at the 0.15625 mm cap -- matching
the ~40 GB the original study recorded. Tetrahedra stay under 500 MB even at
0.012 mm, which is why the resolution floor is per representation: one cap set
for the voxel kernels would also block the cheap tet rungs that carry the
constraint-count axis.

The scheduler therefore admits a case only when both --jobs and
--memory_budget_gb allow it, and runs anything at or above --solo_gb alone.
Wall times from a case that shared the machine are recorded but marked
timing_valid=false: this workload degrades once the working set stops fitting
in cache, so a contended time is not a cost measurement.
"""

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import threading
import time


REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
DEFAULT_RUNGS_MM = (5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625)
# Tet-only extras. Tet constraint count grows like 1/h against the voxel pair's
# 1/h^2, so equal h is nowhere near equal cost; these exist to widen the
# constraint-count axis, not the resolution axis.
DEFAULT_TET_EXTRA_RUNGS_MM = (0.08, 0.04, 0.02, 0.012)
# Per-representation floors, in mm. Voxel grids are memory bound; tet is not.
FINEST_ALLOWED_MM = {
    "tet": 0.012,
    "plane_clip": 0.15625,
    "marching_cubes": 0.15625,
}
DEFAULT_TIME_STEPS_MS = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)
# The paper sweeps eps0 over [0.1, 10]. The extremes are out of reach here: a
# voxel SDF cannot voxelize an infinite plane, so a long slide runs off a
# bounded ground box, and the singular omega -> 0 stop at the low end does not
# converge under fixed-step velocity-level integration.
DEFAULT_EPS0 = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DISK_RADIUS_M = 0.01213
EPS0_SPIN_RAD_S = 12.0
GRID_SWEEP_TIME_STEP_S = 1.25e-4
DT_SWEEP_RUNG_MM = 2.5
EPS0_SWEEP_RUNG_MM = 1.25
# A larger eps0 slides the disk further, so the eps0 sweep needs a wider ground
# than the 0.2 m default. This costs memory, which is why it is not the default
# everywhere: the voxel grid is sized by the box.
EPS0_BOX_FULL_SIZE_M = (0.4, 0.4, 0.02)


# Peak resident set fitted to two measurements of the affine kernel, in GB.
# The constant is process overhead; the cubic term is the box-sized voxel grid.
MEMORY_FIXED_GB = 0.032
MEMORY_PER_INVERSE_CUBIC_MM_GB = 0.162
TET_MEMORY_GB = 0.5
DEFAULT_BOX_AREA_M2 = 0.2 * 0.2


def estimate_memory_gb(case) -> float:
    """Peak resident set this case is expected to reach.

    Tetrahedral meshes do not carry a voxel grid, so they are flat in h. The
    eps0 sweep widens the ground, and the grid is sized by the ground, so its
    footprint scales with the box footprint rather than staying put.
    """
    if case["representation"] == "tet":
        return TET_MEMORY_GB
    box_scale = 1.0
    if case["eps0"] is not None:
        box_scale = (
            EPS0_BOX_FULL_SIZE_M[0] * EPS0_BOX_FULL_SIZE_M[1]
        ) / DEFAULT_BOX_AREA_M2
    return (
        MEMORY_FIXED_GB
        + MEMORY_PER_INVERSE_CUBIC_MM_GB * box_scale / case["h_mm"] ** 3
    )


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _parse_floats(parser, name: str, text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in text.split(","))
    except ValueError as error:
        parser.error(f"invalid {name}: {error}")
    if not values or not all(value > 0.0 for value in values):
        parser.error(f"{name} values must be positive")
    return values


def _grid_cases(rungs, tet_extras, representations):
    """One case per (representation, rung), dropping rungs past each floor.

    Dropped rungs are returned rather than skipped silently: a ladder that
    quietly shortens itself reads downstream as a representation that simply
    stopped converging.
    """
    cases, dropped = [], []
    for representation in representations:
        floor = FINEST_ALLOWED_MM[representation]
        wanted = list(rungs)
        if representation == "tet":
            wanted += [value for value in tet_extras if value not in wanted]
        for h_mm in sorted(wanted, reverse=True):
            if h_mm < floor:
                dropped.append((representation, h_mm, floor))
                continue
            cases.append(
                {
                    "sweep": "grid",
                    "representation": representation,
                    "h_mm": h_mm,
                    "time_step_s": GRID_SWEEP_TIME_STEP_S,
                    "eps0": None,
                    "stem": f"grid__{representation}__h_{_label(h_mm)}mm",
                }
            )
    return cases, dropped


def _dt_cases(time_steps_ms, representations):
    return [
        {
            "sweep": "dt",
            "representation": representation,
            "h_mm": DT_SWEEP_RUNG_MM,
            "time_step_s": dt_ms / 1000.0,
            "eps0": None,
            "stem": f"dt__{representation}__dt_{_label(dt_ms)}ms",
        }
        for representation in representations
        for dt_ms in sorted(time_steps_ms, reverse=True)
    ]


def _eps0_cases(eps0_values, representations):
    return [
        {
            "sweep": "eps0",
            "representation": representation,
            "h_mm": EPS0_SWEEP_RUNG_MM,
            "time_step_s": GRID_SWEEP_TIME_STEP_S,
            "eps0": eps0,
            "stem": f"eps0__{representation}__e_{_label(eps0)}",
        }
        for representation in representations
        for eps0 in eps0_values
    ]


def _command(binary, case, output, num_frames=0):
    """Builds one invocation.

    eps0 = |v| / (omega_z * R), so holding the spin fixed and solving for the
    slide is what turns a target ratio into a kick. The scene's own 0.2 m/s and
    12 rad/s give eps0 = 1.37, which this reproduces exactly at that ratio.
    """
    command = [
        str(binary),
        f"--representation={case['representation']}",
        f"--resolution={case['h_mm'] / 1000.0:.17g}",
        f"--time_step={case['time_step_s']:.17g}",
        f"--output={output}",
    ]
    if num_frames > 0:
        command.append(f"--num_frames={num_frames}")
    if case["eps0"] is not None:
        slide = case["eps0"] * EPS0_SPIN_RAD_S * DISK_RADIUS_M
        command.append(f"--init_linear_velocity={slide:.17g},0,0")
        command.append(f"--init_angular_velocity=0,0,{EPS0_SPIN_RAD_S:.17g}")
        command.append(
            "--box_full_size="
            + ",".join(f"{value:.17g}" for value in EPS0_BOX_FULL_SIZE_M)
        )
    return command


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
        "--sweeps",
        nargs="+",
        choices=("grid", "dt", "eps0"),
        default=("grid", "dt", "eps0"),
        help="Which sweeps to run.",
    )
    parser.add_argument(
        "--rungs_mm",
        default=",".join(f"{value:g}" for value in DEFAULT_RUNGS_MM),
        help="Comma-separated grid sizes in mm for the grid sweep.",
    )
    parser.add_argument(
        "--tet_extra_rungs_mm",
        default=",".join(f"{value:g}" for value in DEFAULT_TET_EXTRA_RUNGS_MM),
        help="Extra grid sizes run for tetrahedra only, to widen the "
        "constraint-count axis. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--time_steps_ms",
        default=",".join(f"{value:g}" for value in DEFAULT_TIME_STEPS_MS),
        help="Comma-separated time steps in ms for the dt sweep.",
    )
    parser.add_argument(
        "--eps0",
        default=",".join(f"{value:g}" for value in DEFAULT_EPS0),
        help="Comma-separated initial slip-to-spin ratios.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Override the post-kick frame count. Use a small value to "
        "exercise the whole pipeline cheaply before a real sweep; the "
        "terminal ratio is meaningless at low frame counts.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum concurrent cases. Memory, not this number, is usually "
        "the binding constraint; see --memory_budget_gb.",
    )
    parser.add_argument(
        "--memory_budget_gb",
        type=float,
        default=120.0,
        help="Ceiling on the summed estimated peak memory of concurrent "
        "cases. Swapping is what makes the machine unusable, so this is the "
        "setting that protects it -- not --jobs.",
    )
    parser.add_argument(
        "--solo_gb",
        type=float,
        default=20.0,
        help="A case estimated at or above this runs alone, whatever --jobs "
        "says.",
    )
    parser.add_argument(
        "--max_case_gb",
        type=float,
        default=0.0,
        help="Skip cases estimated above this, reporting each. 0 runs "
        "everything. Use it to take the cheap majority of a ladder now and "
        "leave the memory-bound tail for a quiet machine.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the cases and stop. Use this before a long sweep.",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not args.dry_run and not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    rungs = _parse_floats(parser, "--rungs_mm", args.rungs_mm)
    tet_extras = (
        _parse_floats(parser, "--tet_extra_rungs_mm", args.tet_extra_rungs_mm)
        if args.tet_extra_rungs_mm.strip()
        else ()
    )
    time_steps_ms = _parse_floats(parser, "--time_steps_ms", args.time_steps_ms)
    eps0_values = _parse_floats(parser, "--eps0", args.eps0)

    cases, dropped = [], []
    if "grid" in args.sweeps:
        grid, dropped = _grid_cases(rungs, tet_extras, args.representations)
        cases += grid
    if "dt" in args.sweeps:
        cases += _dt_cases(time_steps_ms, args.representations)
    if "eps0" in args.sweeps:
        cases += _eps0_cases(eps0_values, args.representations)

    if args.max_case_gb > 0.0:
        kept = [
            case
            for case in cases
            if estimate_memory_gb(case) <= args.max_case_gb
        ]
        for case in cases:
            if case not in kept:
                print(
                    f"SKIP {case['sweep']:5s} {case['representation']:14s} "
                    f"h={case['h_mm']:g} mm needs about "
                    f"{estimate_memory_gb(case):.0f} GB, over the "
                    f"{args.max_case_gb:g} GB per-case limit"
                )
        cases = kept

    for representation, h_mm, floor in dropped:
        print(
            f"SKIP {representation:14s} h={h_mm:g} mm is finer than its "
            f"{floor:g} mm floor"
        )
    print(
        f"Running {len(cases)} disk simulations; "
        f"jobs<={args.jobs}, memory budget {args.memory_budget_gb:g} GB, "
        f"solo at {args.solo_gb:g} GB; "
        f"sweeps={','.join(args.sweeps)}, output={args.output_dir}"
    )
    if args.dry_run:
        for case in cases:
            print(
                f"  {case['sweep']:5s} {case['representation']:14s} "
                f"h={case['h_mm']:8g} mm dt={case['time_step_s'] * 1e3:8g} ms "
                f"eps0={case['eps0'] if case['eps0'] is not None else '-':>4} "
                f"{estimate_memory_gb(case):8.2f} GB"
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    manifest_lock = threading.Lock()
    running_gb = 0.0
    running = 0
    admission = threading.Condition()
    completed = 0

    def run_case(index, case, memory_gb):
        nonlocal running_gb, running, completed
        output = args.output_dir / f"{case['stem']}.csv"
        started = time.monotonic()
        completed_process = subprocess.run(
            _command(binary, case, output, args.num_frames),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.monotonic() - started
        with admission:
            # A case that overlapped anything is timed under contention, and
            # this workload degrades once the working set leaves cache.
            shared = running > 1
            running_gb -= memory_gb
            running -= 1
            completed += 1
            done = completed
            admission.notify_all()
        status = "PASS" if completed_process.returncode == 0 else "FAIL"
        with manifest_lock:
            print(
                f"{status} {done:2d}/{len(cases)} {case['sweep']:5s} "
                f"{case['representation']:14s} h={case['h_mm']:8g} mm "
                f"dt={case['time_step_s'] * 1e3:7g} ms {elapsed:8.1f} s "
                f"{memory_gb:6.2f} GB -> {output.name}"
            )
            if completed_process.returncode != 0:
                print(completed_process.stdout.strip()[-2000:])
            manifest.append(
                {
                    **case,
                    "wall_time_s": elapsed,
                    "estimated_gb": memory_gb,
                    "timing_valid": "false" if shared else "true",
                    "returncode": completed_process.returncode,
                    "csv": output.name,
                }
            )

    threads = []
    for index, case in enumerate(cases, start=1):
        memory_gb = estimate_memory_gb(case)
        solo = memory_gb >= args.solo_gb
        with admission:
            while True:
                fits_jobs = running < (1 if solo else args.jobs)
                fits_memory = running_gb + memory_gb <= args.memory_budget_gb
                if fits_jobs and fits_memory and (running == 0 or not solo):
                    break
                admission.wait(timeout=1.0)
            running_gb += memory_gb
            running += 1
        thread = threading.Thread(
            target=run_case, args=(index, case, memory_gb)
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    manifest.sort(
        key=lambda row: (
            row["sweep"],
            row["representation"],
            -row["h_mm"],
            row["time_step_s"],
        )
    )
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {len(manifest)} rows to {manifest_path}")
    void = sum(1 for row in manifest if row["timing_valid"] == "false")
    failed = sum(1 for row in manifest if row["returncode"] != 0)
    print(
        f"Completed {len(cases)} runs; {failed} failed, "
        f"{void} have contended (unusable) wall times."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
