#!/usr/bin/env python3
"""Measures how the disk's terminal slip-to-spin ratio converges under grid
refinement, with and without the projected rim.

run_rim_convergence.py is about the patch's area at a single settled instant.
This is about the trajectory, and the two do not answer the same question. The
projected rim restores second-order area convergence; it does not restore the
order of eps*, because fixing where the patch ends does not fix the friction
integral over its interior, which a piecewise-linear pressure on a triangulated
surface still resolves at first order. That distinction is the point of running
this separately, so it exists as a file rather than as a claim.

Unlike the static study this uses the benchmark's own settings throughout --
the full 200 x 200 x 20 mm ground, 0.05 s of settling, and 400 post-kick frames
-- because a terminal ratio is a property of the whole trajectory and nothing
about it can be shortened the way a settled patch area can. One case at
0.15625 mm takes eight to eleven minutes, which is why the ladder stops there.

eps* follows the benchmark's own rule rather than the last row: the last
post-kick sample whose spin still exceeds kSpinThreshold = 0.1 rad/s. Past that
the disk has stopped spinning and the ratio |v| / (|wz| R) is either divergent
or undefined.

The reference is the finest affine value rather than theory. The universal
terminal ratio for this scene is about 0.653, but affine converges to 0.6704
here and locks there, and an order measured against the method that has
converged is the honest one. Reading it against 0.653 instead would fold
affine's own residual into both marching-cubes series.

Usage:
  bazel build -c opt //tools/voxel_sdf_experiments/disk_farkas:disk_farkas
  tools/voxel_sdf_experiments/disk_farkas/run_rim_dynamics.py \\
      --output tools/voxel_sdf_experiments/out/rim_convergence/dynamics.csv
"""

import argparse
import csv
import math
import pathlib
import subprocess
import sys
import time

# Masterjohn et al.'s quarter-coin puck; matches disk_plane.yaml.
DISK_RADIUS_M = 0.01213
EXACT_AREA_M2 = math.pi * DISK_RADIUS_M * DISK_RADIUS_M
# disk_farkas.cc's kSpinThreshold. Duplicated deliberately: this script has to
# reproduce the binary's own definition of a terminal ratio, not invent one.
SPIN_THRESHOLD_RAD_S = 0.1

DEFAULT_RUNGS_MM = (2.5, 1.25, 0.625, 0.3125, 0.15625)
DEFAULT_REPRESENTATIONS = ("plane_clip", "marching_cubes",
                           "marching_cubes_exact_rim")
DEFAULT_REFERENCE = "plane_clip"
# The benchmark's own trajectory settings; see the module docstring.
SETTLE_TIME_S = 0.05
SETTLE_TIME_STEP_S = 6.25e-5
TIME_STEP_S = 1.25e-4
FRAMES_PER_SECOND = 2000.0
NUM_FRAMES = 400
DEFAULT_BINARY = "bazel-bin/tools/voxel_sdf_experiments/disk_farkas/disk_farkas"


def _fit_order(points):
    """Least-squares slope of log(error) against log(h), with its R^2."""
    if len(points) < 3:
        return float("nan"), float("nan")
    xs = [math.log(h) for h, _ in points]
    ys = [math.log(e) for _, e in points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    denominator = n * sum(x * x for x in xs) - sum_x * sum_x
    if denominator == 0.0:
        return float("nan"), float("nan")
    slope = (n * sum(x * y for x, y in zip(xs, ys)) - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    mean_y = sum_y / n
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, (1.0 - residual / total if total > 0.0 else float("nan"))


def _terminal_epsilon(rows):
    """The benchmark's rule: the last post-kick sample still spinning."""
    result = float("nan")
    for row in rows:
        if row["post_kick"] != "true":
            continue
        epsilon = float(row["eps"])
        if (math.isfinite(epsilon) and
                abs(float(row["wz_rad_s"])) >= SPIN_THRESHOLD_RAD_S):
            result = epsilon
    return result


def _run_case(binary, representation, h_mm, scratch):
    trajectory = scratch / f"{representation}_{h_mm}.csv"
    command = [
        binary,
        f"--representation={representation}",
        f"--resolution={h_mm * 1e-3:.17g}",
        f"--settle_time={SETTLE_TIME_S:.17g}",
        f"--settle_time_step={SETTLE_TIME_STEP_S:.17g}",
        f"--time_step={TIME_STEP_S:.17g}",
        f"--fps={FRAMES_PER_SECOND:.17g}",
        f"--num_frames={NUM_FRAMES}",
        f"--output={trajectory}",
    ]
    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{representation} at h = {h_mm} mm failed:\n"
            f"{completed.stderr.strip()[-2000:]}")
    with open(trajectory, newline="") as f:
        rows = list(csv.DictReader(f))
    settled = [row for row in rows if row["post_kick"] == "false"][-1]
    final = rows[-1]
    return {
        "representation": representation,
        "h_mm": f"{h_mm:.17g}",
        "terminal_eps": f"{_terminal_epsilon(rows):.6f}",
        "settled_area_over_exact":
            f"{float(settled['contact_area_m2']) / EXACT_AREA_M2:.9f}",
        "settled_normal_force_z_N": settled["normal_force_z_N"],
        "settled_surface_faces": settled["surface_faces"],
        "final_spin_rad_s": f"{abs(float(final['wz_rad_s'])):.6f}",
        "num_components": final["num_components"],
        # Wall times here are recorded, not measured: this script does not
        # schedule against other load, so a contended run is indistinguishable
        # from an uncontended one.
        "seconds": f"{elapsed:.1f}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--rungs_mm", default=",".join(
        f"{h:.17g}" for h in DEFAULT_RUNGS_MM))
    parser.add_argument("--representations", default=",".join(
        DEFAULT_REPRESENTATIONS))
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="Representation whose finest rung is the "
                             "converged value the orders are measured against.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rungs = [float(value) for value in args.rungs_mm.split(",")]
    representations = args.representations.split(",")
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / "trajectories_dynamic"
    scratch.mkdir(exist_ok=True)

    rows = []
    for h_mm in sorted(rungs, reverse=True):
        for representation in representations:
            row = _run_case(args.binary, representation, h_mm, scratch)
            rows.append(row)
            print(f"{representation:<26} h={h_mm:<10.6g} "
                  f"eps*={float(row['terminal_eps']):.4f}  "
                  f"area/exact={float(row['settled_area_over_exact']):.6f}  "
                  f"{float(row['seconds']):7.1f} s", flush=True)
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    finest = min(rungs)
    reference_rows = [r for r in rows
                      if r["representation"] == args.reference
                      and float(r["h_mm"]) == finest]
    if not reference_rows:
        print(f"\nno {args.reference} value at {finest} mm; skipping orders")
        return 0
    reference = float(reference_rows[0]["terminal_eps"])
    print(f"\nreference: {args.reference} at {finest} mm, eps* = "
          f"{reference:.6f}\n")
    print("terminal-ratio error against that reference:")
    for representation in representations:
        if representation == args.reference:
            continue
        points = [(float(r["h_mm"]),
                   abs(float(r["terminal_eps"]) - reference))
                  for r in rows if r["representation"] == representation]
        points = [p for p in points if p[1] > 0.0 and math.isfinite(p[1])]
        order, r_squared = _fit_order(points)
        series = "  ".join(f"{e:.4f}" for _, e in
                           sorted(points, key=lambda p: -p[0]))
        print(f"  {representation:<26} order {order:5.3f}  R^2 {r_squared:6.3f}"
              f"   {series}")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
