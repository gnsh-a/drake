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
about it can be shortened the way a settled patch area can. One case at the
finest rung takes eight to eleven minutes and the voxel grids reach 52 GB, which
is what sets the bottom of the ladder at 1.75/12 mm.

eps* follows the benchmark's own rule rather than the last row: the last
post-kick sample whose spin still exceeds kSpinThreshold = 0.1 rad/s. Past that
the disk has stopped spinning and the ratio |v| / (|wz| R) is either divergent
or undefined.

Errors are measured against each representation's own finest run, which is the
rule the paper already uses, and an order is quoted only when the series decays
by at least DECAY_GATE over the fitted rungs.

An earlier version of this script defaulted to referencing both marching-cubes
series against affine's converged eps* of 0.6704, on the argument that affine
reaching 1.000022 of the closed-form pi R^2 anchors its limit to something
outside the study. That argument is wrong and the ladder refutes it.
Tetrahedra reach 1.000018 of the same closed form and converge to 0.6319, held
to four digits down to 0.012 mm -- two representations with the exact patch
area in the limit, 5.7% apart in eps*. Exact area therefore does not determine
eps* and cannot anchor a reference for it. Theory is about 0.653; tet sits 3.2%
low and affine 2.7% high, so they bracket it and neither is validated by it.

--reference still accepts another representation, because the distance between
two methods is worth looking at. What it produces is a distance, not a
convergence order, and the output labels it that way: a slope through
|eps*_affine - eps*_MC| tells you how fast marching cubes approaches affine's
answer, which is only an error if affine's answer is right.

The default rungs mix the dyadic ladder with the thickness-aligned one,
h = 1.75/n, because a purely dyadic subsequence hides what this scene does.
Marching cubes' eps* is not monotone off the dyadic points -- it rises to
0.6025 at 0.583 mm and falls to 0.5721 at 0.4375 -- which is the rim deficit's
grid-phase dependence showing up in the trajectory. A dyadic-only fit therefore
looks cleaner than the method is. Fits exclude rungs coarser than
--fit_max_h_mm, 2.5 mm by default, because eps* is non-monotone at the coarse
end too: 5 mm reads 0.2941, above 2.5 mm's 0.1892, and is simply outside the
asymptotic range.

Cases already present in the output file are skipped, so a ladder this long can
be extended or resumed without repeating what has run.

The phase test, stated before the data existed. The rim deficit is a phase
quantity set by where the grid falls relative to R, so eps* should scatter with
grid phase; a rim projected onto the boundary is not placed by the grid, so it
should scatter less. Two criteria, both fixed in advance and both computed over
the fitted rungs, h <= 2.5 mm:

  reversals    a rung where refining lowers eps*, against the general rise.
               Plain marching cubes has two, at 0.4375 and 0.2917 mm.
               PASS if the projected rim has strictly fewer.
  backtracking the summed size of those downward steps. Plain marching cubes
               totals 0.0378 in eps*. PASS if the projected rim is under half
               of its own plain counterpart's figure.

Both must pass. Reversal count alone is two events and could go either way by
luck, and backtracking alone could shrink merely because the projected series
sits closer to the limit everywhere. Failing either means the phase account of
the deficit is incomplete, which is a result and not a setback.

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

# The dyadic ladder and the thickness-aligned one, h = 1.75/n, interleaved.
DEFAULT_RUNGS_MM = (5.0, 2.5, 1.25, 0.875, 0.625, 1.75 / 3.0, 0.4375, 0.3125,
                    1.75 / 6.0, 0.21875, 0.175, 0.15625, 1.75 / 12.0)
# Coarser than this is outside the asymptotic range; see the module docstring.
DEFAULT_FIT_MAX_H_MM = 2.5
DEFAULT_REPRESENTATIONS = ("plane_clip", "marching_cubes",
                           "marching_cubes_exact_rim")
# Each representation against its own finest run; see the module docstring
# for why another representation's converged value is not a valid reference.
DEFAULT_REFERENCE = "self"
# The paper's rule: a series that has not decayed this far has not earned a
# slope, whatever the R^2 of one drawn through it.
DECAY_GATE = 10.0
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
    slope = ((n * sum(x * y for x, y in zip(xs, ys)) - sum_x * sum_y) /
             denominator)
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
                        help="'self' measures each representation against its "
                             "own finest run and can quote an order. Naming a "
                             "representation instead measures distance to it, "
                             "which is not a convergence order.")
    parser.add_argument("--fit_max_h_mm", type=float,
                        default=DEFAULT_FIT_MAX_H_MM,
                        help="Exclude coarser rungs from the order fits.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rungs = [float(value) for value in args.rungs_mm.split(",")]
    representations = args.representations.split(",")
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / "trajectories_dynamic"
    scratch.mkdir(exist_ok=True)

    rows = []
    if output.exists():
        with open(output, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"resuming; {len(rows)} cases already in {output}")
    finished = {(r["representation"], float(r["h_mm"])) for r in rows}

    for h_mm in sorted(rungs, reverse=True):
        for representation in representations:
            if (representation, h_mm) in finished:
                continue
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

    fitted = sorted(
        [r for r in rows if float(r["h_mm"]) <= args.fit_max_h_mm],
        key=lambda r: -float(r["h_mm"]))

    print(f"\nphase test, rungs <= {args.fit_max_h_mm} mm "
          f"(criteria fixed before the data; see the module docstring):")
    backtracking = {}
    for representation in representations:
        series = [float(r["terminal_eps"]) for r in fitted
                  if r["representation"] == representation]
        drops = [a - b for a, b in zip(series, series[1:]) if b < a]
        backtracking[representation] = sum(drops)
        print(f"  {representation:<26} n {len(series):2d}  "
              f"{len(drops)} reversal(s)  backtracking {sum(drops):.4f}")

    print(f"\nterminal-ratio convergence, rungs <= {args.fit_max_h_mm} mm:")
    for representation in representations:
        series = [(float(r["h_mm"]), float(r["terminal_eps"])) for r in fitted
                  if r["representation"] == representation]
        if len(series) < 4:
            print(f"  {representation:<26} too few rungs")
            continue
        if args.reference == "self":
            reference = series[-1][1]
            label = "own finest run"
            quotable = True
        else:
            source = [float(r["terminal_eps"]) for r in rows
                      if r["representation"] == args.reference]
            if not source:
                print(f"  {representation:<26} no {args.reference} rows")
                continue
            reference = source[-1]
            label = f"{args.reference}'s finest run"
            # Another representation's answer is not this one's limit, so a
            # slope through the gap is a rate of approach to that answer and
            # is deliberately not called an order below.
            quotable = False
        points = [(h, abs(e - reference)) for h, e in series
                  if abs(e - reference) > 0.0]
        if len(points) < 3:
            print(f"  {representation:<26} degenerate against {label}")
            continue
        decay = max(e for _, e in points) / min(e for _, e in points)
        slope, r_squared = _fit_order(points)
        # The gate verdict is rung-set dependent -- plain marching cubes
        # decays 12.1x over the five dyadic rungs and 7.2x over all thirteen,
        # landing on opposite sides of a tenfold gate -- so n and the decay
        # travel with every verdict and neither can be quoted without them.
        head = (f"  {representation:<26} vs {label:<22} n {len(points):2d}  "
                f"decay {decay:7.1f}x")
        if not quotable:
            print(f"{head}   distance slope {slope:5.3f} (NOT an order)")
        elif decay >= DECAY_GATE:
            print(f"{head}   order {slope:5.3f}  R^2 {r_squared:6.3f}")
        else:
            print(f"{head}   no order (gate {DECAY_GATE:.0f}x)")

    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
