#!/usr/bin/env python3
"""Measures how the disk's contact-patch area converges under grid refinement.

This is the static counterpart to run_ladder.py's grid sweep. It exists because
the trajectory sweep cannot reach the resolutions the question needs: the voxel
grid is sized by the ground box, so the paper's 200 x 200 x 20 mm ground costs
52 GB at 0.146 mm and stops there, and a trajectory at that grid takes ten
minutes. Neither limit is about the geometry being measured.

Two observations remove both limits.

The disk is seeded at its equilibrium penetration -- 0.132 um, where the
compliant normal load carries its weight -- so the settled state is reached in
a couple of steps rather than 0.05 s of simulated settling. The check here
therefore runs two settle frames and one measurement frame.

The ground box is shrunk in x and y only. A Box's characteristic length is its
minimum half width, which the unchanged 10 mm half thickness still sets, so the
pressure scale, the penetration, the equal-pressure surface and hence the exact
patch are all identical; only the grid that dominates memory shrinks, by 25x.
The traversed grid belongs to the disk in any case. Verified: at 2.5, 1.25 and
0.625 mm the 40 mm ground reproduces the 200 mm ground's patch area to every
digit printed.

The reference is exact. Both faces are flat and parallel, and the two surfaces
meet on the circle r = R where both pressures vanish, so the patch is the
disk's full cross section pi R^2 to within (delta / R)^2.

Usage:
  bazel build -c opt //tools/voxel_sdf_experiments/disk_farkas:disk_farkas
  tools/voxel_sdf_experiments/disk_farkas/run_rim_convergence.py \\
      --output tools/voxel_sdf_experiments/out/rim_convergence/areas.csv
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

DEFAULT_RUNGS_MM = (5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125)
DEFAULT_REPRESENTATIONS = ("plane_clip", "marching_cubes",
                           "marching_cubes_exact_rim")
# x and y only; see the module docstring for why the thickness may not change.
DEFAULT_GROUND_FULL_SIZE = "0.04,0.04,0.02"
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


def _run_case(binary, representation, h_mm, ground, scratch):
    trajectory = scratch / f"{representation}_{h_mm}.csv"
    command = [
        binary,
        f"--representation={representation}",
        f"--resolution={h_mm * 1e-3:.17g}",
        f"--box_full_size={ground}",
        "--settle_time=0.001",
        "--num_frames=1",
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
        settled = [row for row in csv.DictReader(f)
                   if row["post_kick"] == "false"][-1]
    area = float(settled["contact_area_m2"])
    ratio = area / EXACT_AREA_M2
    return {
        "representation": representation,
        "h_mm": f"{h_mm:.17g}",
        "area_m2": f"{area:.12e}",
        "area_over_exact": f"{ratio:.9f}",
        # The rim error as a fraction of a cell. A patch that is short by a
        # constant number of cells is the signature this study is looking for.
        "rim_deficit_cells":
            f"{(1.0 - math.sqrt(max(ratio, 0.0))) * DISK_RADIUS_M / (h_mm * 1e-3):.6f}",
        "surface_faces": settled["surface_faces"],
        "normal_force_z_N": settled["normal_force_z_N"],
        "seconds": f"{elapsed:.1f}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--rungs_mm", default=",".join(
        f"{h:.17g}" for h in DEFAULT_RUNGS_MM))
    parser.add_argument("--representations", default=",".join(
        DEFAULT_REPRESENTATIONS))
    parser.add_argument("--ground_full_size", default=DEFAULT_GROUND_FULL_SIZE,
                        help="Shrink x and y freely; changing z moves the "
                             "Box characteristic length and with it the "
                             "penetration the study is posed at.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rungs = [float(value) for value in args.rungs_mm.split(",")]
    representations = args.representations.split(",")
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / "trajectories"
    scratch.mkdir(exist_ok=True)

    rows = []
    for h_mm in sorted(rungs, reverse=True):
        for representation in representations:
            row = _run_case(args.binary, representation, h_mm,
                            args.ground_full_size, scratch)
            rows.append(row)
            print(f"{representation:<26} h={h_mm:<10.6g} "
                  f"area/exact={float(row['area_over_exact']):.6f}  "
                  f"rim={float(row['rim_deficit_cells']):+8.5f} cells  "
                  f"{float(row['seconds']):6.1f} s", flush=True)
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nexact area {EXACT_AREA_M2:.9e} m^2\n")
    print("area-error order, least squares over every rung:")
    for representation in representations:
        points = [(float(r["h_mm"]), abs(1.0 - float(r["area_over_exact"])))
                  for r in rows if r["representation"] == representation]
        points = [p for p in points if p[1] > 0.0]
        order, r_squared = _fit_order(points)
        decay = (max(e for _, e in points) / min(e for _, e in points)
                 if points else float("nan"))
        span = (max(h for h, _ in points) / min(h for h, _ in points)
                if points else float("nan"))
        print(f"  {representation:<26} order {order:5.3f}  R^2 {r_squared:6.3f}"
              f"  decay {decay:9.1f}x over {span:.0f}x in h")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
