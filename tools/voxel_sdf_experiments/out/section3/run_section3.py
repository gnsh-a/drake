#!/usr/bin/env python3
"""Runs the section-3 ladders described by ladder_config.json.

Both ladders and every output land in this directory. Resolutions come from
the config's h = c/n rule, whose n values are all clean-phase; see the config's
phase_rule for what that means and why it matters.
"""

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import sys


def _here() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _workspace_root() -> pathlib.Path:
    return _here().parents[3]


def _plan(config: dict) -> list[dict]:
    c = config["resolution_parameterization"]["c_m"]
    penetration = config["penetration_m"]
    runs = []
    for scene in config["scenes"]:
        matched_h = config["ladders"]["matched_h"]
        for rung, n in enumerate(matched_h["n"]):
            for representation in config["representations"]:
                runs.append(
                    {
                        "ladder": "matched_h",
                        "scene": scene,
                        "representation": representation,
                        "n": n,
                        "h": c / n,
                        "rung": rung,
                        "penetration": penetration,
                    }
                )
        matched_count = config["ladders"]["matched_count"]
        for representation in config["representations"]:
            n_values = matched_count["n_by_representation"][representation]
            for rung, n in enumerate(n_values):
                runs.append(
                    {
                        "ladder": "matched_count",
                        "scene": scene,
                        "representation": representation,
                        "n": n,
                        "h": c / n,
                        "rung": rung,
                        "penetration": penetration,
                    }
                )
    return runs


def _stem(run: dict) -> str:
    return (
        f"{run['ladder']}__{run['scene']}__{run['representation']}__n{run['n']}"
    )


def _execute(binary: pathlib.Path, output_dir: pathlib.Path, run: dict):
    output = output_dir / run["ladder"] / f"{_stem(run)}.csv"
    command = [
        str(binary),
        f"--scene={run['scene']}",
        f"--representation={run['representation']}",
        f"--penetration={run['penetration']:.17g}",
        f"--voxel_width={run['h']:.17g}",
        f"--tet_resolution_hint={run['h']:.17g}",
        f"--output={output}",
    ]
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
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=_here())
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--ladder",
        choices=("matched_h", "matched_count", "both"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads((_here() / "ladder_config.json").read_text())
    runs = _plan(config)
    if args.ladder != "both":
        runs = [run for run in runs if run["ladder"] == args.ladder]

    if args.dry_run:
        for run in runs:
            print(f"{_stem(run):68s} h={run['h'] * 1000:.4f} mm")
        print(f"{len(runs)} runs")
        return 0

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    for ladder in {run["ladder"] for run in runs}:
        (args.output_dir / ladder).mkdir(parents=True, exist_ok=True)

    print(f"Running {len(runs)} frozen queries with {args.jobs} workers")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_execute, binary, args.output_dir, run): run
            for run in runs
        }
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            try:
                output, _ = future.result()
                print(f"PASS {_stem(run):68s} -> {output.name}")
            except subprocess.CalledProcessError as error:
                failures.append(run)
                print(
                    f"FAIL {_stem(run)}\n{error.stdout}",
                    file=sys.stderr,
                )

    if failures:
        print(f"{len(failures)} of {len(runs)} runs failed", file=sys.stderr)
        return 1
    print(f"Completed all {len(runs)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
