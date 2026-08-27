"""Benchmarks simulation-only realtime factor for tet and voxel-SDF cases."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

_TARGET = "//examples/hydroelastic/spatula_slip_control:spatula_slip_control"
_BINARY = (
    "bazel-bin/examples/hydroelastic/spatula_slip_control/spatula_slip_control"
)
_DEFAULT_OUTPUT = "examples/hydroelastic/spatula_slip_control/out"
_VALUE_PATTERN = re.compile(
    r"^(simulation_(?:time_s|wall_time_s|rtf|steps|discrete_updates|"
    r"unrestricted_updates)): (.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Case:
    name: str
    representation: str
    scale: float
    bubble_resolution_mm: float
    handle_resolution_mm: float
    each_bubble_elements: int
    handle_elements: int
    element_name: str

    @property
    def total_elements(self):
        return 2 * self.each_bubble_elements + self.handle_elements


_CASES = (
    Case("tet coarse", "tet", 1.0, 40.0, 5.0, 128, 95, "tets"),
    Case("voxel coarse", "voxel_sdf", 1.0, 40.0, 5.0, 36, 792, "cells"),
    Case("tet fine", "tet", 0.5, 20.0, 2.5, 512, 190, "tets"),
    Case("voxel fine", "voxel_sdf", 0.5, 20.0, 2.5, 175, 6336, "cells"),
    Case("tet finer", "tet", 0.25, 10.0, 1.25, 2048, 380, "tets"),
    Case(
        "voxel finer",
        "voxel_sdf",
        0.25,
        10.0,
        1.25,
        1053,
        50688,
        "cells",
    ),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-sec", type=float, default=30.0)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(_DEFAULT_OUTPUT)
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use the existing bazel-bin executable.",
    )
    args = parser.parse_args()
    if args.simulation_sec <= 0.0:
        parser.error("--simulation-sec must be positive")
    if args.time_step <= 0.0:
        parser.error("--time-step must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    expected_updates = round(args.simulation_sec / args.time_step)
    if not math.isclose(
        expected_updates * args.time_step,
        args.simulation_sec,
        rel_tol=0.0,
        abs_tol=1.0e-12 * max(1.0, args.simulation_sec),
    ):
        parser.error("--simulation-sec must be an integer number of time steps")
    return args


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _absolute(path, root):
    return path if path.is_absolute() else root / path


def _build(root):
    subprocess.run(
        [
            "bazel",
            "--output_user_root=/tmp/bazel-output",
            "build",
            "-c",
            "opt",
            "--experimental_collect_system_network_usage=false",
            _TARGET,
        ],
        cwd=root,
        check=True,
    )


def _parse_timing(stdout):
    values = dict(_VALUE_PATTERN.findall(stdout))
    expected = {
        "simulation_time_s",
        "simulation_wall_time_s",
        "simulation_rtf",
        "simulation_steps",
        "simulation_discrete_updates",
        "simulation_unrestricted_updates",
    }
    if values.keys() != expected:
        raise RuntimeError(
            f"timing output has fields {sorted(values)}, expected "
            f"{sorted(expected)}"
        )
    result = {
        "simulation_time_s": float(values["simulation_time_s"]),
        "simulation_wall_time_s": float(values["simulation_wall_time_s"]),
        "simulation_rtf": float(values["simulation_rtf"]),
        "simulation_steps": int(values["simulation_steps"]),
        "simulation_discrete_updates": int(
            values["simulation_discrete_updates"]
        ),
        "simulation_unrestricted_updates": int(
            values["simulation_unrestricted_updates"]
        ),
    }
    if any(
        not math.isfinite(result[field]) or result[field] <= 0.0
        for field in (
            "simulation_time_s",
            "simulation_wall_time_s",
            "simulation_rtf",
        )
    ):
        raise RuntimeError(
            f"timing output is not finite and positive: {result}"
        )
    return result


def _run_case(args, root, binary, case, run_label):
    tet_scale = case.scale if case.representation == "tet" else 1.0
    voxel_scale = case.scale if case.representation == "voxel_sdf" else 1.0
    command = [
        str(binary),
        f"--hydroelastic_representation={case.representation}",
        f"--tet_resolution_hint_scale={tet_scale}",
        f"--voxel_sdf_width_scale={voxel_scale}",
        f"--simulation_sec={args.simulation_sec}",
        f"--mbp_discrete_update_period={args.time_step}",
        "--gripper_force=1.5",
        "--amplitude=5.0",
        "--duty_cycle=0.5",
        "--period=3.0",
        "--stiction_tolerance=1e-4",
        "--contact_model=hydroelastic",
        "--contact_surface_representation=polygon",
        "--contact_approximation=lagged",
        "--realtime_rate=0.0",
        "--visualize=false",
        "--report_simulation_timing",
    ]
    print(f"{run_label}: {case.name}", flush=True)
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"{case.name} failed with code {result.returncode}")
    timing = _parse_timing(result.stdout)
    expected_updates = round(args.simulation_sec / args.time_step)
    if not math.isclose(
        timing["simulation_time_s"],
        args.simulation_sec,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError(
            f"{case.name} advanced {timing['simulation_time_s']} seconds"
        )
    if timing["simulation_steps"] != expected_updates:
        raise RuntimeError(
            f"{case.name} took {timing['simulation_steps']} steps; expected "
            f"{expected_updates}"
        )
    print(
        f"  wall={timing['simulation_wall_time_s']:.6f} s, "
        f"RTF={timing['simulation_rtf']:.6f}",
        flush=True,
    )
    return timing


def _write_runs(path, rows):
    fields = (
        "case",
        "representation",
        "scale",
        "bubble_resolution_mm",
        "handle_resolution_mm",
        "element_name",
        "total_elements",
        "repetition",
        "simulation_time_s",
        "simulation_wall_time_s",
        "simulation_rtf",
        "simulation_steps",
        "simulation_discrete_updates",
        "simulation_unrestricted_updates",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(cases, rows):
    result = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case.name]
        rtfs = [row["simulation_rtf"] for row in case_rows]
        walls = [row["simulation_wall_time_s"] for row in case_rows]
        result.append(
            {
                "case": case.name,
                "bubble_resolution_mm": case.bubble_resolution_mm,
                "handle_resolution_mm": case.handle_resolution_mm,
                "element_name": case.element_name,
                "total_elements": case.total_elements,
                "median_wall_time_s": statistics.median(walls),
                "median_rtf": statistics.median(rtfs),
                "min_rtf": min(rtfs),
                "max_rtf": max(rtfs),
            }
        )
    return result


def _write_summary(path, args, rows):
    lines = [
        "# Spatula Simulation-only Realtime Factor",
        "",
        f"Optimized build; {args.simulation_sec:g} simulated seconds; "
        f"{args.time_step:g}-second discrete period; "
        f"{args.repetitions} measured repetitions after {args.warmups} "
        "warm-up repetition(s).",
        "",
        "| Case | Bubble resolution | Handle resolution | Elements | "
        "Median wall time | Median RTF | RTF range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['bubble_resolution_mm']:g} mm | "
            f"{row['handle_resolution_mm']:g} mm | "
            f"{row['total_elements']:,} {row['element_name']} | "
            f"{row['median_wall_time_s']:.6f} s | "
            f"{row['median_rtf']:.6f} | "
            f"{row['min_rtf']:.6f}-{row['max_rtf']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = _parse_args()
    root = _repo_root()
    if not args.skip_build:
        _build(root)
    binary = root / _BINARY
    if not binary.is_file():
        raise RuntimeError(f"Built binary is missing: {binary}")

    for warmup in range(args.warmups):
        for case in _CASES:
            _run_case(args, root, binary, case, f"warm-up {warmup + 1}")

    measured = []
    for repetition in range(args.repetitions):
        # Rotate the order to avoid always assigning the same thermal state to
        # a case while keeping the order deterministic.
        offset = repetition % len(_CASES)
        ordered_cases = _CASES[offset:] + _CASES[:offset]
        for case in ordered_cases:
            timing = _run_case(
                args,
                root,
                binary,
                case,
                f"repetition {repetition + 1}",
            )
            measured.append(
                {
                    "case": case.name,
                    "representation": case.representation,
                    "scale": case.scale,
                    "bubble_resolution_mm": case.bubble_resolution_mm,
                    "handle_resolution_mm": case.handle_resolution_mm,
                    "element_name": case.element_name,
                    "total_elements": case.total_elements,
                    "repetition": repetition + 1,
                    **timing,
                }
            )

    output_dir = _absolute(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "spatula_rtf_runs.csv"
    summary_path = output_dir / "spatula_rtf_summary.md"
    _write_runs(runs_path, measured)
    summary_rows = _summary(_CASES, measured)
    _write_summary(summary_path, args, summary_rows)
    print(f"runs_csv: {runs_path}", flush=True)
    print(f"summary_md: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
