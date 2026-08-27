#!/usr/bin/env python3
"""Runs the bounded dynamic spatula-slip refinement tiers serially.

The trajectory truth is one tet simulation at resolution scale 0.125. The
three study tiers are fixed at 1.0, 0.5, and 0.25, and each contains tet,
plane_clip, and marching_cubes. The next 0.125 three-way tier is deliberately
not a study tier: the frozen-scene second-order forecast leaves only 1.2x
margin over the measured 10x reference-floor limit there. At 0.0625 the
forecast is below the limit.

Wall time and peak RSS include scene construction, simulation, and CSV output.
They are end-to-end costs, not the simulation-only RTF benchmark reserved for
extraction_timing.
"""

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
import json
import math
import pathlib
import subprocess
import sys


REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
REFERENCE_SCALE = 0.125
FORCE_REFERENCE_FLOOR = 8.9347e-5
AREA_REFERENCE_FLOOR = 7.2653e-5
FORCE_PRACTICAL_LIMIT = 10.0 * FORCE_REFERENCE_FLOOR
AREA_PRACTICAL_LIMIT = 10.0 * AREA_REFERENCE_FLOOR
FINEST_STATIC_MC_FORCE_ERROR = 0.004407
TIME_FORMAT = "wall_s=%e\nmax_rss_kib=%M\nexit_code=%x"
# The scene's baseline grip. It is deliberately marginal: the spatula slides
# through the fingers for the whole run in every representation, which is what
# makes the scene a slip test rather than a hold test.
DEFAULT_GRIPPER_FORCE_N = 1.5


@dataclass(frozen=True)
class Tier:
    name: str
    scale: float

    @property
    def static_floor_margin(self) -> float:
        forecast_error = FINEST_STATIC_MC_FORCE_ERROR * (self.scale / 0.25) ** 2
        return forecast_error / FORCE_PRACTICAL_LIMIT


TIERS = (
    Tier("coarse", 1.0),
    Tier("fine", 0.5),
    Tier("finer", 0.25),
)


@dataclass(frozen=True)
class Case:
    label: str
    tier: str
    representation: str
    scale: float
    is_reference: bool = False
    gripper_force: float = DEFAULT_GRIPPER_FORCE_N


def _label(value: float) -> str:
    """A filename-safe label for a numeric parameter, as 1.25 -> 1p25."""
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _expected_rows(duration: float, sample_period: float) -> int:
    return round(duration / sample_period) + 1


def _case_paths(
    output_dir: pathlib.Path, case: Case
) -> tuple[pathlib.Path, pathlib.Path]:
    return (
        output_dir / "trajectories" / f"{case.label}.csv",
        output_dir / "timing" / f"{case.label}.json",
    )


def _load_timing(path: pathlib.Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    return {
        "wall_s": float(raw["wall_s"]),
        "max_rss_kib": float(raw["max_rss_kib"]),
    }


def _trajectory_is_complete(
    path: pathlib.Path,
    case: Case,
    duration: float,
    sample_period: float,
) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != _expected_rows(duration, sample_period):
        return False
    return all(
        row["representation"] == case.representation
        and math.isclose(
            float(row["resolution_scale"]),
            case.scale,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for row in rows
    )


def _parse_time_output(path: pathlib.Path) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            key, value = line.strip().split("=", maxsplit=1)
            values[key] = value
    if values.get("exit_code") != "0":
        raise RuntimeError(f"time reported exit code {values.get('exit_code')}")
    return {
        "wall_s": float(values["wall_s"]),
        "max_rss_kib": float(values["max_rss_kib"]),
    }


def _run_case(
    binary: pathlib.Path,
    output_dir: pathlib.Path,
    case: Case,
    duration: float,
    time_step: float,
    sample_period: float,
    reuse_existing: bool,
) -> tuple[pathlib.Path, dict[str, float]]:
    csv_path, timing_path = _case_paths(output_dir, case)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        reuse_existing
        and timing_path.is_file()
        and _trajectory_is_complete(csv_path, case, duration, sample_period)
    ):
        timing = _load_timing(timing_path)
        print(f"REUSE {case.label:28s} -> {csv_path.name}")
        return csv_path, timing

    raw_timing_path = timing_path.with_suffix(".time")
    command = [
        "/usr/bin/time",
        "-f",
        TIME_FORMAT,
        "-o",
        str(raw_timing_path),
        str(binary),
        f"--representation={case.representation}",
        f"--resolution_scale={case.scale:.17g}",
        f"--duration={duration:.17g}",
        f"--time_step={time_step:.17g}",
        f"--sample_period={sample_period:.17g}",
        f"--gripper_force={case.gripper_force:.17g}",
        f"--output={csv_path}",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        raise RuntimeError(
            f"{case.label} failed with exit code {completed.returncode}"
        )
    timing = _parse_time_output(raw_timing_path)
    timing_path.write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    raw_timing_path.unlink()
    if not _trajectory_is_complete(csv_path, case, duration, sample_period):
        raise RuntimeError(f"{case.label} wrote an incomplete trajectory")
    print(
        f"PASS  {case.label:28s} {timing['wall_s']:9.2f} s "
        f"{timing['max_rss_kib'] / 1024.0:9.1f} MiB -> {csv_path.name}"
    )
    if completed.stdout.strip():
        print(f"  {completed.stdout.strip()}")
    return csv_path, timing


# Every other column is parsed as a float and checked finite, so a column
# holding text has to be named here or the read fails on it.
_STRING_FIELDS = {
    "git_commit",
    "git_dirty",
    "representation",
    "sap_converged",
}


def _read_trajectory(path: pathlib.Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for raw in reader:
            row: dict[str, float | str] = {}
            for field, value in raw.items():
                if field in _STRING_FIELDS:
                    row[field] = value
                else:
                    number = float(value)
                    if not math.isfinite(number):
                        raise RuntimeError(
                            f"{path} contains non-finite {field}"
                        )
                    row[field] = number
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{path} contains no samples")
    return rows


def _vector(
    row: dict[str, float | str], fields: tuple[str, ...]
) -> tuple[float, ...]:
    return tuple(float(row[field]) for field in fields)


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _subtract(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _quaternion_angle(
    row: dict[str, float | str], reference: dict[str, float | str]
) -> float:
    fields = ("qw", "qx", "qy", "qz")
    q = _vector(row, fields)
    q_ref = _vector(reference, fields)
    cosine = abs(sum(a * b for a, b in zip(q, q_ref, strict=True)))
    cosine /= _norm(q) * _norm(q_ref)
    return 2.0 * math.acos(max(-1.0, min(1.0, cosine)))


def _handle_axis_swing(rows: list[dict[str, float | str]]) -> list[float]:
    fields = ("handle_axis_x", "handle_axis_y", "handle_axis_z")
    initial = _vector(rows[0], fields)
    swings = []
    for row in rows:
        axis = _vector(row, fields)
        cosine = sum(a * b for a, b in zip(initial, axis, strict=True))
        cosine /= _norm(initial) * _norm(axis)
        swings.append(math.acos(max(-1.0, min(1.0, cosine))))
    return swings


def _paired_channel_metrics(
    rows: list[dict[str, float | str]],
    reference: list[dict[str, float | str]],
    left_fields: tuple[str, ...],
    right_fields: tuple[str, ...],
) -> tuple[float, float, float]:
    errors = []
    reference_values = []
    maximum = 0.0
    for row, ref in zip(rows, reference, strict=True):
        for fields in (left_fields, right_fields):
            value = _vector(row, fields)
            ref_value = _vector(ref, fields)
            error = _norm(_subtract(value, ref_value))
            errors.append(error)
            reference_values.append(_norm(ref_value))
            maximum = max(maximum, error)
    absolute_rms = _rms(errors)
    reference_rms = _rms(reference_values)
    if reference_rms == 0.0:
        raise RuntimeError("Reference channel has zero RMS magnitude")
    return absolute_rms, maximum, absolute_rms / reference_rms


def _compare(
    case: Case,
    rows: list[dict[str, float | str]],
    reference: list[dict[str, float | str]],
) -> dict[str, float | str | bool]:
    """Trajectory error against the reference, plus the task outcome.

    A run that dropped the spatula reports no trajectory error. Its position
    diverges as free fall, so an RMS against a reference measures the length of
    the fall: the coarse affine run reads 47.7 m, which is not an accuracy and
    must not enter a table or a fit as though it were one. The outcome fields
    stay populated, because "dropped at 23.3 s" is the finding.
    """
    if len(rows) != len(reference):
        raise RuntimeError(
            f"{case.label} has {len(rows)} samples; reference has {len(reference)}"
        )
    for index, (row, ref) in enumerate(zip(rows, reference, strict=True)):
        if not math.isclose(
            float(row["time_s"]),
            float(ref["time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"{case.label} time mismatch at sample {index}")

    position_errors = [
        _norm(
            _subtract(
                _vector(row, ("x_m", "y_m", "z_m")),
                _vector(ref, ("x_m", "y_m", "z_m")),
            )
        )
        for row, ref in zip(rows, reference, strict=True)
    ]
    orientation_errors = [
        _quaternion_angle(row, ref)
        for row, ref in zip(rows, reference, strict=True)
    ]
    finger_errors = [
        abs(float(row[field]) - float(ref[field]))
        for row, ref in zip(rows, reference, strict=True)
        for field in ("left_finger_m", "right_finger_m")
    ]
    swings = _handle_axis_swing(rows)
    reference_swings = _handle_axis_swing(reference)
    swing_errors = [
        abs(value - ref_value)
        for value, ref_value in zip(swings, reference_swings, strict=True)
    ]

    force_rms, force_max, force_relative = _paired_channel_metrics(
        rows,
        reference,
        (
            "left_contact_force_x_n",
            "left_contact_force_y_n",
            "left_contact_force_z_n",
        ),
        (
            "right_contact_force_x_n",
            "right_contact_force_y_n",
            "right_contact_force_z_n",
        ),
    )
    torque_rms, torque_max, torque_relative = _paired_channel_metrics(
        rows,
        reference,
        ("left_handle_axis_torque_nm",),
        ("right_handle_axis_torque_nm",),
    )
    area_rms, area_max, area_relative = _paired_channel_metrics(
        rows,
        reference,
        ("left_contact_area_m2",),
        ("right_contact_area_m2",),
    )
    contact_mismatches = sum(
        int(float(row["spatula_contacts"]) != float(ref["spatula_contacts"]))
        for row, ref in zip(rows, reference, strict=True)
    )
    two_finger_samples = sum(
        int(float(row["spatula_contacts"]) == 2.0) for row in rows
    )
    force_floor_margin = force_relative / FORCE_PRACTICAL_LIMIT
    area_floor_margin = area_relative / AREA_PRACTICAL_LIMIT
    outcome = _outcome(rows)
    trajectory = {
        "position_rms_m": _rms(position_errors),
        "position_max_m": max(position_errors),
        "orientation_rms_rad": _rms(orientation_errors),
        "orientation_max_rad": max(orientation_errors),
        "handle_axis_swing_rms_error_rad": _rms(swing_errors),
        "handle_axis_swing_max_error_rad": max(swing_errors),
        "contact_force_rms_n": force_rms,
        "contact_force_max_n": force_max,
        "contact_force_relative_rms": force_relative,
        "contact_torque_rms_nm": torque_rms,
        "contact_torque_max_nm": torque_max,
        "contact_torque_relative_rms": torque_relative,
        "contact_area_rms_m2": area_rms,
        "contact_area_max_m2": area_max,
        "contact_area_relative_rms": area_relative,
        "force_floor_margin_x": force_floor_margin,
        "area_floor_margin_x": area_floor_margin,
        "floor_limited": force_floor_margin <= 1.0 or area_floor_margin <= 1.0,
    }
    if not outcome["held"]:
        trajectory = dict.fromkeys(trajectory, None)
    return {
        "tier": case.tier,
        "representation": case.representation,
        "resolution_scale": case.scale,
        "reference_resolution_scale": REFERENCE_SCALE,
        "samples": len(rows),
        **outcome,
        "reference_handle_axis_slip_m": _outcome(reference)[
            "handle_axis_slip_m"
        ],
        "peak_handle_axis_swing_rad": max(swings),
        "reference_peak_handle_axis_swing_rad": max(reference_swings),
        "finger_position_rms_m": _rms(finger_errors)
        if outcome["held"]
        else None,
        "finger_position_max_m": max(finger_errors)
        if outcome["held"]
        else None,
        **trajectory,
        "two_finger_contact_fraction": two_finger_samples / len(rows),
        "contact_count_mismatch_fraction": contact_mismatches / len(rows),
    }


def _outcome(rows: list[dict]) -> dict:
    """The task result: did the gripper hold, and how far did the spatula slide.

    Contact is lost and regained throughout normal slipping, so the reported
    loss is the start of the run of lost samples that reaches the end -- the
    one the gripper never recovered from. Slip is measured only while held and
    is projected onto the initial handle direction, so it stays about sliding
    through the fingers rather than about the spatula's own rotation.
    """
    contacts = [float(row["spatula_contacts"]) for row in rows]
    held = contacts[-1] > 0.0
    loss_time = None
    if not held:
        for row, count in zip(reversed(rows), reversed(contacts)):
            if count > 0.0:
                break
            loss_time = float(row["time_s"])
    start = _vector(rows[0], ("x_m", "y_m", "z_m"))
    axis = _vector(rows[0], ("handle_axis_x", "handle_axis_y", "handle_axis_z"))
    slip = 0.0
    for row, count in zip(rows, contacts):
        if count == 0.0:
            break
        offset = _subtract(start, _vector(row, ("x_m", "y_m", "z_m")))
        slip = max(slip, sum(a * b for a, b in zip(offset, axis, strict=True)))
    fragments = [
        (
            int(float(row[f"{side}_num_components"])),
            float(row[f"{side}_largest_component_area_fraction"]),
        )
        for row in rows
        for side in ("left", "right")
        if float(row[f"{side}_num_components"]) > 0.0
    ]
    sap = [int(float(row["sap_iters"])) for row in rows]
    line_search = [int(float(row["sap_line_search_iters"])) for row in rows]
    return {
        "held": held,
        "first_contact_loss_time_s": loss_time,
        "handle_axis_slip_m": slip,
        "max_num_components": max((n for n, _ in fragments), default=0),
        "min_largest_component_area_fraction": min(
            (f for _, f in fragments), default=0.0
        ),
        "mean_sap_iters": sum(sap) / len(sap),
        "max_sap_iters": max(sap),
        "max_sap_line_search_iters": max(line_search),
        "sap_nonconverged_samples": sum(
            int(row["sap_converged"] == "false") for row in rows
        ),
    }


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty summary {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_grip_sweep(
    binary: pathlib.Path,
    args: argparse.Namespace,
    tiers: tuple[Tier, ...],
    forces: list[float],
) -> int:
    """Finds the grip each representation needs to hold the spatula.

    The refinement study answers "how wrong is the trajectory"; this answers
    "did the task succeed", which is the question a manipulation scene is
    actually posed to ask. The two are not interchangeable: the baseline grip
    is marginal, so a representation that transmits less tangential force
    drops the object outright and its trajectory error stops meaning anything.
    """
    cases = [
        Case(
            f"grip__{tier.name}__{representation}__f_{_label(force)}n",
            tier.name,
            representation,
            tier.scale,
            gripper_force=force,
        )
        for tier in tiers
        for representation in REPRESENTATIONS
        for force in forces
    ]
    print(
        f"Running {len(cases)} grip-sweep simulations, jobs<={args.jobs}; "
        f"forces={', '.join(f'{force:g}' for force in forces)} N; tiers="
        + ", ".join(f"{tier.name}:{tier.scale:g}" for tier in tiers)
    )
    rows = []
    failures = 0

    def run(case: Case) -> dict | None:
        try:
            csv_path, timing = _run_case(
                binary,
                args.output_dir,
                case,
                args.duration,
                args.time_step,
                args.sample_period,
                args.reuse_existing,
            )
        except (OSError, RuntimeError) as error:
            print(f"FAILED {case.label}: {error}", file=sys.stderr)
            return None
        outcome = _outcome(_read_trajectory(csv_path))
        return {
            "tier": case.tier,
            "representation": case.representation,
            "resolution_scale": case.scale,
            "gripper_force_n": case.gripper_force,
            **outcome,
            # Wall time is recorded but flagged, because a sweep run wide is
            # not a cost measurement. The refinement study is the cost data.
            "wall_time_s": timing["wall_s"],
            "timing_valid": args.jobs == 1,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(run, cases):
            if row is None:
                failures += 1
                continue
            print(
                f"{'HELD  ' if row['held'] else 'DROPPED'} "
                f"{row['tier']:6s} {row['representation']:15s} "
                f"f={row['gripper_force_n']:4g} N  "
                f"slip={row['handle_axis_slip_m']:.4f} m  "
                f"comp={row['max_num_components']}"
            )
            rows.append(row)
    if not rows:
        print("No grip-sweep run produced a trajectory", file=sys.stderr)
        return 1
    rows.sort(
        key=lambda row: (
            row["tier"],
            row["representation"],
            row["gripper_force_n"],
        )
    )
    _write_csv(args.output_dir / "grip_sweep.csv", rows)
    print(f"Wrote {len(rows)} rows to {args.output_dir / 'grip_sweep.csv'}")
    _report_thresholds(rows)
    return 1 if failures else 0


def _report_thresholds(rows: list[dict]) -> None:
    """The lowest swept force at which each case held, and where it failed."""
    print("\nGrip threshold (lowest swept force that held):")
    keys = sorted({(row["tier"], row["representation"]) for row in rows})
    for tier, representation in keys:
        group = sorted(
            (
                row
                for row in rows
                if row["tier"] == tier
                and row["representation"] == representation
            ),
            key=lambda row: row["gripper_force_n"],
        )
        held = [row["gripper_force_n"] for row in group if row["held"]]
        highest_dropped = [
            row["gripper_force_n"] for row in group if not row["held"]
        ]
        # A threshold is only bracketed if some swept force failed below it;
        # otherwise the sweep never found the boundary and says so.
        threshold = f"{min(held):g} N" if held else "above every swept force"
        bracket = (
            f" (highest failure {max(highest_dropped):g} N)"
            if highest_dropped
            else " (never failed in this sweep)"
        )
        print(f"  {tier:6s} {representation:15s} {threshold}{bracket}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=_workspace_root()
        / "bazel-bin/tools/voxel_sdf_experiments/spatula_slip/spatula_slip",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_workspace_root()
        / "tools/voxel_sdf_experiments/out/spatula_slip_tiers",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--time-step", type=float, default=0.04)
    parser.add_argument("--sample-period", type=float, default=0.04)
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=[tier.name for tier in TIERS],
        default=[tier.name for tier in TIERS],
        help="Subset of the fixed tiers to run; arbitrary scales are not accepted.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--gripper-forces",
        default="",
        help="Comma-separated finger forces in N. Given, the run becomes a "
        "grip-threshold sweep instead of a refinement study: each tier and "
        "representation is run at every force and only the task outcome is "
        "reported. Trajectory error is not, because a different grip is a "
        "different trajectory and comparing it to the baseline reference "
        "would measure the force change rather than the representation.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent simulations. The refinement study stays at 1 because "
        "its wall times are the cost data for the paper and a contended time "
        "is not a measurement. The grip sweep reports pass or fail, so it can "
        "be run wide.",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if (
        args.duration <= 0.0
        or args.time_step <= 0.0
        or args.sample_period <= 0.0
    ):
        parser.error("duration and time/sample steps must be positive")
    selected_tiers = tuple(tier for tier in TIERS if tier.name in args.tiers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.gripper_forces:
        forces = [float(value) for value in args.gripper_forces.split(",")]
        if any(force <= 0.0 for force in forces):
            parser.error("gripper forces must be positive")
        return _run_grip_sweep(binary, args, selected_tiers, forces)
    if args.jobs != 1:
        parser.error(
            "the refinement study is serial; --jobs applies to the "
            "grip sweep only"
        )

    reference = Case(
        "fine_tet_reference",
        "reference",
        "tet",
        REFERENCE_SCALE,
        is_reference=True,
    )
    cases = [reference] + [
        Case(
            f"{tier.name}__{representation}",
            tier.name,
            representation,
            tier.scale,
        )
        for tier in selected_tiers
        for representation in REPRESENTATIONS
    ]
    print(
        f"Running {len(cases)} simulations SERIALly; reference=tet scale "
        f"{REFERENCE_SCALE:g}, tiers="
        + ", ".join(f"{tier.name}:{tier.scale:g}" for tier in selected_tiers)
    )
    paths = {}
    timing = {}
    for case in cases:
        try:
            paths[case.label], timing[case.label] = _run_case(
                binary,
                args.output_dir,
                case,
                args.duration,
                args.time_step,
                args.sample_period,
                args.reuse_existing,
            )
        except (OSError, RuntimeError) as error:
            print(f"FAILED {case.label}: {error}", file=sys.stderr)
            print("Completed outputs were preserved for --reuse-existing.")
            return 1

    reference_rows = _read_trajectory(paths[reference.label])
    metrics = []
    run_rows = []
    for case in cases:
        case_timing = timing[case.label]
        run_rows.append(
            {
                "label": case.label,
                "tier": case.tier,
                "representation": case.representation,
                "resolution_scale": case.scale,
                "is_reference": case.is_reference,
                "wall_time_s": case_timing["wall_s"],
                "peak_rss_mib": case_timing["max_rss_kib"] / 1024.0,
            }
        )
        if not case.is_reference:
            metrics.append(
                _compare(
                    case, _read_trajectory(paths[case.label]), reference_rows
                )
            )

    tier_rows = []
    for tier in selected_tiers:
        tier_runs = [row for row in run_rows if row["tier"] == tier.name]
        tier_metrics = [row for row in metrics if row["tier"] == tier.name]
        tier_rows.append(
            {
                "tier": tier.name,
                "resolution_scale": tier.scale,
                "ellipsoid_resolution_m": 0.04 * tier.scale,
                "cylinder_resolution_m": 0.005 * tier.scale,
                "wall_time_sum_s": sum(row["wall_time_s"] for row in tier_runs),
                "peak_rss_max_mib": max(
                    row["peak_rss_mib"] for row in tier_runs
                ),
                "static_mc_force_floor_margin_x": tier.static_floor_margin,
                "measured_min_force_floor_margin_x": min(
                    row["force_floor_margin_x"] for row in tier_metrics
                ),
                "measured_min_area_floor_margin_x": min(
                    row["area_floor_margin_x"] for row in tier_metrics
                ),
                "contact_loss_cases": sum(
                    row["two_finger_contact_fraction"] < 1.0
                    for row in tier_metrics
                ),
                "max_contact_count_mismatch_fraction": max(
                    row["contact_count_mismatch_fraction"]
                    for row in tier_metrics
                ),
                "floor_limited": any(
                    row["floor_limited"] for row in tier_metrics
                ),
            }
        )

    _write_csv(args.output_dir / "runs.csv", run_rows)
    _write_csv(args.output_dir / "trajectory_metrics.csv", metrics)
    _write_csv(args.output_dir / "tier_summary.csv", tier_rows)
    reference_margin = (
        FINEST_STATIC_MC_FORCE_ERROR
        * (REFERENCE_SCALE / 0.25) ** 2
        / FORCE_PRACTICAL_LIMIT
    )
    reference_timing = timing[reference.label]
    print(
        f"REFERENCE: wall={reference_timing['wall_s']:.2f} s, peak RSS="
        f"{reference_timing['max_rss_kib'] / 1024.0:.1f} MiB."
    )
    print(
        "Reference static-floor forecast margin: "
        f"{reference_margin:.3f}x (marginal, not floor-limited)."
    )
    for row in tier_rows:
        if row["floor_limited"]:
            warning = " FLOOR-LIMITED"
        elif row["measured_min_force_floor_margin_x"] < 2.0:
            warning = " NEAR-FLOOR"
        else:
            warning = ""
        print(
            f"TIER {row['tier']:6s}: wall={row['wall_time_sum_s']:.2f} s, "
            f"peak RSS={row['peak_rss_max_mib']:.1f} MiB, "
            f"static margin={row['static_mc_force_floor_margin_x']:.2f}x, "
            f"measured force/area margins="
            f"{row['measured_min_force_floor_margin_x']:.2f}x/"
            f"{row['measured_min_area_floor_margin_x']:.2f}x{warning}"
        )
    for row in metrics:
        if row["two_finger_contact_fraction"] < 1.0:
            print(
                "CONTACT LOSS: "
                f"{row['tier']} {row['representation']} retained both "
                f"contacts for {row['two_finger_contact_fraction']:.1%} "
                "of samples."
            )
    print(f"Wrote summaries under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
