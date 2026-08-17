#!/usr/bin/env python3
"""Runs the cheapest end-to-end smoke case for every voxel-SDF study."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time

REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
PROJECTED_TOTAL_SECONDS = 288.0
BUILD_TIMEOUT_SECONDS = 600.0
ANALYZER_TIMEOUT_SECONDS = 60.0
SCOPE_NOTE = (
    "Scope reduction: settling runs sphere_sphere only to keep the whole "
    "serial smoke under 10 minutes; all three representations remain covered."
)
SURFACE_STUDIES = ("frozen_surface", "frozen_spatula")
SOLVER_STUDIES = ("settling", "disk_farkas", "spatula_slip")

FROZEN_SURFACE_HEADER = (
    "schema_version,git_commit,git_dirty,scene,representation,penetration_m,"
    "voxel_width_m,tet_resolution_hint_m,radius_m,box_half_width_x_m,"
    "box_half_width_y_m,box_half_width_z_m,modulus_a_pa,modulus_b_pa,"
    "grid_roll_deg,grid_pitch_deg,grid_yaw_deg,surface_distance_rms_m,"
    "surface_distance_max_m,normal_force_n,reference_force_n,"
    "normal_force_relative_error,pressure_error_rms_pa,"
    "pressure_error_max_pa,peak_pressure_pa,reference_peak_pressure_pa,"
    "peak_pressure_relative_error,projected_area_m2,reference_area_m2,"
    "area_relative_error,patch_radius_m,reference_patch_radius_m,"
    "patch_radius_relative_error,num_faces,num_vertices,centroid_x_m,"
    "centroid_y_m,centroid_z_m,centroid_position_error_m,"
    "largest_component_area_fraction"
).split(",")

SETTLING_HEADER = (
    "schema_version,git_commit,git_dirty,scene,representation,mass_kg,"
    "voxel_width_m,tet_resolution_hint_m,hydroelastic_modulus_pa,"
    "initial_gap_m,time_step_s,dissipation_s_m,duration_input_s,"
    "settling_window_input_s,grid_roll_deg,grid_pitch_deg,grid_yaw_deg,"
    "weight_N,analytic_equilibrium_penetration_m,contact_stiffness_N_m,"
    "natural_period_s,duration_s,settling_window_s,patch_radius_m,"
    "elements_across_patch,contact_plane_voxel_phase,"
    "equilibrium_penetration_m,penetration_error_m,"
    "penetration_relative_error,mean_support_force_N,"
    "mean_support_force_relative_error,penetration_span_m,"
    "max_abs_axial_velocity_m_s,max_lateral_offset_m,"
    "max_angular_speed_rad_s,mean_faces,mean_contact_area_m2,"
    "largest_component_area_fraction,max_penetration_m,"
    "first_contact_time_s,first_contact_loss_time_s,steps,wall_time_s,"
    "axial_velocity_settled,penetration_span_settled,settled"
).split(",")

DISK_FARKAS_HEADER = (
    "schema_version,git_commit,git_dirty,representation,resolution_m,"
    "time_step_s,settle_time_step_s,time_s,x_m,y_m,z_m,qx,qy,qz,qw,"
    "wx_rad_s,wy_rad_s,wz_rad_s,vx_m_s,vy_m_s,vz_m_s,"
    "angular_speed_rad_s,linear_speed_m_s,point_contacts,hydro_contacts,"
    "contact_force_x_N,contact_force_y_N,contact_force_z_N,contact_area_m2,"
    "surface_vertices,surface_faces,normal_force_z_N,friction_force_x_N,"
    "friction_force_y_N,friction_torque_z_Nm,eps,post_kick"
).split(",")

FROZEN_SPATULA_HEADER = (
    "schema_version,git_commit,git_dirty,pose,requested_penetration_m,"
    "demo_directional_penetration_m,realized_directional_penetration_m,"
    "rigid_signed_distance_m,resolution_scale,ellipsoid_resolution_m,"
    "cylinder_resolution_m,fine_tet_resolution_hint_m,representation,"
    "reference_available,in_contact,is_triangle,num_faces,num_vertices,"
    "total_surface_area_m2,projected_area_m2,reference_projected_area_m2,"
    "area_relative_error,pressure_integral_n,force_norm_n,reference_force_n,"
    "force_relative_error,normal_force_n,transverse_force_n,min_pressure_pa,"
    "max_pressure_pa,surface_distance_rms_m,surface_distance_max_m,"
    "pressure_error_rms_pa,pressure_error_max_pa,centroid_x_m,centroid_y_m,"
    "centroid_z_m,reference_centroid_x_m,reference_centroid_y_m,"
    "reference_centroid_z_m,centroid_position_error_m,connected_components,"
    "largest_component_area_fraction,has_nonfinite,has_negative_pressure,"
    "reference_construction_wall_s"
).split(",")

SPATULA_SLIP_HEADER = (
    "schema_version,git_commit,git_dirty,representation,resolution_scale,"
    "ellipsoid_resolution_m,cylinder_resolution_m,time_step_s,"
    "sample_period_s,time_s,x_m,y_m,z_m,qw,qx,qy,qz,handle_axis_x,"
    "handle_axis_y,handle_axis_z,left_finger_m,right_finger_m,"
    "point_contacts,hydro_contacts,spatula_contacts,left_contact_force_x_n,"
    "left_contact_force_y_n,left_contact_force_z_n,"
    "left_handle_axis_torque_nm,left_contact_area_m2,left_surface_faces,"
    "right_contact_force_x_n,right_contact_force_y_n,"
    "right_contact_force_z_n,right_handle_axis_torque_nm,"
    "right_contact_area_m2,right_surface_faces"
).split(",")

SPATULA_SLIP_RUNS_HEADER = (
    "label,tier,representation,resolution_scale,is_reference,wall_time_s,"
    "peak_rss_mib"
).split(",")

SPATULA_SLIP_METRICS_HEADER = (
    "tier,representation,resolution_scale,reference_resolution_scale,samples,"
    "position_rms_m,position_max_m,orientation_rms_rad,orientation_max_rad,"
    "handle_axis_swing_rms_error_rad,handle_axis_swing_max_error_rad,"
    "peak_handle_axis_swing_rad,reference_peak_handle_axis_swing_rad,"
    "finger_position_rms_m,finger_position_max_m,contact_force_rms_n,"
    "contact_force_max_n,contact_force_relative_rms,contact_torque_rms_nm,"
    "contact_torque_max_nm,contact_torque_relative_rms,contact_area_rms_m2,"
    "contact_area_max_m2,contact_area_relative_rms,"
    "two_finger_contact_fraction,contact_count_mismatch_fraction,"
    "force_floor_margin_x,area_floor_margin_x,floor_limited"
).split(",")

SPATULA_SLIP_TIER_HEADER = (
    "tier,resolution_scale,ellipsoid_resolution_m,cylinder_resolution_m,"
    "wall_time_sum_s,peak_rss_max_mib,static_mc_force_floor_margin_x,"
    "measured_min_force_floor_margin_x,measured_min_area_floor_margin_x,"
    "contact_loss_cases,max_contact_count_mismatch_fraction,floor_limited"
).split(",")


@dataclass(frozen=True)
class Study:
    name: str
    command: tuple[str, ...]
    output_dir: pathlib.Path
    timeout_seconds: float
    projected_seconds: float


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    wall_time_seconds: float
    output: str
    timed_out: bool


@dataclass
class ValidationResult:
    row_counts: dict[str, int]
    errors: dict[str, list[str]]


@dataclass
class TableRow:
    study: str
    representation: str
    wall_time_seconds: float
    row_count: int
    status: str


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def _format_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def _run_command(
    command: tuple[str, ...], timeout_seconds: float
) -> CommandResult:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=_workspace_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return CommandResult(
            process.returncode,
            time.monotonic() - started,
            output,
            False,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
        return CommandResult(
            None,
            time.monotonic() - started,
            output,
            True,
        )


def _read_csv(
    path: pathlib.Path, expected_header: list[str]
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected_header:
                raise RuntimeError(f"unexpected header in {path}")
            rows = list(reader)
    except OSError as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows


def _has_finite_row(
    rows: list[dict[str, str]], fields: tuple[str, ...]
) -> bool:
    for row in rows:
        try:
            if all(math.isfinite(float(row[field])) for field in fields):
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _empty_validation() -> ValidationResult:
    return ValidationResult(
        row_counts={representation: 0 for representation in REPRESENTATIONS},
        errors={representation: [] for representation in REPRESENTATIONS},
    )


def _record_rows(
    result: ValidationResult,
    path: pathlib.Path,
    expected_header: list[str],
    primary_fields: tuple[str, ...],
    expected_representation: str,
) -> None:
    try:
        rows = _read_csv(path, expected_header)
        if any(
            row.get("schema_version") != "1"
            or row.get("representation") != expected_representation
            for row in rows
        ):
            raise RuntimeError(f"metadata mismatch in {path}")
        if not _has_finite_row(rows, primary_fields):
            raise RuntimeError(f"no row has finite {primary_fields} in {path}")
        result.row_counts[expected_representation] += len(rows)
    except RuntimeError as error:
        result.errors[expected_representation].append(str(error))


def _validate_frozen_surface(output_dir: pathlib.Path) -> ValidationResult:
    result = _empty_validation()
    for scene in ("sphere_sphere", "sphere_box"):
        for representation in REPRESENTATIONS:
            path = output_dir / f"{scene}__{representation}__h_10mm.csv"
            _record_rows(
                result,
                path,
                FROZEN_SURFACE_HEADER,
                ("normal_force_n", "projected_area_m2", "num_faces"),
                representation,
            )
    return result


def _validate_settling(output_dir: pathlib.Path) -> ValidationResult:
    result = _empty_validation()
    all_rows = 0
    for scene in ("sphere_sphere",):
        for representation in REPRESENTATIONS:
            path = output_dir / f"{scene}__{representation}__h_10mm.csv"
            before = result.row_counts[representation]
            _record_rows(
                result,
                path,
                SETTLING_HEADER,
                (
                    "equilibrium_penetration_m",
                    "mean_support_force_N",
                    "wall_time_s",
                ),
                representation,
            )
            all_rows += result.row_counts[representation] - before
    try:
        summary_rows = _read_csv(output_dir / "summary.csv", SETTLING_HEADER)
        if len(summary_rows) != all_rows:
            raise RuntimeError(
                f"settling summary has {len(summary_rows)} rows; "
                f"expected {all_rows}"
            )
    except RuntimeError as error:
        for representation in REPRESENTATIONS:
            result.errors[representation].append(str(error))
    return result


def _validate_disk_farkas(output_dir: pathlib.Path) -> ValidationResult:
    result = _empty_validation()
    for representation in REPRESENTATIONS:
        path = output_dir / f"{representation}__h_5mm.csv"
        _record_rows(
            result,
            path,
            DISK_FARKAS_HEADER,
            ("time_s", "x_m", "angular_speed_rad_s", "normal_force_z_N"),
            representation,
        )
    return result


def _validate_frozen_spatula(output_dir: pathlib.Path) -> ValidationResult:
    result = _empty_validation()
    paths = (
        output_dir / "cases/demo__delta_0mm__scale_1.csv",
        output_dir / "cases/first_touch__delta_0mm__scale_1.csv",
    )
    expected_total_rows = 0
    for path in paths:
        try:
            rows = _read_csv(path, FROZEN_SPATULA_HEADER)
        except RuntimeError as error:
            for representation in REPRESENTATIONS:
                result.errors[representation].append(str(error))
            continue
        if len(rows) != len(REPRESENTATIONS):
            error = f"{path} has {len(rows)} rows; expected 3"
            for representation in REPRESENTATIONS:
                result.errors[representation].append(error)
            continue
        expected_total_rows += len(rows)
        for representation in REPRESENTATIONS:
            matches = [
                row
                for row in rows
                if row.get("schema_version") == "1"
                and row.get("representation") == representation
            ]
            if len(matches) != 1:
                result.errors[representation].append(
                    f"{path} has {len(matches)} rows for {representation}"
                )
            elif not _has_finite_row(
                matches,
                ("force_norm_n", "projected_area_m2", "num_faces"),
            ):
                result.errors[representation].append(
                    f"no primary finite row for {representation} in {path}"
                )
            else:
                result.row_counts[representation] += 1
    try:
        summary_rows = _read_csv(
            output_dir / "summary.csv", FROZEN_SPATULA_HEADER
        )
        if len(summary_rows) != expected_total_rows:
            raise RuntimeError(
                f"frozen_spatula summary has {len(summary_rows)} rows; "
                f"expected {expected_total_rows}"
            )
    except RuntimeError as error:
        for representation in REPRESENTATIONS:
            result.errors[representation].append(str(error))
    return result


def _validate_spatula_slip(output_dir: pathlib.Path) -> ValidationResult:
    result = _empty_validation()
    for representation in REPRESENTATIONS:
        path = output_dir / "trajectories" / f"coarse__{representation}.csv"
        _record_rows(
            result,
            path,
            SPATULA_SLIP_HEADER,
            ("time_s", "x_m", "left_contact_force_y_n", "left_contact_area_m2"),
            representation,
        )
    supplemental = (
        (
            output_dir / "trajectories/fine_tet_reference.csv",
            SPATULA_SLIP_HEADER,
            1,
        ),
        (output_dir / "runs.csv", SPATULA_SLIP_RUNS_HEADER, 4),
        (
            output_dir / "trajectory_metrics.csv",
            SPATULA_SLIP_METRICS_HEADER,
            3,
        ),
        (output_dir / "tier_summary.csv", SPATULA_SLIP_TIER_HEADER, 1),
    )
    for path, header, minimum_rows in supplemental:
        try:
            rows = _read_csv(path, header)
            if len(rows) < minimum_rows:
                raise RuntimeError(
                    f"{path} has {len(rows)} rows; expected at least "
                    f"{minimum_rows}"
                )
            if path.name == "fine_tet_reference.csv" and not _has_finite_row(
                rows, ("time_s", "x_m", "left_contact_force_y_n")
            ):
                raise RuntimeError(f"no primary finite row in {path}")
        except RuntimeError as error:
            for representation in REPRESENTATIONS:
                result.errors[representation].append(str(error))
    return result


def _validate(study: str, output_dir: pathlib.Path) -> ValidationResult:
    validators = {
        "frozen_surface": _validate_frozen_surface,
        "settling": _validate_settling,
        "disk_farkas": _validate_disk_farkas,
        "frozen_spatula": _validate_frozen_spatula,
        "spatula_slip": _validate_spatula_slip,
    }
    return validators[study](output_dir)


def _make_studies(run_root: pathlib.Path) -> tuple[Study, ...]:
    root = _workspace_root()
    experiments = root / "tools/voxel_sdf_experiments"
    python = sys.executable
    return (
        Study(
            "frozen_surface",
            (
                python,
                str(experiments / "frozen_surface/run_ladder.py"),
                "--rungs_mm",
                "10",
                "--penetration",
                "0.0199",
                "--output-dir",
                str(run_root / "frozen_surface"),
                "--jobs",
                "1",
            ),
            run_root / "frozen_surface",
            60.0,
            12.0,
        ),
        Study(
            "settling",
            (
                python,
                str(experiments / "settling/run_ladder.py"),
                "--rungs_mm",
                "10",
                "--jobs",
                "1",
                "--scene",
                "sphere_sphere",
                "--output-dir",
                str(run_root / "settling"),
            ),
            run_root / "settling",
            300.0,
            180.0,
        ),
        Study(
            "disk_farkas",
            (
                python,
                str(experiments / "disk_farkas/run_ladder.py"),
                "--rungs_mm",
                "5",
                "--output-dir",
                str(run_root / "disk_farkas"),
            ),
            run_root / "disk_farkas",
            180.0,
            36.0,
        ),
        Study(
            "frozen_spatula",
            (
                python,
                str(experiments / "frozen_spatula/run_sweep.py"),
                "--resolution-scales",
                "1",
                "--max-penetration-mm",
                "0",
                "--penetration-step-mm",
                "1",
                "--fine-tet-resolution-hint",
                "0.005",
                "--jobs",
                "1",
                "--output-dir",
                str(run_root / "frozen_spatula"),
            ),
            run_root / "frozen_spatula",
            120.0,
            6.0,
        ),
        Study(
            "spatula_slip",
            (
                python,
                str(experiments / "spatula_slip/run_tiers.py"),
                "--tiers",
                "coarse",
                "--output-dir",
                str(run_root / "spatula_slip"),
            ),
            run_root / "spatula_slip",
            180.0,
            54.0,
        ),
    )


def _build_command(
    bazel_output_user_root: pathlib.Path | None,
) -> tuple[str, ...]:
    prefix = "//tools/voxel_sdf_experiments"
    command = ["bazel"]
    if bazel_output_user_root is not None:
        command.extend(
            ("--batch", f"--output_user_root={bazel_output_user_root}")
        )
    command.append("build")
    if bazel_output_user_root is not None:
        command.append("--experimental_collect_system_network_usage=false")
    command.extend(
        (
            f"{prefix}/frozen_surface:frozen_surface",
            f"{prefix}/settling:settling",
            f"{prefix}/disk_farkas:disk_farkas",
            f"{prefix}/frozen_spatula:frozen_spatula",
            f"{prefix}/spatula_slip:spatula_slip",
        )
    )
    return tuple(command)


def _analyzer_command(study: str, output_dir: pathlib.Path) -> tuple[str, ...]:
    experiments = _workspace_root() / "tools/voxel_sdf_experiments"
    if study in SURFACE_STUDIES:
        analyzer = experiments / "analyze_surface.py"
    elif study in SOLVER_STUDIES:
        analyzer = experiments / "analyze_solver.py"
    else:
        raise ValueError(f"unknown study: {study}")
    return (
        sys.executable,
        str(analyzer),
        study,
        str(output_dir),
        "--output-dir",
        str(output_dir / "analysis"),
    )


def _print_plan(
    studies: tuple[Study, ...],
    run_root: pathlib.Path,
    bazel_output_user_root: pathlib.Path | None,
) -> None:
    projected_minutes = PROJECTED_TOTAL_SECONDS / 60.0
    print(f"Smoke output: {run_root}")
    print("Coverage: all 3 representations in all 5 studies.")
    print(SCOPE_NOTE)
    print(
        f"Projected serial study time: {projected_minutes:.1f} min "
        "(incremental build and analyzer excluded)."
    )
    print("Disk Farkas remains serial by driver contract.")
    print("\nWould run:")
    print(f"  build ({BUILD_TIMEOUT_SECONDS:.0f} s timeout)")
    print(f"    {_format_command(_build_command(bazel_output_user_root))}")
    for study in studies:
        print(
            f"  {study.name} (projected {study.projected_seconds:.0f} s; "
            f"timeout {study.timeout_seconds:.0f} s)"
        )
        print(f"    {_format_command(study.command)}")
        analyzer = _analyzer_command(study.name, study.output_dir)
        print(
            f"  {study.name} analyzer "
            f"({ANALYZER_TIMEOUT_SECONDS:.0f} s timeout)"
        )
        print(f"    {_format_command(analyzer)}")
    print("\nAnalyzer policy:")
    print("  analyze_surface.py  frozen_surface, frozen_spatula (required)")
    print(
        "  analyze_solver.py   settling, disk_farkas, spatula_slip (required)"
    )
    print(
        "  Solver contact loss is reported and excluded from fits; it is a "
        "study result, not an analyzer failure."
    )


def _print_table(rows: list[TableRow]) -> None:
    headers = ("study", "representation", "wall time", "row count", "status")
    values = [
        (
            row.study,
            row.representation,
            f"{row.wall_time_seconds:.1f} s",
            str(row.row_count),
            row.status,
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(
        "  ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _build_failure_rows(result: CommandResult) -> list[TableRow]:
    status = "TIMEOUT" if result.timed_out else "FAIL (build)"
    return [
        TableRow(study, representation, 0.0, 0, status)
        for study in (
            "frozen_surface",
            "settling",
            "disk_farkas",
            "frozen_spatula",
            "spatula_slip",
        )
        for representation in REPRESENTATIONS
    ]


def _run_analyzer(
    study: str, output_dir: pathlib.Path, details: list[str]
) -> tuple[str, bool]:
    command = _analyzer_command(study, output_dir)
    analyzer = pathlib.Path(command[1])
    if not analyzer.is_file():
        details.append(f"{study} analyzer is absent: {analyzer}")
        return "FAIL (absent)", False
    result = _run_command(command, ANALYZER_TIMEOUT_SECONDS)
    (output_dir / "analyzer.log").write_text(result.output, encoding="utf-8")
    if result.timed_out:
        details.append(f"{study} analyzer timed out")
        return "TIMEOUT", False
    if result.returncode != 0:
        details.append(
            f"{study} analyzer exited {result.returncode}; see analyzer.log"
        )
        return f"FAIL (exit {result.returncode})", False
    return f"PASS ({result.wall_time_seconds:.1f} s)", True


def _run_all(
    studies: tuple[Study, ...],
    run_root: pathlib.Path,
    bazel_output_user_root: pathlib.Path | None,
) -> int:
    total_started = time.monotonic()
    run_root.mkdir(parents=True)
    details = []
    rows = []
    analyzer_status = {
        "frozen_surface": "NOT RUN",
        "settling": "NOT RUN",
        "disk_farkas": "NOT RUN",
        "frozen_spatula": "NOT RUN",
        "spatula_slip": "NOT RUN",
    }

    print(f"Output: {run_root}")
    print("Coverage: all 3 representations in all 5 studies.")
    print(SCOPE_NOTE)
    print("Building the five driver prerequisites...")
    build_result = _run_command(
        _build_command(bazel_output_user_root), BUILD_TIMEOUT_SECONDS
    )
    (run_root / "build.log").write_text(build_result.output, encoding="utf-8")
    if build_result.timed_out or build_result.returncode != 0:
        rows = _build_failure_rows(build_result)
        if build_result.timed_out:
            details.append("Bazel build timed out; see build.log")
        else:
            details.append(
                f"Bazel build exited {build_result.returncode}; see build.log"
            )
        print("\nPass/fail table")
        _print_table(rows)
        print("\nAnalyzer status")
        for study, status in analyzer_status.items():
            print(f"  {study:15s} {status}")
        print("\nFailures")
        for detail in details:
            print(f"  {detail}")
        print(f"\nTotal wall time: {time.monotonic() - total_started:.1f} s")
        return 1
    print(f"Build passed in {build_result.wall_time_seconds:.1f} s.")

    for study in studies:
        study.output_dir.mkdir(parents=True)
        print(
            f"Running {study.name} (timeout {study.timeout_seconds:.0f} s)...",
            flush=True,
        )
        command_result = _run_command(study.command, study.timeout_seconds)
        (study.output_dir / "driver.log").write_text(
            command_result.output, encoding="utf-8"
        )
        validation = _validate(study.name, study.output_dir)
        if command_result.timed_out:
            common_status = "TIMEOUT"
            details.append(f"{study.name} timed out; see driver.log")
        elif command_result.returncode != 0:
            common_status = f"FAIL (exit {command_result.returncode})"
            details.append(
                f"{study.name} exited {command_result.returncode}; "
                "see driver.log"
            )
        else:
            common_status = "PASS"

        analyzer_passed = True
        if common_status == "PASS":
            analyzer_status[study.name], analyzer_passed = _run_analyzer(
                study.name, study.output_dir, details
            )

        for representation in REPRESENTATIONS:
            status = common_status
            errors = validation.errors[representation]
            if status == "PASS" and errors:
                status = "FAIL (CSV)"
            if status == "PASS" and not analyzer_passed:
                status = "FAIL (analyzer)"
            rows.append(
                TableRow(
                    study.name,
                    representation,
                    command_result.wall_time_seconds,
                    validation.row_counts[representation],
                    status,
                )
            )
            details.extend(
                f"{study.name}/{representation}: {error}" for error in errors
            )
        print(
            f"{study.name}: {common_status} in "
            f"{command_result.wall_time_seconds:.1f} s"
        )

    print("\nPass/fail table")
    print(
        "Wall time is the enclosing driver invocation and is repeated for "
        "representations produced together."
    )
    _print_table(rows)
    print("\nAnalyzer status")
    for study, status in analyzer_status.items():
        print(f"  {study:15s} {status}")
    if details:
        print("\nFailure details")
        for detail in details:
            print(f"  {detail}")
    total_elapsed = time.monotonic() - total_started
    print(f"\nTotal wall time: {total_elapsed:.1f} s")
    failed = any(row.status != "PASS" for row in rows)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="Print commands, coverage, and projected cost without running.",
    )
    parser.add_argument(
        "--bazel-output-user-root",
        type=pathlib.Path,
        help="Optional writable Bazel output root, for restricted checkouts.",
    )
    args = parser.parse_args()

    output_root = _workspace_root() / "tools/voxel_sdf_experiments/out/smoke"
    run_root = output_root / _run_id()
    studies = _make_studies(run_root)
    if args.list_only:
        _print_plan(studies, run_root, args.bazel_output_user_root)
        return 0
    return _run_all(studies, run_root, args.bazel_output_user_root)


if __name__ == "__main__":
    sys.exit(main())
