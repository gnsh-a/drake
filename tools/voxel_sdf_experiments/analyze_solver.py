#!/usr/bin/env python3
"""Analyzes a solver-run voxel-SDF study, including contact loss."""

import argparse
import collections
import csv
import math
import pathlib
import sys

from common import analysis

# Settling pins the support force to mg, so force error collapses to solver
# tolerance and discretization error appears as a penetration offset. The force
# error remains in the run matrix as a red flag, but it is not fitted.
SETTLING_METRICS = (
    analysis.Metric("penetration_relative_error", 0.0, "penetration rel err"),
    # Below a femtometre the penetration offset is double-precision dust on an
    # 8 cm scene.
    analysis.Metric("penetration_error_m", 1e-15, "penetration err (m)"),
    # Drift and spin hold machine precision when the representation preserves
    # the scene symmetry.
    analysis.Metric(
        "lateral_drift_rate_m_s", 1e-12, "lateral drift rate (m/s)"
    ),
    analysis.Metric(
        "max_angular_speed_rad_s", 1e-12, "max angular speed (rad/s)"
    ),
    # Settled runs chatter at about 2e-10 m regardless of h.
    analysis.Metric("penetration_span_m", 1e-9, "penetration span (m)"),
    # This complement saturates toward one for fragmented plane-clip patches;
    # its slope would describe saturation instead of convergence order.
    analysis.Metric(
        "fragmented_area_fraction",
        0.0,
        "fragmented area fraction",
        0.5,
    ),
)
SETTLING_PRIMARY_METRIC = "penetration_relative_error"
SETTLED_GATE = analysis.RowGate(
    "settled",
    lambda row: row.get("settled", "").strip().lower() == "true",
)
DISK_TERMINAL_EPSILON = 0.653


def _is_settled(row: analysis.Row) -> bool:
    return SETTLED_GATE.predicate(row)


def _settling_metric_value(row: analysis.Row, metric_name: str) -> float:
    """Returns the magnitude of a settling metric used by a log-log fit."""
    if metric_name == "fragmented_area_fraction":
        return max(0.0, 1.0 - float(row["largest_component_area_fraction"]))
    if metric_name == "lateral_drift_rate_m_s":
        duration = float(row["duration_s"])
        if duration <= 0.0:
            raise ValueError(f"{row['_path']} has non-positive duration_s")
        return abs(float(row["max_lateral_offset_m"])) / duration
    return abs(float(row[metric_name]))


def _settling_faces_for_one_percent(
    rows: list[analysis.Row],
) -> list[dict[str, object]]:
    """Fits settled penetration error against mean contact-face count."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["representation"])].append(row)
    fits = []
    for (scene, representation), group in sorted(grouped.items()):
        samples = [
            (
                float(row["mean_faces"]),
                _settling_metric_value(row, SETTLING_PRIMARY_METRIC),
            )
            for row in group
            if _is_settled(row) and float(row["mean_faces"]) > 0.0
        ]
        samples = [pair for pair in samples if pair[1] > 0.0]
        if len(samples) < 2:
            continue
        intercept, slope, residual = analysis.linear_fit(
            [math.log(sample[0]) for sample in samples],
            [math.log(sample[1]) for sample in samples],
        )
        if slope == 0.0:
            continue
        faces = math.exp((math.log(0.01) - intercept) / slope)
        fits.append(
            {
                "scene": scene,
                "representation": representation,
                "samples": len(samples),
                "error_vs_faces_slope": slope,
                "log_residual_rms": residual,
                "faces_for_1pct_penetration_error": faces,
            }
        )
    return fits


def _analyze_settling(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    rows = analysis.read_rows(input_dir)
    order_fits, skipped_series = analysis.order_fits(
        rows,
        SETTLING_METRICS,
        _settling_metric_value,
        row_gate=SETTLED_GATE,
    )
    face_fits = _settling_faces_for_one_percent(rows)
    order_fields = (
        "scene",
        "representation",
        "metric",
        "samples",
        "settled_rungs",
        "order",
        "log_intercept",
        "log_residual_rms",
    )
    fit_rows = [
        {
            **fit,
            "settled_rungs": fit["eligible_rungs"],
        }
        for fit in order_fits
    ]
    analysis.write_csv(
        output_dir / "order_fits.csv",
        order_fields,
        [{field: fit[field] for field in order_fields} for fit in fit_rows],
    )
    analysis.write_csv(
        output_dir / "skipped_series.csv",
        ("scene", "representation", "metric", "reason", "detail"),
        skipped_series,
    )
    analysis.write_csv(
        output_dir / "faces_for_1pct.csv",
        (
            "scene",
            "representation",
            "samples",
            "error_vs_faces_slope",
            "log_residual_rms",
            "faces_for_1pct_penetration_error",
        ),
        face_fits,
    )
    summary_fields = (
        "scene",
        "representation",
        "voxel_width_m",
        "settled",
        "mean_faces",
        "elements_across_patch",
        "equilibrium_penetration_m",
        "penetration_relative_error",
        "mean_support_force_relative_error",
        "penetration_span_m",
        "max_lateral_offset_m",
        "max_angular_speed_rad_s",
        "largest_component_area_fraction",
        "contact_plane_voxel_phase",
    )
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row["scene"],
            row["representation"],
            -float(row["voxel_width_m"]),
        ),
    )
    summary_rows = [
        {field: row[field] for field in summary_fields} for row in ordered_rows
    ]
    analysis.write_csv(output_dir / "summary.csv", summary_fields, summary_rows)

    def primary_value(row: analysis.Row) -> float:
        return _settling_metric_value(row, SETTLING_PRIMARY_METRIC)

    analysis.write_svg(
        output_dir / "error_vs_h.svg",
        rows,
        "voxel_width_m",
        "h (m)",
        primary_value,
        title="Penetration relative error",
        y_label="penetration relative error",
        row_gate=SETTLED_GATE,
        hollow_note="hollow = did not settle",
    )
    analysis.write_svg(
        output_dir / "error_vs_elements.svg",
        rows,
        "mean_faces",
        "mean faces N",
        primary_value,
        title="Penetration relative error",
        y_label="penetration relative error",
        row_gate=SETTLED_GATE,
        hollow_note="hollow = did not settle",
    )

    unsettled = [row for row in ordered_rows if not _is_settled(row)]
    print("Error vs h order fits (settled rungs only)")
    analysis.print_table(
        (
            "scene",
            "representation",
            "metric",
            "n",
            "order",
            "log residual",
        ),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                str(fit["metric"]),
                str(fit["samples"]),
                f"{float(fit['order']):.4f}",
                f"{float(fit['log_residual_rms']):.4f}",
            )
            for fit in order_fits
        ],
    )
    print("\nSeries not fit, and why")
    analysis.print_table(
        ("scene", "representation", "metric", "reason", "detail"),
        [
            (
                str(entry["scene"]),
                str(entry["representation"]),
                str(entry["metric"]),
                str(entry["reason"]),
                str(entry["detail"]),
            )
            for entry in skipped_series
        ],
    )
    print("\nFaces for 1% penetration error (fit vs mean contact faces N)")
    analysis.print_table(
        ("scene", "representation", "slope", "log residual", "N at 1%"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                f"{float(fit['error_vs_faces_slope']):.4f}",
                f"{float(fit['log_residual_rms']):.4f}",
                f"{float(fit['faces_for_1pct_penetration_error']):.1f}",
            )
            for fit in face_fits
        ],
    )
    print("\nRun matrix summary")
    analysis.print_table(
        (
            "scene",
            "representation",
            "h mm",
            "settled",
            "faces",
            "pen rel err",
            "force rel err",
            "span m",
        ),
        [
            (
                row["scene"],
                row["representation"],
                f"{1000.0 * float(row['voxel_width_m']):g}",
                "yes" if _is_settled(row) else "NO",
                f"{float(row['mean_faces']):.6g}",
                f"{float(row['penetration_relative_error']):+.6g}",
                f"{float(row['mean_support_force_relative_error']):.3g}",
                f"{float(row['penetration_span_m']):.3g}",
            )
            for row in ordered_rows
        ],
    )
    if unsettled:
        print(
            f"\n{len(unsettled)} of {len(ordered_rows)} rungs did not "
            "settle and were excluded from all fits:"
        )
        for row in unsettled:
            print(
                f"  {row['scene']} {row['representation']} "
                f"h={1000.0 * float(row['voxel_width_m']):g} mm"
            )


def _read_csv_rows(path: pathlib.Path) -> list[analysis.Row]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows


def _terminal_epsilon_samples(rows: list[analysis.Row]) -> list[float]:
    """Matches disk_farkas.cc's finite-epsilon, |wz| >= 0.1 gate."""
    values = []
    for row in rows:
        try:
            value = float(row["eps"])
            spin = abs(float(row["wz_rad_s"]))
        except (KeyError, ValueError):
            continue
        if math.isfinite(value) and spin >= 0.1:
            values.append(value)
    return values


def _analyze_disk_farkas(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    paths = tuple(sorted(input_dir.glob("*.csv")))
    if not paths:
        raise RuntimeError(f"no disk trajectory CSVs found in {input_dir}")
    summary_rows = []
    exclusions = []
    for path in paths:
        rows = _read_csv_rows(path)
        if any(row.get("schema_version") != "1" for row in rows):
            raise RuntimeError(f"{path} does not use schema version 1")
        representation = rows[0].get("representation", "")
        resolution = rows[0].get("resolution_m", "")
        if not representation or not resolution:
            raise RuntimeError(
                f"{path} is missing representation or resolution"
            )
        if any(
            row.get("representation") != representation
            or row.get("resolution_m") != resolution
            for row in rows
        ):
            raise RuntimeError(f"{path} mixes representations or resolutions")
        post_kick = [
            row
            for row in rows
            if row.get("post_kick", "").strip().lower() == "true"
        ]
        if not post_kick:
            raise RuntimeError(f"{path} has no post-kick samples")
        contact = [float(row["hydro_contacts"]) > 0.0 for row in post_kick]
        contact_fraction = sum(contact) / len(contact)
        loss_rows = [
            row
            for row, in_contact in zip(post_kick, contact, strict=True)
            if not in_contact
        ]
        epsilons = _terminal_epsilon_samples(post_kick)
        if not epsilons:
            raise RuntimeError(f"{path} has no finite post-kick epsilon")
        first_loss_time = loss_rows[0]["time_s"] if loss_rows else ""
        summary_rows.append(
            {
                "representation": representation,
                "resolution_m": resolution,
                "samples": len(rows),
                "post_kick_samples": len(post_kick),
                "contact_fraction": contact_fraction,
                "lost_contact": bool(loss_rows),
                "first_contact_loss_time_s": first_loss_time,
                "initial_epsilon": epsilons[0],
                "terminal_epsilon": epsilons[-1],
                "terminal_epsilon_target": DISK_TERMINAL_EPSILON,
            }
        )
        if loss_rows:
            exclusions.append(
                {
                    "study": "disk_farkas",
                    "representation": representation,
                    "resolution": resolution,
                    "reason": "lost_contact",
                    "detail": f"{len(loss_rows)} of {len(post_kick)} "
                    "post-kick samples have no hydroelastic contact",
                }
            )
    summary_rows.sort(
        key=lambda row: (
            str(row["representation"]),
            -float(row["resolution_m"]),
        )
    )
    analysis.write_csv(
        output_dir / "summary.csv",
        (
            "representation",
            "resolution_m",
            "samples",
            "post_kick_samples",
            "contact_fraction",
            "lost_contact",
            "first_contact_loss_time_s",
            "initial_epsilon",
            "terminal_epsilon",
            "terminal_epsilon_target",
        ),
        summary_rows,
    )
    analysis.write_csv(
        output_dir / "excluded_runs.csv",
        ("study", "representation", "resolution", "reason", "detail"),
        exclusions,
    )
    print("Disk trajectory summary (no convergence fit is claimed)")
    analysis.print_table(
        (
            "representation",
            "h mm",
            "contact",
            "eps initial",
            "eps terminal",
            "physical run",
        ),
        [
            (
                str(row["representation"]),
                f"{1000.0 * float(row['resolution_m']):g}",
                f"{float(row['contact_fraction']):.1%}",
                f"{float(row['initial_epsilon']):.6g}",
                f"{float(row['terminal_epsilon']):.6g}",
                "NO (lost contact)" if row["lost_contact"] else "yes",
            )
            for row in summary_rows
        ],
    )
    if exclusions:
        print(f"\n{len(exclusions)} run(s) lost contact and are excluded.")


def _trajectory_contact_fraction(path: pathlib.Path) -> float:
    rows = _read_csv_rows(path)
    if any(row.get("schema_version") != "1" for row in rows):
        raise RuntimeError(f"{path} does not use schema version 1")
    return sum(float(row["spatula_contacts"]) == 2.0 for row in rows) / len(
        rows
    )


def _analyze_spatula_slip(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    metric_rows = _read_csv_rows(input_dir / "trajectory_metrics.csv")
    summary_rows = []
    exclusions = []
    for row in metric_rows:
        tier = row["tier"]
        representation = row["representation"]
        trajectory = (
            input_dir / "trajectories" / f"{tier}__{representation}.csv"
        )
        trajectory_fraction = _trajectory_contact_fraction(trajectory)
        recorded_fraction = float(row["two_finger_contact_fraction"])
        if not math.isclose(
            trajectory_fraction,
            recorded_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"{trajectory} contact fraction {trajectory_fraction} does "
                f"not match trajectory_metrics.csv value {recorded_fraction}"
            )
        lost_contact = trajectory_fraction < 1.0
        floor_limited = row["floor_limited"].strip().lower() == "true"
        fit_eligible = not lost_contact and not floor_limited
        summary_rows.append(
            {
                "tier": tier,
                "representation": representation,
                "resolution_scale": row["resolution_scale"],
                "samples": row["samples"],
                "position_rms_m": row["position_rms_m"],
                "peak_handle_axis_swing_rad": row["peak_handle_axis_swing_rad"],
                "contact_force_relative_rms": row["contact_force_relative_rms"],
                "two_finger_contact_fraction": row[
                    "two_finger_contact_fraction"
                ],
                "contact_count_mismatch_fraction": row[
                    "contact_count_mismatch_fraction"
                ],
                "force_floor_margin_x": row["force_floor_margin_x"],
                "area_floor_margin_x": row["area_floor_margin_x"],
                "floor_limited": floor_limited,
                "fit_eligible": fit_eligible,
            }
        )
        if lost_contact:
            exclusions.append(
                {
                    "study": "spatula_slip",
                    "representation": representation,
                    "resolution": row["resolution_scale"],
                    "reason": "lost_contact",
                    "detail": "both-finger contact held for "
                    f"{trajectory_fraction:.3%} of trajectory samples",
                }
            )
        if floor_limited:
            exclusions.append(
                {
                    "study": "spatula_slip",
                    "representation": representation,
                    "resolution": row["resolution_scale"],
                    "reason": "reference_floor_limited",
                    "detail": "the emitted force or area error is at the "
                    "practical reference floor",
                }
            )
    tier_order = {"coarse": 0, "fine": 1, "finer": 2}
    summary_rows.sort(
        key=lambda row: (
            tier_order.get(str(row["tier"]), len(tier_order)),
            str(row["representation"]),
        )
    )
    summary_fields = (
        "tier",
        "representation",
        "resolution_scale",
        "samples",
        "position_rms_m",
        "peak_handle_axis_swing_rad",
        "contact_force_relative_rms",
        "two_finger_contact_fraction",
        "contact_count_mismatch_fraction",
        "force_floor_margin_x",
        "area_floor_margin_x",
        "floor_limited",
        "fit_eligible",
    )
    analysis.write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    analysis.write_csv(
        output_dir / "excluded_runs.csv",
        ("study", "representation", "resolution", "reason", "detail"),
        exclusions,
    )
    print(
        "Spatula trajectory summary (errors are non-monotone; no convergence "
        "fit is claimed)"
    )
    analysis.print_table(
        (
            "tier",
            "representation",
            "both contact",
            "position RMS m",
            "peak swing rad",
            "force rel RMS",
            "fit eligible",
        ),
        [
            (
                str(row["tier"]),
                str(row["representation"]),
                f"{float(row['two_finger_contact_fraction']):.1%}",
                f"{float(row['position_rms_m']):.6g}",
                f"{float(row['peak_handle_axis_swing_rad']):.6g}",
                f"{float(row['contact_force_relative_rms']):.3%}",
                "yes" if row["fit_eligible"] else "NO",
            )
            for row in summary_rows
        ],
    )
    if exclusions:
        print(f"\n{len(exclusions)} explicit fit exclusion(s) reported.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study", choices=("settling", "disk_farkas", "spatula_slip")
    )
    parser.add_argument("input_dir", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Defaults to INPUT_DIR/analysis.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzers = {
        "settling": _analyze_settling,
        "disk_farkas": _analyze_disk_farkas,
        "spatula_slip": _analyze_spatula_slip,
    }
    analyzers[args.study](args.input_dir, output_dir)
    print(f"\nWrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"analyze_solver.py: {error}", file=sys.stderr)
        sys.exit(1)
