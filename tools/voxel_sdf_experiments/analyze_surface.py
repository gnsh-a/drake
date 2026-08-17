#!/usr/bin/env python3
"""Analyzes a frozen-configuration surface study."""

import argparse
import collections
import csv
import math
import pathlib
import sys

from common import analysis

FROZEN_SURFACE_METRICS = (
    # Coordinate errors below 10 femtometres on these 0.1 m scenes are
    # roundoff, not surface-discretization error.
    analysis.Metric(
        "surface_distance_rms_m", 1e-14, "surface distance RMS (m)"
    ),
    analysis.Metric("normal_force_relative_error", 0.0, "normal force rel err"),
    analysis.Metric("pressure_error_rms_pa", 0.0, "pressure error RMS (Pa)"),
    analysis.Metric(
        "peak_pressure_relative_error", 0.0, "peak pressure rel err"
    ),
    analysis.Metric("area_relative_error", 0.0, "area rel err"),
    analysis.Metric("patch_radius_relative_error", 0.0, "patch radius rel err"),
    analysis.Metric("centroid_position_error_m", 1e-14, "centroid error (m)"),
    analysis.Metric(
        "fragmented_area_fraction", 0.0, "fragmented area fraction"
    ),
)

# The frozen-spatula reference floors are Richardson estimates measured at the
# saturation rung. The 10x practical limits keep reference error from being
# mistaken for representation error.
FROZEN_SPATULA_FORCE_PRACTICAL_LIMIT = 10.0 * 8.9347e-5
FROZEN_SPATULA_AREA_PRACTICAL_LIMIT = 10.0 * 7.2653e-5
FROZEN_SPATULA_METRICS = (
    analysis.Metric(
        "force_relative_error",
        FROZEN_SPATULA_FORCE_PRACTICAL_LIMIT,
        "force rel err",
    ),
    analysis.Metric(
        "area_relative_error",
        FROZEN_SPATULA_AREA_PRACTICAL_LIMIT,
        "area rel err",
    ),
)


def _frozen_surface_metric_value(row: analysis.Row, metric_name: str) -> float:
    if metric_name == "fragmented_area_fraction":
        return max(0.0, 1.0 - float(row["largest_component_area_fraction"]))
    return float(row[metric_name])


def _elements_for_one_percent(
    rows: list[analysis.Row], face_field: str, error_field: str
) -> list[dict[str, object]]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["representation"])].append(row)
    fits = []
    for (scene, representation), group in sorted(grouped.items()):
        samples = [
            (float(row[face_field]), abs(float(row[error_field])))
            for row in group
            if float(row[face_field]) > 0.0
            and abs(float(row[error_field])) > 0.0
        ]
        if len(samples) < 2:
            continue
        intercept, slope, residual = analysis.linear_fit(
            [math.log(sample[0]) for sample in samples],
            [math.log(sample[1]) for sample in samples],
        )
        elements = math.exp((math.log(0.01) - intercept) / slope)
        fits.append(
            {
                "scene": scene,
                "representation": representation,
                "samples": len(samples),
                "error_vs_faces_slope": slope,
                "log_residual_rms": residual,
                "elements_for_1pct_force_error": elements,
            }
        )
    return fits


def _write_surface_outputs(
    output_dir: pathlib.Path,
    rows: list[analysis.Row],
    metrics: tuple[analysis.Metric, ...],
    metric_value: analysis.MetricValue,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    order_fits, classified_series = analysis.order_fits(
        rows, metrics, metric_value
    )
    order_fields = (
        "scene",
        "representation",
        "metric",
        "samples",
        "order",
        "log_intercept",
        "log_residual_rms",
    )
    analysis.write_csv(
        output_dir / "order_fits.csv",
        order_fields,
        [{field: fit[field] for field in order_fields} for fit in order_fits],
    )
    analysis.write_csv(
        output_dir / "skipped_series.csv",
        ("scene", "representation", "metric", "reason", "detail"),
        classified_series,
    )
    return order_fits, classified_series


def _print_fit_tables(
    order_fits: list[dict[str, object]],
    classified_series: list[dict[str, object]],
    heading: str,
) -> None:
    print(heading)
    analysis.print_table(
        ("scene", "representation", "metric", "order", "log residual"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                str(fit["metric"]),
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
            for entry in classified_series
        ],
    )


def _analyze_frozen_surface(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    rows = analysis.read_rows(input_dir)
    order_fits, classified_series = _write_surface_outputs(
        output_dir,
        rows,
        FROZEN_SURFACE_METRICS,
        _frozen_surface_metric_value,
    )
    element_fits = _elements_for_one_percent(
        rows, "num_faces", "normal_force_relative_error"
    )
    analysis.write_csv(
        output_dir / "elements_for_1pct.csv",
        (
            "scene",
            "representation",
            "samples",
            "error_vs_faces_slope",
            "log_residual_rms",
            "elements_for_1pct_force_error",
        ),
        element_fits,
    )
    summary_fields = (
        "scene",
        "representation",
        "voxel_width_m",
        "num_faces",
        "num_vertices",
        "surface_distance_rms_m",
        "normal_force_relative_error",
        "pressure_error_rms_pa",
        "area_relative_error",
        "largest_component_area_fraction",
    )
    summary_rows = [
        {field: row[field] for field in summary_fields}
        for row in sorted(
            rows,
            key=lambda row: (
                row["scene"],
                row["representation"],
                -float(row["voxel_width_m"]),
            ),
        )
    ]

    def primary_value(row: analysis.Row) -> float:
        return float(row["normal_force_relative_error"])

    analysis.write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    analysis.write_svg(
        output_dir / "error_vs_h.svg",
        rows,
        "voxel_width_m",
        "h (m)",
        primary_value,
        title="Normal-force relative error",
        y_label="relative error",
    )
    analysis.write_svg(
        output_dir / "error_vs_elements.svg",
        rows,
        "num_faces",
        "faces N",
        primary_value,
        title="Normal-force relative error",
        y_label="relative error",
    )

    _print_fit_tables(order_fits, classified_series, "Error vs h order fits")
    print("\nElements for 1% normal-force error (fit vs contact faces N)")
    analysis.print_table(
        ("scene", "representation", "slope", "log residual", "N at 1%"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                f"{float(fit['error_vs_faces_slope']):.4f}",
                f"{float(fit['log_residual_rms']):.4f}",
                f"{float(fit['elements_for_1pct_force_error']):.1f}",
            )
            for fit in element_fits
        ],
    )
    print("\nRun matrix summary")
    analysis.print_table(
        ("scene", "representation", "h mm", "faces", "force rel", "connected"),
        [
            (
                row["scene"],
                row["representation"],
                f"{1000.0 * float(row['voxel_width_m']):g}",
                row["num_faces"],
                f"{float(row['normal_force_relative_error']):.6g}",
                f"{float(row['largest_component_area_fraction']):.6g}",
            )
            for row in summary_rows
        ],
    )


def _read_frozen_spatula_rows(input_dir: pathlib.Path) -> list[analysis.Row]:
    summary = input_dir / "summary.csv"
    if summary.is_file():
        paths = (summary,)
    else:
        paths = tuple(sorted((input_dir / "cases").glob("*.csv")))
        if not paths:
            paths = tuple(sorted(input_dir.glob("*.csv")))
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            file_rows = list(csv.DictReader(stream))
        for row in file_rows:
            if row.get("schema_version") != "1":
                raise RuntimeError(f"{path} does not use schema version 1")
            missing = {
                field
                for field in (
                    "pose",
                    "requested_penetration_m",
                    "representation",
                    "resolution_scale",
                )
                if not row.get(field)
            }
            if missing:
                fields = ", ".join(sorted(missing))
                raise RuntimeError(
                    f"{path} is missing required field(s): {fields}"
                )
            delta_mm = 1000.0 * float(row["requested_penetration_m"])
            row["scene"] = f"{row['pose']}__delta_{delta_mm:g}mm"
            row["voxel_width_m"] = row["resolution_scale"]
            row["_path"] = str(path)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no frozen-spatula CSV inputs found in {input_dir}")
    return rows


def _analyze_frozen_spatula(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    rows = _read_frozen_spatula_rows(input_dir)

    def metric_value(row: analysis.Row, metric_name: str) -> float:
        return abs(float(row[metric_name]))

    order_fits, classified_series = _write_surface_outputs(
        output_dir, rows, FROZEN_SPATULA_METRICS, metric_value
    )
    summary_fields = (
        "pose",
        "requested_penetration_m",
        "resolution_scale",
        "representation",
        "in_contact",
        "num_faces",
        "projected_area_m2",
        "reference_projected_area_m2",
        "area_relative_error",
        "force_norm_n",
        "reference_force_n",
        "force_relative_error",
        "largest_component_area_fraction",
        "reference_construction_wall_s",
    )
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row["pose"],
            float(row["requested_penetration_m"]),
            row["representation"],
            -float(row["resolution_scale"]),
        ),
    )
    analysis.write_csv(
        output_dir / "summary.csv",
        summary_fields,
        [
            {field: row[field] for field in summary_fields}
            for row in ordered_rows
        ],
    )

    def primary_value(row: analysis.Row) -> float:
        return abs(float(row["force_relative_error"]))

    analysis.write_svg(
        output_dir / "error_vs_scale.svg",
        rows,
        "voxel_width_m",
        "resolution scale",
        primary_value,
        title="Force relative error",
        y_label="relative error",
    )
    analysis.write_svg(
        output_dir / "error_vs_elements.svg",
        rows,
        "num_faces",
        "faces N",
        primary_value,
        title="Force relative error",
        y_label="relative error",
    )
    _print_fit_tables(
        order_fits,
        classified_series,
        "Error vs resolution-scale order fits",
    )
    print(
        "\nReference practical limits: force "
        f"{FROZEN_SPATULA_FORCE_PRACTICAL_LIMIT:.6g}, area "
        f"{FROZEN_SPATULA_AREA_PRACTICAL_LIMIT:.6g}."
    )
    print("\nRun matrix summary")
    analysis.print_table(
        (
            "pose",
            "delta mm",
            "scale",
            "representation",
            "faces",
            "force rel",
            "area rel",
        ),
        [
            (
                row["pose"],
                f"{1000.0 * float(row['requested_penetration_m']):g}",
                f"{float(row['resolution_scale']):g}",
                row["representation"],
                row["num_faces"],
                f"{float(row['force_relative_error']):.6g}",
                f"{float(row['area_relative_error']):.6g}",
            )
            for row in ordered_rows
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("frozen_surface", "frozen_spatula"))
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
        "frozen_surface": _analyze_frozen_surface,
        "frozen_spatula": _analyze_frozen_spatula,
    }
    analyzers[args.study](args.input_dir, output_dir)
    print(f"\nWrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"analyze_surface.py: {error}", file=sys.stderr)
        sys.exit(1)
