#!/usr/bin/env python3
"""Analyzes the Phase 1 frozen contact-surface convergence ladder."""

import argparse
import collections
import math
import pathlib
import sys

# This file is also invoked directly by run_ladder.py and run_smoke.py. Add
# voxel_sdf_experiments itself so the sibling common package is importable in
# that mode, where Python otherwise adds only frozen_surface to sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import analysis  # noqa: E402

ORDER_METRICS = (
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


def _metric_value(row: analysis.Row, metric_name: str) -> float:
    if metric_name == "fragmented_area_fraction":
        return max(0.0, 1.0 - float(row["largest_component_area_fraction"]))
    return float(row[metric_name])


def _elements_for_one_percent(
    rows: list[analysis.Row],
) -> list[dict[str, object]]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["representation"])].append(row)
    fits = []
    for (scene, representation), group in sorted(grouped.items()):
        samples = [
            (float(row["num_faces"]), float(row["normal_force_relative_error"]))
            for row in group
            if float(row["num_faces"]) > 0.0
            and float(row["normal_force_relative_error"]) > 0.0
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Defaults to INPUT_DIR/analysis.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = analysis.read_rows(args.input_dir)
    order_fits, classified_series = analysis.order_fits(
        rows, ORDER_METRICS, _metric_value
    )
    element_fits = _elements_for_one_percent(rows)
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

    print("Error vs h order fits")
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
    print(f"\nWrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"analyze.py: {error}", file=sys.stderr)
        sys.exit(1)
