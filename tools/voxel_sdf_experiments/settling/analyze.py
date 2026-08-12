#!/usr/bin/env python3
"""Analyzes the Phase 2 free-body settling convergence ladder."""

import argparse
import collections
import csv
import dataclasses
import html
import math
import pathlib
import sys


@dataclasses.dataclass(frozen=True)
class Metric:
    """One convergence series, plus the threshold below which its samples are
    noise rather than discretization error.

    floor is a value the metric is not expected to resolve past. A series that
    lies entirely below its floor is reported as floor-limited instead of being
    fit, because a log-log slope through noise is a number with no meaning. The
    floors are not tuning knobs; each is justified where it is set below."""

    name: str
    floor: float
    label: str
    # Above this the metric has saturated and its log-log slope is not an
    # order. None means the metric cannot saturate. See fragmented_area_
    # fraction below for the case this exists to catch.
    ceiling: float | None = None


# Settling pins the support force to mg, so the force error collapses to ~1e-6
# and the discretization error surfaces as a penetration offset instead. That
# makes penetration_relative_error the primary channel here, exactly where
# normal_force_relative_error sits in Phase 1.
#
# mean_support_force_relative_error is deliberately NOT in this list. It is
# pinned by construction rather than converging, so an order fit through it
# would be a fit through solver tolerance. It is carried in the run matrix
# instead, where a value that stops looking like ~1e-6 is a red flag.
ORDER_METRICS = (
    Metric("penetration_relative_error", 0.0, "penetration rel err"),
    # Below a femtometre the penetration offset is double-precision dust on an
    # 8 cm scene.
    Metric("penetration_error_m", 1e-15, "penetration err (m)"),
    # Drift and spin hold machine precision whenever the representation
    # preserves the scene's symmetry, which is ~1e-15 over a run of seconds.
    Metric("lateral_drift_rate_m_s", 1e-12, "lateral drift rate (m/s)"),
    Metric("max_angular_speed_rad_s", 1e-12, "max angular speed (rad/s)"),
    # Settled runs chatter at ~2e-10 m regardless of h; that is the solver's
    # fixed-point tolerance, eight orders below the penetration itself.
    Metric("penetration_span_m", 1e-9, "penetration span (m)"),
    # This one is not an error metric and is only weakly a convergence metric.
    # plane_clip fragments harder as h refines -- its largest component runs
    # 0.102 / 0.026 / 0.006 over 10 / 5 / 2.5 mm -- so the complement saturates
    # toward 1 and its slope is a number about saturation, not about order.
    # tet never fragments at all, so its series is identically zero. Both ends
    # are caught below rather than reported as orders. Kept in the list so the
    # behaviour is stated in the output instead of being invisible; a gap-width
    # metric would serve this role better, but the CSV does not carry one.
    Metric("fragmented_area_fraction", 0.0, "fragmented area fraction", 0.5),
)
PRIMARY_METRIC = "penetration_relative_error"
REPRESENTATION_COLORS = {
    "tet": "#1f77b4",
    "plane_clip": "#d62728",
    "marching_cubes": "#2ca02c",
}


def _linear_fit(
    x_values: list[float], y_values: list[float]
) -> tuple[float, float, float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("a fit needs at least two paired samples")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("fit abscissas are identical")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual = math.sqrt(
        sum(
            (y_value - (intercept + slope * x_value)) ** 2
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / len(x_values)
    )
    return intercept, slope, residual


def _read_rows(input_dir: pathlib.Path) -> list[dict[str, str]]:
    """Reads the per-run CSVs, skipping the ladder's own summary.csv so its
    rows are not counted twice."""
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            file_rows = list(csv.DictReader(stream))
        if len(file_rows) != 1:
            raise RuntimeError(f"{path} has {len(file_rows)} data rows, "
                               "expected 1")
        row = file_rows[0]
        if row.get("schema_version") != "1":
            raise RuntimeError(f"{path} does not use schema version 1")
        row["_path"] = str(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no per-run CSV inputs found in {input_dir}")
    return rows


def _is_settled(row: dict[str, str]) -> bool:
    return row.get("settled", "").strip().lower() == "true"


def _metric_value(row: dict[str, str], metric_name: str) -> float:
    """Returns the magnitude of one metric for a row.

    Two metrics are derived rather than read straight from the CSV:

    fragmented_area_fraction complements the emitted largest-component share,
    matching the Phase 1 analyzer so the two studies report the same quantity.

    lateral_drift_rate_m_s divides the emitted max lateral offset by the run
    duration. The max alone grows with run length -- drift accumulates
    monotonically -- so comparing a long run against a short one on the raw max
    compares run lengths, not representations. The rate is the comparable form.

    penetration_relative_error is signed in the CSV, negative where the
    representation under-penetrates. The sign is reported in the run matrix;
    fits use the magnitude, since a log-log slope needs a positive series."""
    if metric_name == "fragmented_area_fraction":
        return max(0.0, 1.0 - float(row["largest_component_area_fraction"]))
    if metric_name == "lateral_drift_rate_m_s":
        duration = float(row["duration_s"])
        if duration <= 0.0:
            raise ValueError(f"{row['_path']} has non-positive duration_s")
        return abs(float(row["max_lateral_offset_m"])) / duration
    return abs(float(row[metric_name]))


def _group_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["scene"], row["representation"])


def _order_fits(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fits error against h for every (scene, representation, metric) series.

    Returns the fits and, separately, a row per series that could not be fit.
    Nothing is dropped silently: a series excluded because it sits at its noise
    floor, is identically zero, or has too few settled rungs is reported with
    the reason, so a blank line in the fit table is never ambiguous."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    fits: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for (scene, representation), group in sorted(grouped.items()):
        settled = [row for row in group if _is_settled(row)]
        for metric in ORDER_METRICS:
            values = [
                (float(row["voxel_width_m"]), _metric_value(row, metric.name))
                for row in settled
            ]
            samples = [pair for pair in values if pair[1] > metric.floor]
            saturated = (
                metric.ceiling is not None
                and bool(values)
                and all(value >= metric.ceiling for _, value in values)
            )
            if saturated:
                smallest = min(value for _, value in values)
                skipped.append(
                    {
                        "scene": scene,
                        "representation": representation,
                        "metric": metric.name,
                        "reason": "saturated",
                        "detail": f"smallest settled value {smallest:.3g} is "
                        f"at or above ceiling {metric.ceiling:g}; the slope "
                        "would describe saturation, not order",
                    }
                )
                continue
            if len(samples) >= 2:
                intercept, order, residual = _linear_fit(
                    [math.log(sample[0]) for sample in samples],
                    [math.log(sample[1]) for sample in samples],
                )
                fits.append(
                    {
                        "scene": scene,
                        "representation": representation,
                        "metric": metric.name,
                        "samples": len(samples),
                        "settled_rungs": len(settled),
                        "order": order,
                        "log_intercept": intercept,
                        "log_residual_rms": residual,
                    }
                )
                if len(samples) < len(values):
                    skipped.append(
                        {
                            "scene": scene,
                            "representation": representation,
                            "metric": metric.name,
                            "reason": "partially_below_floor",
                            "detail": f"{len(values) - len(samples)} of "
                            f"{len(values)} rungs below floor "
                            f"{metric.floor:g}; fit uses the rest",
                        }
                    )
                continue
            if not settled:
                reason, detail = "no_settled_rungs", (
                    f"{len(group)} rungs ran, none settled"
                )
            elif len(settled) < 2:
                reason, detail = "too_few_settled_rungs", (
                    f"only {len(settled)} of {len(group)} rungs settled"
                )
            elif all(value == 0.0 for _, value in values):
                reason, detail = "identically_zero", (
                    "every settled rung is exactly zero"
                )
            else:
                largest = max(value for _, value in values)
                reason, detail = "at_noise_floor", (
                    f"largest settled value {largest:.3g} is at or below "
                    f"floor {metric.floor:g}"
                )
            skipped.append(
                {
                    "scene": scene,
                    "representation": representation,
                    "metric": metric.name,
                    "reason": reason,
                    "detail": detail,
                }
            )
    return fits, skipped


def _faces_for_one_percent(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Fits penetration error against contact-face count.

    The Phase 1 analog fits force error against faces; settling's error lives
    in the penetration channel, so the same question -- how many elements buy
    1% accuracy -- is asked of that channel here."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    fits = []
    for (scene, representation), group in sorted(grouped.items()):
        samples = [
            (float(row["mean_faces"]), _metric_value(row, PRIMARY_METRIC))
            for row in group
            if _is_settled(row) and float(row["mean_faces"]) > 0.0
        ]
        samples = [pair for pair in samples if pair[1] > 0.0]
        if len(samples) < 2:
            continue
        intercept, slope, residual = _linear_fit(
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


def _write_csv(
    path: pathlib.Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(headers: tuple[str, ...], table_rows: list[tuple[str, ...]]) -> None:
    if not table_rows:
        print("  (none)")
        return
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _write_svg(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    x_field: str,
    x_label: str,
) -> None:
    """Plots the primary error channel against h or against face count.

    Settled rungs are joined by a line; unsettled ones are drawn as hollow
    markers with no line through them, so a representation that lost contact
    cannot be mistaken for one that merely converged badly."""
    scenes = sorted({row["scene"] for row in rows})
    width = 900
    panel_height = 390
    height = 70 + panel_height * len(scenes)
    margin_left = 95
    margin_right = 30
    plot_width = width - margin_left - margin_right
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;font-size:13px}'
        '.title{font-size:17px;font-weight:bold}.axis{stroke:#222;fill:none}'
        '.grid{stroke:#ddd;stroke-width:1}</style>',
        '<text class="title" x="20" y="28">Penetration relative error '
        '(hollow = did not settle)</text>',
    ]
    for scene_index, scene in enumerate(scenes):
        top = 55 + scene_index * panel_height
        bottom = top + 285
        scene_rows = [
            row
            for row in rows
            if row["scene"] == scene
            and float(row[x_field]) > 0.0
            and _metric_value(row, PRIMARY_METRIC) > 0.0
        ]
        if not scene_rows:
            continue
        x_logs = [math.log10(float(row[x_field])) for row in scene_rows]
        # A run that lost contact reports a penetration error of order 100,
        # four decades above a converging one. Letting those set the y range
        # would flatten every real curve into the bottom pixel row, so the
        # scale follows the settled rungs and the unsettled markers are clamped
        # into view at the panel edge.
        scale_rows = [row for row in scene_rows if _is_settled(row)] or scene_rows
        y_logs = [
            math.log10(_metric_value(row, PRIMARY_METRIC)) for row in scale_rows
        ]
        x_min, x_max = min(x_logs), max(x_logs)
        y_min, y_max = min(y_logs), max(y_logs)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        x_pad = 0.05 * (x_max - x_min)
        y_pad = 0.08 * (y_max - y_min)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

        def x_pixel(value: float) -> float:
            return margin_left + plot_width * (math.log10(value) - x_min) / (
                x_max - x_min
            )

        def y_pixel(value: float) -> float:
            raw = bottom - 285 * (math.log10(value) - y_min) / (y_max - y_min)
            return min(max(raw, bottom - 285), bottom)

        pieces.append(
            f'<rect class="axis" x="{margin_left}" y="{top}" '
            f'width="{plot_width}" height="285"/>'
        )
        pieces.append(
            f'<text x="{margin_left}" y="{top - 10}">{html.escape(scene)}</text>'
        )
        pieces.append(
            f'<text x="{width / 2}" y="{bottom + 42}" text-anchor="middle">'
            f'{html.escape(x_label)} (log scale)</text>'
        )
        pieces.append(
            f'<text transform="translate(24 {top + 142}) rotate(-90)" '
            'text-anchor="middle">penetration relative error (log scale)</text>'
        )
        for representation, color in REPRESENTATION_COLORS.items():
            method_rows = sorted(
                [row for row in scene_rows if row["representation"] == representation],
                key=lambda row: float(row[x_field]),
            )
            if not method_rows:
                continue
            settled_rows = [row for row in method_rows if _is_settled(row)]
            if len(settled_rows) >= 2:
                points = " ".join(
                    f'{x_pixel(float(row[x_field])):.2f},'
                    f'{y_pixel(_metric_value(row, PRIMARY_METRIC)):.2f}'
                    for row in settled_rows
                )
                pieces.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2"/>'
                )
            for row in method_rows:
                center = (
                    f'cx="{x_pixel(float(row[x_field])):.2f}" '
                    f'cy="{y_pixel(_metric_value(row, PRIMARY_METRIC)):.2f}" '
                    'r="3.5"'
                )
                if _is_settled(row):
                    pieces.append(f'<circle {center} fill="{color}"/>')
                else:
                    pieces.append(
                        f'<circle {center} fill="white" stroke="{color}" '
                        'stroke-width="2"/>'
                    )
        legend_x = margin_left + 12
        for legend_index, (representation, color) in enumerate(
            REPRESENTATION_COLORS.items()
        ):
            legend_y = top + 20 + legend_index * 20
            pieces.append(
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="3"/>'
            )
            pieces.append(
                f'<text x="{legend_x + 26}" y="{legend_y + 5}">'
                f'{html.escape(representation)}</text>'
            )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


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

    rows = _read_rows(args.input_dir)
    order_fits, skipped_series = _order_fits(rows)
    face_fits = _faces_for_one_percent(rows)
    _write_csv(
        output_dir / "order_fits.csv",
        (
            "scene",
            "representation",
            "metric",
            "samples",
            "settled_rungs",
            "order",
            "log_intercept",
            "log_residual_rms",
        ),
        order_fits,
    )
    _write_csv(
        output_dir / "skipped_series.csv",
        ("scene", "representation", "metric", "reason", "detail"),
        skipped_series,
    )
    _write_csv(
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
    _write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    _write_svg(output_dir / "error_vs_h.svg", rows, "voxel_width_m", "h (m)")
    _write_svg(
        output_dir / "error_vs_elements.svg", rows, "mean_faces", "mean faces N"
    )

    unsettled = [row for row in ordered_rows if not _is_settled(row)]
    print("Error vs h order fits (settled rungs only)")
    _print_table(
        ("scene", "representation", "metric", "n", "order", "log residual"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                str(fit["metric"]),
                str(fit["samples"]),
                f'{float(fit["order"]):.4f}',
                f'{float(fit["log_residual_rms"]):.4f}',
            )
            for fit in order_fits
        ],
    )
    print("\nSeries not fit, and why")
    _print_table(
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
    _print_table(
        ("scene", "representation", "slope", "log residual", "N at 1%"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                f'{float(fit["error_vs_faces_slope"]):.4f}',
                f'{float(fit["log_residual_rms"]):.4f}',
                f'{float(fit["faces_for_1pct_penetration_error"]):.1f}',
            )
            for fit in face_fits
        ],
    )
    print("\nRun matrix summary")
    _print_table(
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
                f'{1000.0 * float(row["voxel_width_m"]):g}',
                "yes" if _is_settled(row) else "NO",
                f'{float(row["mean_faces"]):.6g}',
                f'{float(row["penetration_relative_error"]):+.6g}',
                f'{float(row["mean_support_force_relative_error"]):.3g}',
                f'{float(row["penetration_span_m"]):.3g}',
            )
            for row in ordered_rows
        ],
    )
    if unsettled:
        # Losing contact is a result, not a failure, so these rows are kept and
        # reported -- but they are excluded from every fit above, since a run
        # that never reached equilibrium has no equilibrium error to converge.
        print(
            f"\n{len(unsettled)} of {len(ordered_rows)} rungs did not settle "
            "and were excluded from all fits:"
        )
        for row in unsettled:
            print(
                f"  {row['scene']} {row['representation']} "
                f"h={1000.0 * float(row['voxel_width_m']):g} mm"
            )
    print(f"\nWrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"analyze.py: {error}", file=sys.stderr)
        sys.exit(1)
