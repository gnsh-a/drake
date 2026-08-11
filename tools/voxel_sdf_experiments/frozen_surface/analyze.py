#!/usr/bin/env python3
"""Analyzes the Phase 1 frozen contact-surface convergence ladder."""

import argparse
import collections
import csv
import html
import math
import pathlib
import sys


ORDER_METRICS = (
    "surface_distance_rms_m",
    "normal_force_relative_error",
    "pressure_error_rms_pa",
    "peak_pressure_relative_error",
    "area_relative_error",
    "patch_radius_relative_error",
    "centroid_position_error_m",
    "fragmented_area_fraction",
)
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
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            file_rows = list(csv.DictReader(stream))
        if len(file_rows) != 1:
            raise RuntimeError(f"{path} has {len(file_rows)} data rows, expected 1")
        row = file_rows[0]
        if row.get("schema_version") != "1":
            raise RuntimeError(f"{path} does not use schema version 1")
        row["_path"] = str(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no CSV inputs found in {input_dir}")
    return rows


def _write_csv(
    path: pathlib.Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _order_fits(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["representation"])].append(row)
    fits = []
    for (scene, representation), group in sorted(grouped.items()):
        for metric in ORDER_METRICS:
            minimum_error = (
                1e-14
                if metric
                in (
                    "surface_distance_rms_m",
                    "centroid_position_error_m",
                )
                else 0.0
            )

            def metric_value(row: dict[str, str]) -> float:
                if metric == "fragmented_area_fraction":
                    return max(
                        0.0,
                        1.0 - float(row["largest_component_area_fraction"]),
                    )
                return float(row[metric])

            samples = [
                (float(row["voxel_width_m"]), metric_value(row))
                for row in group
                if metric_value(row) > minimum_error
            ]
            if len(samples) < 2:
                continue
            intercept, order, residual = _linear_fit(
                [math.log(sample[0]) for sample in samples],
                [math.log(sample[1]) for sample in samples],
            )
            fits.append(
                {
                    "scene": scene,
                    "representation": representation,
                    "metric": metric,
                    "samples": len(samples),
                    "order": order,
                    "log_intercept": intercept,
                    "log_residual_rms": residual,
                }
            )
    return fits


def _elements_for_one_percent(
    rows: list[dict[str, str]],
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
        intercept, slope, residual = _linear_fit(
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


def _print_table(headers: tuple[str, ...], table_rows: list[tuple[str, ...]]) -> None:
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
        '<text class="title" x="20" y="28">Normal-force relative error</text>',
    ]
    for scene_index, scene in enumerate(scenes):
        top = 55 + scene_index * panel_height
        bottom = top + 285
        scene_rows = [
            row
            for row in rows
            if row["scene"] == scene
            and float(row[x_field]) > 0.0
            and float(row["normal_force_relative_error"]) > 0.0
        ]
        x_logs = [math.log10(float(row[x_field])) for row in scene_rows]
        y_logs = [
            math.log10(float(row["normal_force_relative_error"]))
            for row in scene_rows
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
            return bottom - 285 * (math.log10(value) - y_min) / (y_max - y_min)

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
            'text-anchor="middle">relative error (log scale)</text>'
        )
        for representation, color in REPRESENTATION_COLORS.items():
            method_rows = sorted(
                [row for row in scene_rows if row["representation"] == representation],
                key=lambda row: float(row[x_field]),
            )
            if not method_rows:
                continue
            points = " ".join(
                f'{x_pixel(float(row[x_field])):.2f},'
                f'{y_pixel(float(row["normal_force_relative_error"])):.2f}'
                for row in method_rows
            )
            pieces.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2"/>'
            )
            for row in method_rows:
                pieces.append(
                    f'<circle cx="{x_pixel(float(row[x_field])):.2f}" '
                    f'cy="{y_pixel(float(row["normal_force_relative_error"])):.2f}" '
                    f'r="3.5" fill="{color}"/>'
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
    order_fits = _order_fits(rows)
    element_fits = _elements_for_one_percent(rows)
    _write_csv(
        output_dir / "order_fits.csv",
        (
            "scene",
            "representation",
            "metric",
            "samples",
            "order",
            "log_intercept",
            "log_residual_rms",
        ),
        order_fits,
    )
    _write_csv(
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
    _write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    _write_svg(output_dir / "error_vs_h.svg", rows, "voxel_width_m", "h (m)")
    _write_svg(output_dir / "error_vs_elements.svg", rows, "num_faces", "faces N")

    print("Error vs h order fits")
    _print_table(
        ("scene", "representation", "metric", "order", "log residual"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                str(fit["metric"]),
                f'{float(fit["order"]):.4f}',
                f'{float(fit["log_residual_rms"]):.4f}',
            )
            for fit in order_fits
        ],
    )
    print("\nElements for 1% normal-force error (fit vs contact faces N)")
    _print_table(
        ("scene", "representation", "slope", "log residual", "N at 1%"),
        [
            (
                str(fit["scene"]),
                str(fit["representation"]),
                f'{float(fit["error_vs_faces_slope"]):.4f}',
                f'{float(fit["log_residual_rms"]):.4f}',
                f'{float(fit["elements_for_1pct_force_error"]):.1f}',
            )
            for fit in element_fits
        ],
    )
    print("\nRun matrix summary")
    _print_table(
        ("scene", "representation", "h mm", "faces", "force rel", "connected"),
        [
            (
                row["scene"],
                row["representation"],
                f'{1000.0 * float(row["voxel_width_m"]):g}',
                row["num_faces"],
                f'{float(row["normal_force_relative_error"]):.6g}',
                f'{float(row["largest_component_area_fraction"]):.6g}',
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
