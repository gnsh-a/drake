"""Shared analysis machinery for the voxel-SDF experiment studies."""

import collections
from collections.abc import Callable
import csv
import dataclasses
import html
import math
import pathlib

Row = dict[str, str]
MetricValue = Callable[[Row, str], float]


@dataclasses.dataclass(frozen=True)
class Metric:
    """One convergence series and its meaningful numeric range.

    ``floor`` is the value the metric is not expected to resolve past. A
    series entirely below its floor is reported as floor-limited instead of
    being fit, because a log-log slope through noise has no meaning. Floors
    are justified by each study where its metrics are declared; they are not
    tuning knobs.

    ``ceiling`` optionally identifies a saturation range. If every eligible
    value is at or above it, the series is reported as saturated instead of
    fitting a slope that does not represent convergence order.
    """

    name: str
    floor: float
    label: str
    ceiling: float | None = None

    def __post_init__(self) -> None:
        if self.floor < 0.0:
            raise ValueError("a metric floor must be nonnegative")
        if self.ceiling is not None and self.ceiling <= self.floor:
            raise ValueError("a metric ceiling must be greater than its floor")


@dataclasses.dataclass(frozen=True)
class RowGate:
    """Selects the rows eligible for fits and names that eligibility state.

    For example, a caller can supply a gate named ``valid``. The engine then
    reports ``no_valid_rungs`` or ``too_few_valid_rungs`` without itself
    knowing what validity means. Ungated studies fit every rung.
    """

    name: str
    predicate: Callable[[Row], bool]

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name.startswith("_"):
            raise ValueError("a row-gate name must be a public identifier")


REPRESENTATION_COLORS = {
    "tet": "#1f77b4",
    "plane_clip": "#d62728",
    "marching_cubes": "#2ca02c",
}


def linear_fit(
    x_values: list[float], y_values: list[float]
) -> tuple[float, float, float]:
    """Returns intercept, slope, and RMS residual for a linear fit."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("a fit needs at least two paired samples")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("fit abscissas are identical")
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    residual = math.sqrt(
        sum(
            (y_value - (intercept + slope * x_value)) ** 2
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / len(x_values)
    )
    return intercept, slope, residual


def read_rows(input_dir: pathlib.Path) -> list[Row]:
    """Reads version-1, single-row CSVs emitted for individual runs.

    A ladder driver's ``summary.csv`` is ignored so its copies of the run rows
    are not counted a second time.
    """
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            file_rows = list(csv.DictReader(stream))
        if len(file_rows) != 1:
            raise RuntimeError(
                f"{path} has {len(file_rows)} data rows, expected 1"
            )
        row = file_rows[0]
        if row.get("schema_version") != "1":
            raise RuntimeError(f"{path} does not use schema version 1")
        missing = {
            field
            for field in ("scene", "representation", "voxel_width_m")
            if not row.get(field)
        }
        if missing:
            fields = ", ".join(sorted(missing))
            raise RuntimeError(f"{path} is missing required field(s): {fields}")
        row["_path"] = str(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no per-run CSV inputs found in {input_dir}")
    return rows


def order_fits(
    rows: list[Row],
    metrics: tuple[Metric, ...],
    metric_value: MetricValue,
    *,
    row_gate: RowGate | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fits each metric against voxel width for every emitted series.

    Returns the fits and, separately, explicit classification rows for every
    series that could not use all of its rungs. No series is silently dropped:
    exact zeros, floor-limited values, saturation, and insufficient eligible
    rungs all carry a reason.
    """
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["representation"])].append(row)
    fits: list[dict[str, object]] = []
    classifications: list[dict[str, object]] = []
    for (scene, representation), group in sorted(grouped.items()):
        eligible = (
            group
            if row_gate is None
            else [row for row in group if row_gate.predicate(row)]
        )
        for metric in metrics:
            values = [
                (float(row["voxel_width_m"]), metric_value(row, metric.name))
                for row in eligible
            ]
            samples = [pair for pair in values if pair[1] > metric.floor]
            saturated = (
                metric.ceiling is not None
                and bool(values)
                and all(value >= metric.ceiling for _, value in values)
            )
            if saturated:
                smallest = min(value for _, value in values)
                qualifier = f" {row_gate.name}" if row_gate is not None else ""
                detail = (
                    f"smallest{qualifier} value {smallest:.3g} is at or above "
                    f"ceiling {metric.ceiling:g}; the slope would describe "
                    "saturation, not order"
                )
                classifications.append(
                    {
                        "scene": scene,
                        "representation": representation,
                        "metric": metric.name,
                        "reason": "saturated",
                        "detail": detail,
                    }
                )
                continue
            if len(samples) >= 2:
                intercept, order, residual = linear_fit(
                    [math.log(sample[0]) for sample in samples],
                    [math.log(sample[1]) for sample in samples],
                )
                fits.append(
                    {
                        "scene": scene,
                        "representation": representation,
                        "metric": metric.name,
                        "samples": len(samples),
                        "eligible_rungs": len(eligible),
                        "order": order,
                        "log_intercept": intercept,
                        "log_residual_rms": residual,
                    }
                )
                if len(samples) < len(values):
                    classifications.append(
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
            if not eligible:
                if row_gate is None:
                    reason = "no_rungs"
                    detail = "no rungs available"
                else:
                    reason = f"no_{row_gate.name}_rungs"
                    detail = f"{len(group)} rungs ran, none {row_gate.name}"
            elif len(eligible) < 2:
                if row_gate is None:
                    reason = "too_few_rungs"
                    detail = f"only {len(eligible)} rung available"
                else:
                    reason = f"too_few_{row_gate.name}_rungs"
                    detail = (
                        f"only {len(eligible)} of {len(group)} rungs "
                        f"{row_gate.name}"
                    )
            elif all(value == 0.0 for _, value in values):
                reason = "identically_zero"
                qualifier = f" {row_gate.name}" if row_gate is not None else ""
                detail = f"every{qualifier} rung is exactly zero"
            else:
                largest = max(value for _, value in values)
                reason = "at_noise_floor"
                qualifier = f" {row_gate.name}" if row_gate is not None else ""
                detail = (
                    f"largest{qualifier} value {largest:.3g} is at or below "
                    f"floor {metric.floor:g}"
                )
            classifications.append(
                {
                    "scene": scene,
                    "representation": representation,
                    "metric": metric.name,
                    "reason": reason,
                    "detail": detail,
                }
            )
    return fits, classifications


def write_csv(
    path: pathlib.Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    """Writes dictionaries using an explicit, stable column order."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(
    headers: tuple[str, ...], table_rows: list[tuple[str, ...]]
) -> None:
    """Prints a compact, left-aligned plain-text table."""
    if not table_rows:
        print("  (none)")
        return
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def write_svg(
    path: pathlib.Path,
    rows: list[Row],
    x_field: str,
    x_label: str,
    metric_value: Callable[[Row], float],
    *,
    title: str,
    y_label: str,
    row_gate: RowGate | None = None,
    hollow_note: str | None = None,
    representation_colors: dict[str, str] | None = None,
) -> None:
    """Writes a dependency-free log-log SVG for one primary error channel.

    When a row gate is supplied, eligible rungs are joined and filled while
    ineligible rungs are hollow. The eligible values determine the y scale so
    a failed run cannot flatten a converging curve.
    """
    colors = representation_colors or REPRESENTATION_COLORS
    scenes = sorted({row["scene"] for row in rows})
    width = 900
    panel_height = 390
    height = 70 + panel_height * len(scenes)
    margin_left = 95
    margin_right = 30
    plot_width = width - margin_left - margin_right
    displayed_title = title
    if hollow_note is not None:
        displayed_title += f" ({hollow_note})"
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:sans-serif;font-size:13px}"
        ".title{font-size:17px;font-weight:bold}.axis{stroke:#222;fill:none}"
        ".grid{stroke:#ddd;stroke-width:1}</style>",
        f'<text class="title" x="20" y="28">'
        f"{html.escape(displayed_title)}</text>",
    ]
    for scene_index, scene in enumerate(scenes):
        top = 55 + scene_index * panel_height
        bottom = top + 285
        scene_rows = [
            row
            for row in rows
            if row["scene"] == scene
            and float(row[x_field]) > 0.0
            and metric_value(row) > 0.0
        ]
        if not scene_rows:
            continue
        x_logs = [math.log10(float(row[x_field])) for row in scene_rows]
        eligible_rows = (
            scene_rows
            if row_gate is None
            else [row for row in scene_rows if row_gate.predicate(row)]
        )
        scale_rows = eligible_rows or scene_rows
        y_logs = [math.log10(metric_value(row)) for row in scale_rows]
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
            if row_gate is None:
                return raw
            return min(max(raw, bottom - 285), bottom)

        pieces.append(
            f'<rect class="axis" x="{margin_left}" y="{top}" '
            f'width="{plot_width}" height="285"/>'
        )
        escaped_scene = html.escape(scene)
        pieces.append(
            f'<text x="{margin_left}" y="{top - 10}">{escaped_scene}</text>'
        )
        pieces.append(
            f'<text x="{width / 2}" y="{bottom + 42}" text-anchor="middle">'
            f"{html.escape(x_label)} (log scale)</text>"
        )
        pieces.append(
            f'<text transform="translate(24 {top + 142}) rotate(-90)" '
            f'text-anchor="middle">{html.escape(y_label)} (log scale)</text>'
        )
        for representation, color in colors.items():
            method_rows = sorted(
                [
                    row
                    for row in scene_rows
                    if row["representation"] == representation
                ],
                key=lambda row: float(row[x_field]),
            )
            if not method_rows:
                continue
            line_rows = (
                method_rows
                if row_gate is None
                else [row for row in method_rows if row_gate.predicate(row)]
            )
            if row_gate is None or len(line_rows) >= 2:
                points = " ".join(
                    f"{x_pixel(float(row[x_field])):.2f},"
                    f"{y_pixel(metric_value(row)):.2f}"
                    for row in line_rows
                )
                pieces.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2"/>'
                )
            for row in method_rows:
                center = (
                    f'cx="{x_pixel(float(row[x_field])):.2f}" '
                    f'cy="{y_pixel(metric_value(row)):.2f}" r="3.5"'
                )
                if row_gate is None or row_gate.predicate(row):
                    pieces.append(f'<circle {center} fill="{color}"/>')
                else:
                    pieces.append(
                        f'<circle {center} fill="white" stroke="{color}" '
                        'stroke-width="2"/>'
                    )
        legend_x = margin_left + 12
        for legend_index, (representation, color) in enumerate(colors.items()):
            legend_y = top + 20 + legend_index * 20
            pieces.append(
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="3"/>'
            )
            pieces.append(
                f'<text x="{legend_x + 26}" y="{legend_y + 5}">'
                f"{html.escape(representation)}</text>"
            )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")
