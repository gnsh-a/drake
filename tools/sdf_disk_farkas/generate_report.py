"""Generates an evidence-backed tet versus affine-SDF disk report."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

_EPS_CONTINUUM = 0.653
_EPS_DISCRETE = 0.644
_OMEGA_TERMINAL_CUTOFF = 0.1
_RADIUS = 0.01213
_THICKNESS = 0.00175
_DENSITY = 7010.0
_GRAVITY = 9.81
_MASS = _DENSITY * math.pi * _RADIUS**2 * _THICKNESS
_MG = _MASS * _GRAVITY


@dataclass(frozen=True)
class RunSpec:
    target_voxel_size: float
    time_step: float
    csv_path: Path


def _parse_voxel_run(value):
    try:
        voxel_size_text, time_step_text, path_text = value.split(",", 2)
        spec = RunSpec(
            target_voxel_size=float(voxel_size_text),
            time_step=float(time_step_text),
            csv_path=Path(path_text),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected TARGET_VOXEL_SIZE,TIME_STEP,CSV_PATH"
        ) from error
    if spec.target_voxel_size <= 0.0 or spec.time_step <= 0.0:
        raise argparse.ArgumentTypeError(
            "voxel size and time step must be positive"
        )
    return spec


def _parse_args():
    root = Path("tools/hydro_compare/out/sdf_disk_farkas")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tet-csv",
        type=Path,
        default=root / "lagged_tet_h2p5mm_dt1ms.csv",
    )
    parser.add_argument("--tet-target-voxel-size", type=float, default=0.0025)
    parser.add_argument("--tet-time-step", type=float, default=0.001)
    parser.add_argument(
        "--voxel-run",
        type=_parse_voxel_run,
        action="append",
        required=True,
        metavar="TARGET_VOXEL_SIZE,TIME_STEP,CSV_PATH",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "disk_farkas_affine_sdf.html"
    )
    args = parser.parse_args()
    if args.tet_target_voxel_size <= 0.0 or args.tet_time_step <= 0.0:
        parser.error("tet voxel size and time step must be positive")
    return args


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                row[key] = float(value)
            rows.append(row)
    if len(rows) < 2:
        raise RuntimeError(f"{path} must contain at least two samples")
    return rows


def _yaw(row):
    return 2.0 * math.atan2(row["qz"], row["qw"])


def _first_stopped_time(rows):
    for row in rows[1:]:
        if row["linear_speed"] < 1e-6 and row["angular_speed"] < 1e-4:
            return row["time"]
    return math.nan


def _summarize(rows):
    dynamic = rows[1:]
    # The raw ratio becomes numerically meaningless once angular speed is near
    # zero.  Use the last still-spinning sample, as in the original report.
    terminal_candidates = [
        row
        for row in dynamic
        if row["angular_speed"] >= _OMEGA_TERMINAL_CUTOFF
        and math.isfinite(row["eps"])
    ]
    if not terminal_candidates:
        raise RuntimeError("No samples satisfy the terminal-epsilon cutoff")
    terminal = terminal_candidates[-1]
    load_window = [row for row in dynamic if row["time"] <= 0.1]
    normal_ratios = [row["normal_force_z"] / _MG for row in load_window]
    first = dynamic[0]
    final = rows[-1]
    return {
        "terminal_eps": terminal["eps"],
        "terminal_time": terminal["time"],
        "continuum_error": terminal["eps"] - _EPS_CONTINUUM,
        "discrete_error": terminal["eps"] - _EPS_DISCRETE,
        "stop_time": _first_stopped_time(rows),
        "mean_normal_ratio": statistics.fmean(normal_ratios),
        "max_normal_ratio_error": max(
            abs(value - 1.0) for value in normal_ratios
        ),
        "first_normal_force": first["normal_force_z"],
        "first_contact_area": first["contact_area"],
        "first_surface_vertices": int(first["surface_vertices"]),
        "first_surface_faces": int(first["surface_faces"]),
        "hydro_min": min(int(row["hydro_contacts"]) for row in dynamic),
        "hydro_max": max(int(row["hydro_contacts"]) for row in dynamic),
        "point_max": max(int(row["point_contacts"]) for row in rows),
        "final_x": final["x"],
        "final_y": final["y"],
        "final_yaw": _yaw(final),
    }


def _trajectory_difference(tet, voxel):
    if len(tet) != len(voxel):
        raise RuntimeError("Tet and voxel CSV files have different row counts")
    position_sq = []
    yaw_sq = []
    for row_t, row_v in zip(tet, voxel, strict=True):
        if row_t["time"] != row_v["time"]:
            raise RuntimeError("Tet and voxel CSV time grids differ")
        position_sq.append(
            (row_v["x"] - row_t["x"]) ** 2 + (row_v["y"] - row_t["y"]) ** 2
        )
        yaw_sq.append((_yaw(row_v) - _yaw(row_t)) ** 2)
    return math.sqrt(statistics.fmean(position_sq)), math.sqrt(
        statistics.fmean(yaw_sq)
    )


def _polyline(points, x_map, y_map):
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def _line_chart(series, y_label, reference_lines=(), x_label="time [s]"):
    width, height = 720, 300
    left, top, plot_width, plot_height = 62, 18, 630, 225
    points = [point for _, values, _ in series for point in values]
    if not points:
        raise RuntimeError("Cannot plot an empty series")
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    y_values = [y for _, y in points] + [value for value, _ in reference_lines]
    y_min, y_max = min(y_values), max(y_values)
    y_pad = 0.08 * (y_max - y_min) or 0.1
    y_min -= y_pad
    y_max += y_pad

    def x_map(value):
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_map(value):
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    svg = [f'<svg viewBox="0 0 {width} {height}" class="plot">']
    svg.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" '
        f'height="{plot_height}" class="frame"/>'
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_value = y_min + fraction * (y_max - y_min)
        y_pixel = y_map(y_value)
        svg.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" '
            f'x2="{left + plot_width}" y2="{y_pixel:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{left - 7}" y="{y_pixel + 4:.2f}" '
            f'text-anchor="end" class="tick">{y_value:.3g}</text>'
        )
    for value, label in reference_lines:
        y_pixel = y_map(value)
        svg.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" '
            f'x2="{left + plot_width}" y2="{y_pixel:.2f}" class="reference"/>'
        )
        svg.append(
            f'<text x="{left + plot_width - 4}" y="{y_pixel - 5:.2f}" '
            f'text-anchor="end" class="reference-label">{label}</text>'
        )
    for name, values, css_class in series:
        svg.append(
            f'<polyline points="{_polyline(values, x_map, y_map)}" '
            f'class="{css_class}"/>'
        )
        svg.append(
            f'<text x="{x_map(values[-1][0]) - 4:.2f}" '
            f'y="{y_map(values[-1][1]) - 7:.2f}" text-anchor="end" '
            f'class="{css_class}-label">{name}</text>'
        )
    svg.append(
        f'<text x="{left + plot_width / 2}" y="{height - 8}" '
        f'text-anchor="middle" class="axis">{x_label}</text>'
    )
    svg.append(
        f'<text x="14" y="{top + plot_height / 2}" text-anchor="middle" '
        f'class="axis" transform="rotate(-90 14 {top + plot_height / 2})">'
        f"{y_label}</text>"
    )
    svg.append("</svg>")
    return "".join(svg)


def _active_eps(rows):
    return [
        (row["time"], row["eps"])
        for row in rows[1:]
        if row["angular_speed"] >= _OMEGA_TERMINAL_CUTOFF
        and math.isfinite(row["eps"])
    ]


def _normal_ratio(rows):
    return [
        (row["time"], row["normal_force_z"] / _MG)
        for row in rows[1:]
        if row["time"] <= 0.14
    ]


def _format(value, digits=6):
    return "—" if not math.isfinite(value) else f"{value:.{digits}g}"


@dataclass(frozen=True)
class RunResult:
    spec: RunSpec
    rows: list
    summary: dict


def _is_close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def _make_html(tet_rows, tet_spec, voxel_specs):
    tet = _summarize(tet_rows)
    results = [
        RunResult(spec, _read_csv(spec.csv_path), None) for spec in voxel_specs
    ]
    results = [
        RunResult(result.spec, result.rows, _summarize(result.rows))
        for result in results
    ]
    spatial = [
        result
        for result in results
        if _is_close(result.spec.time_step, tet_spec.time_step)
    ]
    spatial.sort(key=lambda result: result.spec.target_voxel_size, reverse=True)
    if len(spatial) < 3:
        raise RuntimeError("Expected at least three fixed-time-step voxel runs")
    fine = spatial[-1]
    time_step_checks = [
        result
        for result in results
        if _is_close(result.spec.target_voxel_size, fine.spec.target_voxel_size)
        and result.spec.time_step < fine.spec.time_step
    ]
    if not time_step_checks:
        raise RuntimeError("Expected a finer time-step run at the finest grid")
    half_step = min(time_step_checks, key=lambda result: result.spec.time_step)

    refinement_deltas = [
        abs(current.summary["terminal_eps"] - previous.summary["terminal_eps"])
        for previous, current in zip(spatial, spatial[1:])
    ]
    previous_delta = refinement_deltas[-2]
    final_delta = refinement_deltas[-1]
    time_step_delta = abs(
        half_step.summary["terminal_eps"] - fine.summary["terminal_eps"]
    )
    deltas_shrink = final_delta < previous_delta
    terminal_values = [result.summary["terminal_eps"] for result in spatial]
    terminal_differences = [
        current - previous
        for previous, current in zip(terminal_values, terminal_values[1:])
    ]
    monotonic = all(value >= 0.0 for value in terminal_differences) or all(
        value <= 0.0 for value in terminal_differences
    )
    time_step_is_comparable = time_step_delta >= 0.5 * final_delta
    all_results = spatial + [half_step]
    contact_invariants_hold = all(
        result.summary["hydro_min"] == 1
        and result.summary["hydro_max"] == 1
        and result.summary["point_max"] == 0
        for result in all_results
    )
    if not monotonic and time_step_is_comparable:
        verdict = (
            "This is stabilization in a narrow band, not clean asymptotic "
            "convergence: the spatial sequence is non-monotonic and the "
            "time-step effect is comparable to the final spatial change."
        )
    elif deltas_shrink and monotonic and not time_step_is_comparable:
        verdict = (
            "The sampled results show consistent spatial stabilization, "
            "though more resolutions would be needed for an observed order."
        )
    else:
        verdict = "The sampled runs do not yet establish spatial convergence."
    contact_text = (
        "Every affine run maintained one hydroelastic contact and zero point "
        "contacts."
        if contact_invariants_hold
        else "At least one affine run violated the expected contact invariant."
    )

    position_rms, yaw_rms = _trajectory_difference(tet_rows, fine.rows)
    eps_chart = _line_chart(
        [
            ("Tet", _active_eps(tet_rows), "tet-line"),
            ("Affine SDF (finest)", _active_eps(fine.rows), "voxel-line"),
        ],
        "eps = |v| / (|omega| R)",
        (
            (_EPS_CONTINUUM, "continuum 0.653"),
            (_EPS_DISCRETE, "discrete 0.644"),
        ),
    )
    load_chart = _line_chart(
        [
            ("Tet", _normal_ratio(tet_rows), "tet-line"),
            ("Affine SDF (finest)", _normal_ratio(fine.rows), "voxel-line"),
        ],
        "normal force / mg",
        ((1.0, "mg"),),
    )
    convergence_points = sorted(
        (
            result.spec.target_voxel_size * 1e3,
            result.summary["terminal_eps"],
        )
        for result in spatial
    )
    convergence_chart = _line_chart(
        [("Affine SDF", convergence_points, "voxel-line")],
        "terminal eps",
        (
            (_EPS_CONTINUUM, "continuum 0.653"),
            (_EPS_DISCRETE, "discrete 0.644"),
        ),
        "target voxel size [mm]",
    )

    physics_rows = []
    surface_rows = []
    for index, result in enumerate(spatial):
        delta = "—" if index == 0 else _format(refinement_deltas[index - 1])
        summary = result.summary
        physics_rows.append(
            "<tr>"
            f"<td>{result.spec.target_voxel_size * 1e3:g}</td>"
            f"<td>{_format(summary['terminal_eps'])}</td>"
            f"<td>{_format(summary['continuum_error'])}</td>"
            f"<td>{delta}</td>"
            f"<td>{_format(summary['mean_normal_ratio'])}</td>"
            f"<td>{_format(summary['max_normal_ratio_error'])}</td>"
            f"<td>{_format(summary['first_contact_area'])}</td>"
            "</tr>"
        )
        surface_rows.append(
            "<tr>"
            f"<td>{result.spec.target_voxel_size * 1e3:g}</td>"
            f"<td>{summary['first_surface_vertices']}</td>"
            f"<td>{summary['first_surface_faces']}</td>"
            f"<td>{summary['hydro_min']}–{summary['hydro_max']}</td>"
            f"<td>{summary['point_max']}</td>"
            "</tr>"
        )
    physics_rows = "".join(physics_rows)
    surface_rows = "".join(surface_rows)

    def comparison_row(label, key, digits=6):
        return (
            f"<tr><td>{label}</td><td>{_format(tet[key], digits)}</td>"
            f"<td>{_format(fine.summary[key], digits)}</td>"
            f"<td>{_format(half_step.summary[key], digits)}</td></tr>"
        )

    comparison_rows = "".join(
        [
            comparison_row("terminal eps", "terminal_eps"),
            comparison_row("terminal eps - continuum", "continuum_error"),
            comparison_row("stop time [s]", "stop_time"),
            comparison_row("mean normal force / mg", "mean_normal_ratio"),
            comparison_row("final x [m]", "final_x"),
            comparison_row("final yaw [rad]", "final_yaw"),
        ]
    )
    provenance = "<br>\n".join(
        f"Affine h={result.spec.target_voxel_size * 1e3:g} mm, "
        f"dt={result.spec.time_step * 1e3:g} ms: "
        f"<code>{result.spec.csv_path}</code>"
        for result in results
    )
    fine_h_mm = fine.spec.target_voxel_size * 1e3
    fine_dt_ms = fine.spec.time_step * 1e3
    half_dt_ms = half_step.spec.time_step * 1e3
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Disk Farkas: affine-SDF convergence study</title>
<style>
:root {{
  --bg:#fff; --fg:#1d2228; --muted:#626b75; --card:#f6f8fa;
  --border:#ccd2d9; --grid:#e7ebef; --tet:#44617a;
  --voxel:#c45a35; --ref:#71802d;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#15181c; --fg:#e8eaed; --muted:#aab1b9; --card:#1d2228;
    --border:#414850; --grid:#292f35; --tet:#80a9c8;
    --voxel:#ef8b68; --ref:#b4c56a;
  }}
}}
* {{ box-sizing:border-box }}
body {{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,sans-serif;
}}
main {{ max-width:900px; margin:auto; padding:34px 22px 60px }}
h1 {{ font-size:25px; margin:0 }}
h2 {{
  font-size:18px; margin-top:30px;
  border-bottom:1px solid var(--border);
}}
.sub,.note {{ color:var(--muted) }}
.summary {{
  background:var(--card); border:1px solid var(--border);
  border-radius:9px; padding:13px 16px; margin:20px 0;
}}
table {{ border-collapse:collapse; width:100% }}
.table-scroll {{ overflow-x:auto }}
th,td {{
  padding:7px 10px; border-bottom:1px solid var(--grid); text-align:right;
  white-space:nowrap;
}}
th:first-child,td:first-child {{ text-align:left }}
.plot {{ width:100%; height:auto }}
.frame {{ fill:none; stroke:var(--border) }}
.grid {{ stroke:var(--grid) }}
.tick,.axis {{ fill:var(--muted); font-size:11px }}
polyline {{ fill:none; stroke-width:2 }}
.tet-line {{ stroke:var(--tet) }}
.voxel-line {{ stroke:var(--voxel) }}
.tet-line-label {{ fill:var(--tet); font-size:11px }}
.voxel-line-label {{ fill:var(--voxel); font-size:11px }}
.reference {{ stroke:var(--ref); stroke-dasharray:5 4 }}
.reference-label {{ fill:var(--ref); font-size:10px }}
code {{ background:var(--card); padding:1px 4px; border-radius:3px }}
</style>
<main>
<h1>Sliding and spinning disk: affine-SDF convergence study</h1>
<p class="sub">
Farkas benchmark · compliant Cylinder on compliant Box · SAP (lagged),
polygon contact · unchanged initial overlap
</p>
<div class="summary"><b>Convergence evidence.</b>
At fixed dt={fine_dt_ms:g} ms, the final spatial-refinement change in terminal
eps is {_format(final_delta)}, compared with {_format(previous_delta)} on the
previous refinement. Halving the finest-grid time step to {half_dt_ms:g} ms
changes terminal eps by {_format(time_step_delta)}. {verdict} The finest fixed
dt result remains {_format(fine.summary["continuum_error"])} from the continuum
reference. {contact_text}
</div>
<h2>Spatial refinement at dt={fine_dt_ms:g} ms</h2>
<div class="table-scroll"><table>
<thead><tr>
<th>target h [mm]</th><th>terminal eps</th><th>eps - 0.653</th>
<th>change from prior h</th><th>mean N/mg</th><th>max |N/mg-1|</th>
<th>first area [m²]</th>
</tr></thead>
<tbody>{physics_rows}</tbody>
</table></div>
<p class="note">
The terminal statistic deliberately uses the last sample with |omega| &gt;=
{_OMEGA_TERMINAL_CUTOFF} rad/s. Near-zero post-stop samples are excluded
because eps becomes ill-conditioned.
</p>
{convergence_chart}
<h2>Surface complexity and contact invariants</h2>
<table>
<thead><tr><th>target h [mm]</th><th>first-frame vertices</th>
<th>first-frame polygons</th><th>hydro contacts</th>
<th>max point contacts</th></tr></thead>
<tbody>{surface_rows}</tbody>
</table>
<h2>Time-step check and tet control</h2>
<div class="table-scroll"><table>
<thead><tr><th>Metric</th><th>Tet h={tet_spec.target_voxel_size * 1e3:g} mm,
dt={tet_spec.time_step * 1e3:g} ms</th>
<th>Affine h={fine_h_mm:g} mm, dt={fine_dt_ms:g} ms</th>
<th>Affine h={fine_h_mm:g} mm, dt={half_dt_ms:g} ms</th></tr></thead>
<tbody>{comparison_rows}</tbody>
</table></div>
<h2>Finest-grid slip/spin evolution</h2>{eps_chart}
<h2>Finest-grid normal-load consistency</h2>{load_chart}
<p class="note">
Against the tet control, the finest affine dt={fine_dt_ms:g} ms trajectory has
RMS differences of {_format(position_rms)} m in translation and
{_format(yaw_rms)} rad in yaw.
</p>
<h2>Provenance</h2>
<p class="note">
Tet CSV: <code>{tet_spec.csv_path}</code><br>
{provenance}<br>
Disk mass: {_MASS:.9g} kg; mg: {_MG:.9g} N. The unchanged initial overlap was
calibrated by the original tet experiment; no representation-specific height
retuning was applied.
</p>
</main>"""


def main():
    args = _parse_args()
    tet_rows = _read_csv(args.tet_csv)
    tet_spec = RunSpec(
        args.tet_target_voxel_size, args.tet_time_step, args.tet_csv
    )
    html = _make_html(tet_rows, tet_spec, args.voxel_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
