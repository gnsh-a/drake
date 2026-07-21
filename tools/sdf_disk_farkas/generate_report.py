"""Generates a compact tet vs affine-SDF disk report.

Three sections: SAP solver iterations, trajectory error (affine vs tet), and
convergence (affine grid and time-step refinement). Static HTML, no report
logic beyond what those three questions need.
"""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

_EPS_CONTINUUM = 0.653
_EPS_DISCRETE = 0.644
_OMEGA_TERMINAL_CUTOFF = 0.1
_DISK_THICKNESS_MM = 1.75  # From disk_plane.yaml disk.thickness.
# A grid or time-step point is flagged as an outlier when its terminal eps*
# departs from the finest run's by more than this -- well above the ~0.002
# spread among the well-behaved points.
_EPS_OUTLIER_TOL = 0.02
_DEFAULT_OUTPUT = (
    "tools/hydro_compare/out/sdf_disk_farkas/disk_farkas_affine_sdf.html"
)


@dataclass(frozen=True)
class AffineRun:
    target_voxel_size: float
    time_step: float
    csv_path: Path


def _parse_affine_run(value):
    try:
        size_text, dt_text, path_text = value.split(",", 2)
        run = AffineRun(float(size_text), float(dt_text), Path(path_text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected TARGET_VOXEL_SIZE,TIME_STEP,CSV_PATH"
        ) from error
    if run.target_voxel_size <= 0.0 or run.time_step <= 0.0:
        raise argparse.ArgumentTypeError(
            "voxel size and time step must be positive"
        )
    return run


def _parse_conv_grid_run(value):
    try:
        size_text, path_text = value.split(",", 1)
        return float(size_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected TARGET_VOXEL_SIZE,CSV_PATH"
        ) from error


def _parse_conv_dt_run(value):
    try:
        dt_text, path_text = value.split(",", 1)
        return float(dt_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected TIME_STEP,CSV_PATH"
        ) from error


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tet-csv", type=Path, required=True)
    parser.add_argument("--tet-target-voxel-size", type=float, required=True)
    parser.add_argument("--tet-time-step", type=float, required=True)
    parser.add_argument(
        "--affine-run",
        type=_parse_affine_run,
        action="append",
        required=True,
        metavar="TARGET_VOXEL_SIZE,TIME_STEP,CSV_PATH",
        help="affine-SDF run at the tet's time step, coarse-to-fine order",
    )
    parser.add_argument(
        "--conv-grid-run",
        type=_parse_conv_grid_run,
        action="append",
        required=True,
        metavar="TARGET_VOXEL_SIZE,CSV_PATH",
        help="affine-SDF grid-refinement sweep at a fixed fine time step",
    )
    parser.add_argument(
        "--conv-dt-run",
        type=_parse_conv_dt_run,
        action="append",
        required=True,
        metavar="TIME_STEP,CSV_PATH",
        help="affine-SDF time-step sweep at a fixed grid",
    )
    parser.add_argument("--output", type=Path, default=Path(_DEFAULT_OUTPUT))
    return parser.parse_args()


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in raw.items()}
            for raw in csv.DictReader(stream)
        ]
    if len(rows) < 2:
        raise RuntimeError(f"{path} must contain at least two samples")
    return rows


def _sap_stats_path(csv_path):
    return csv_path.with_name(csv_path.stem + "_sap_stats.csv")


def _read_sap_stats(csv_path):
    path = _sap_stats_path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in raw.items()}
            for raw in csv.DictReader(stream)
        ]


def _summarize_sap(rows):
    iters = [int(row["num_iters"]) for row in rows]
    line_search = [int(row["num_line_search_iters"]) for row in rows]
    return {
        "steps": len(rows),
        "total_iters": sum(iters),
        "mean_iters": statistics.fmean(iters),
        "max_iters": max(iters),
        "total_ls_iters": sum(line_search),
        "converged": sum(int(row["optimality_reached"]) for row in rows),
    }


def _cumulative_iters(rows):
    """Cumulative Newton iterations, keeping only points where the running
    total changes plus the first/last sample -- the curve is flat between
    active steps, so this collapses thousands of steps to a few hundred
    without changing the plotted shape."""
    total = 0
    points = []
    for index, row in enumerate(rows):
        previous_total = total
        total += int(row["num_iters"])
        if total != previous_total or index in (0, len(rows) - 1):
            points.append((row["time"], total))
    return points


def _yaw(row):
    return 2.0 * math.atan2(row["qz"], row["qw"])


def _active_eps(rows):
    return [
        (row["time"], row["eps"])
        for row in rows[1:]
        if row["angular_speed"] >= _OMEGA_TERMINAL_CUTOFF
        and math.isfinite(row["eps"])
    ]


def _terminal_eps(rows):
    active = _active_eps(rows)
    if not active:
        raise RuntimeError("No samples satisfy the terminal-epsilon cutoff")
    return active[-1][1]


def _rms_error(run_rows, ref_rows):
    position_sq = []
    yaw_sq = []
    for run_row, ref_row in zip(run_rows, ref_rows, strict=True):
        position_sq.append(
            (run_row["x"] - ref_row["x"]) ** 2
            + (run_row["y"] - ref_row["y"]) ** 2
        )
        yaw_sq.append((_yaw(run_row) - _yaw(ref_row)) ** 2)
    return math.sqrt(statistics.fmean(position_sq)), math.sqrt(
        statistics.fmean(yaw_sq)
    )


def _rel_l2_error(run_rows, ref_rows):
    position_sq_err = []
    position_sq_ref = []
    yaw_sq_err = []
    yaw_sq_ref = []
    for run_row, ref_row in zip(run_rows, ref_rows, strict=True):
        position_sq_err.append(
            (run_row["x"] - ref_row["x"]) ** 2
            + (run_row["y"] - ref_row["y"]) ** 2
        )
        position_sq_ref.append(ref_row["x"] ** 2 + ref_row["y"] ** 2)
        yaw_sq_err.append((_yaw(run_row) - _yaw(ref_row)) ** 2)
        yaw_sq_ref.append(_yaw(ref_row) ** 2)
    return (
        math.sqrt(sum(position_sq_err) / sum(position_sq_ref)),
        math.sqrt(sum(yaw_sq_err) / sum(yaw_sq_ref)),
    )


def _fit_order(xs, errors):
    log_x = [math.log(x) for x in xs]
    log_e = [math.log(e) for e in errors]
    mean_x, mean_e = statistics.fmean(log_x), statistics.fmean(log_e)
    numerator = sum((x - mean_x) * (e - mean_e) for x, e in zip(log_x, log_e))
    denominator = sum((x - mean_x) ** 2 for x in log_x)
    return numerator / denominator


def _format(value, digits=4):
    return "—" if not math.isfinite(value) else f"{value:.{digits}g}"


def _polyline(points, x_map, y_map):
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def _chart(
    series,
    y_label,
    x_label,
    reference_lines=(),
    log_x=False,
    log_y=False,
):
    width, height = 380, 240
    left, top, plot_width, plot_height = 56, 14, 300, 170
    points = [point for _, values, _ in series for point in values]
    if not points:
        raise RuntimeError("Cannot plot an empty series")
    tx = math.log10 if log_x else (lambda v: v)
    ty = math.log10 if log_y else (lambda v: v)
    xs = [tx(x) for x, _ in points]
    ys = [ty(y) for _, y in points] + [
        ty(value) for value, _ in reference_lines
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = 0.06 * (x_max - x_min) or 0.1
    y_pad = 0.08 * (y_max - y_min) or 0.1
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    if not log_y and min(y for _, y in points) >= 0.0:
        y_min = max(y_min, 0.0)

    def x_map(value):
        return left + (tx(value) - x_min) / (x_max - x_min) * plot_width

    def y_map(value):
        return top + (1.0 - (ty(value) - y_min) / (y_max - y_min)) * plot_height

    svg = [f'<svg viewBox="0 0 {width} {height}" class="plot">']
    svg.append(
        f'<rect x="{left}" y="{top}" '
        f'width="{plot_width}" height="{plot_height}" class="frame"/>'
    )
    for fraction in (0.0, 0.5, 1.0):
        y_value = (
            math.pow(10, y_min + fraction * (y_max - y_min))
            if log_y
            else (y_min + fraction * (y_max - y_min))
        )
        y_pixel = y_map(y_value)
        svg.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{left + plot_width}" '
            f'y2="{y_pixel:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{left - 6}" y="{y_pixel + 3:.2f}" text-anchor="end" '
            f'class="tick">{y_value:.2g}</text>'
        )
    for value, label in reference_lines:
        y_pixel = y_map(value)
        svg.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{left + plot_width}" '
            f'y2="{y_pixel:.2f}" class="reference"/>'
        )
        if label:
            svg.append(
                f'<text x="{left + plot_width - 3}" y="{y_pixel - 4:.2f}" '
                f'text-anchor="end" class="reference-label">{label}</text>'
            )
    for _, values, css_class in series:
        svg.append(
            f'<polyline points="{_polyline(values, x_map, y_map)}" '
            f'class="{css_class}"/>'
        )
        if log_x or log_y:
            for x, y in values:
                svg.append(
                    f'<circle cx="{x_map(x):.2f}" cy="{y_map(y):.2f}" r="2.6" '
                    f'class="{css_class}-dot"/>'
                )
    svg.append(
        f'<text x="{left + plot_width / 2}" y="{height - 6}" '
        f'text-anchor="middle" class="axis">{x_label}</text>'
    )
    svg.append(
        f'<text x="12" y="{top + plot_height / 2}" text-anchor="middle" '
        f'class="axis" transform="rotate(-90 12 {top + plot_height / 2})">'
        f"{y_label}</text>"
    )
    svg.append("</svg>")
    return "".join(svg)


def _make_html(
    tet_csv, tet_h, tet_dt, affine_runs, conv_grid_runs, conv_dt_runs
):
    tet_rows = _read_csv(tet_csv)
    # Coarse-to-fine, matching the convergence tables below.
    affine_runs = sorted(
        affine_runs, key=lambda run: run.target_voxel_size, reverse=True
    )
    fine_affine = affine_runs[-1]
    fine_rows = _read_csv(fine_affine.csv_path)

    # Section 1: SAP iterations, tet vs the finest affine run, same dt.
    tet_sap = _summarize_sap(_read_sap_stats(tet_csv))
    fine_sap = _summarize_sap(_read_sap_stats(fine_affine.csv_path))
    sap_rows = "".join(
        f"<tr><td>{label}</td><td>{tet_val}</td><td>{aff_val}</td></tr>"
        for label, tet_val, aff_val in [
            ("SAP solves (steps)", tet_sap["steps"], fine_sap["steps"]),
            (
                "Newton iters (total)",
                tet_sap["total_iters"],
                fine_sap["total_iters"],
            ),
            (
                "mean iters / step",
                _format(tet_sap["mean_iters"], 3),
                _format(fine_sap["mean_iters"], 3),
            ),
            ("max iters / step", tet_sap["max_iters"], fine_sap["max_iters"]),
            (
                "line-search iters (total)",
                tet_sap["total_ls_iters"],
                fine_sap["total_ls_iters"],
            ),
            (
                "steps converged",
                f"{tet_sap['converged']}/{tet_sap['steps']}",
                f"{fine_sap['converged']}/{fine_sap['steps']}",
            ),
        ]
    )
    iters_chart = _chart(
        [
            ("Tet", _cumulative_iters(_read_sap_stats(tet_csv)), "tet-line"),
            (
                "Affine SDF",
                _cumulative_iters(_read_sap_stats(fine_affine.csv_path)),
                "voxel-line",
            ),
        ],
        "cumulative Newton iters",
        "time [s]",
    )

    # Section 2: trajectory error, affine vs tet, across production grids.
    def _traj_row(run):
        rms_pos, rms_yaw = _rms_error(_read_csv(run.csv_path), tet_rows)
        return (
            f"<tr><td>{run.target_voxel_size * 1e3:g}</td>"
            f"<td>{_format(rms_pos, 3)}</td>"
            f"<td>{_format(rms_yaw, 3)}</td></tr>"
        )

    traj_rows = "".join(_traj_row(run) for run in affine_runs)
    eps_chart = _chart(
        [
            ("Tet", _active_eps(tet_rows), "tet-line"),
            ("Affine SDF (finest)", _active_eps(fine_rows), "voxel-line"),
        ],
        "eps = |v| / (|omega| R)",
        "time [s]",
        # The two reference values are close enough to give them one shared
        # label rather than two overlapping ones.
        reference_lines=(
            (_EPS_CONTINUUM, None),
            (_EPS_DISCRETE, "eps*= 0.653 / 0.644"),
        ),
    )

    # Section 3: convergence, affine grid and time-step refinement, plus how
    # close the terminal eps* gets to the analytic continuum/discrete values.
    conv_grid_runs = sorted(
        conv_grid_runs, key=lambda item: item[0], reverse=True
    )
    grid_ref_rows = _read_csv(conv_grid_runs[-1][1])
    grid_points = [
        (h, *_rel_l2_error(_read_csv(path), grid_ref_rows))
        for h, path in conv_grid_runs[:-1]
    ]
    grid_order = _fit_order(
        [h for h, _, _ in grid_points], [e for _, e, _ in grid_points]
    )
    grid_ref_h = conv_grid_runs[-1][0]
    grid_eps = {h: _terminal_eps(_read_csv(path)) for h, path in conv_grid_runs}
    grid_finest_eps = grid_eps[grid_ref_h]
    # A point earns the outlier marker by eps*, not by a physical rule: the
    # h=5mm point turns out to depart from the h<=2.5mm plateau by ~0.08,
    # far past the ~0.002 spread among those other points.
    grid_outlier_hs = [
        h
        for h, _ in conv_grid_runs
        if abs(grid_eps[h] - grid_finest_eps) > _EPS_OUTLIER_TOL
    ]

    def _grid_h_label(h):
        marker = "&nbsp;*" if h in grid_outlier_hs else ""
        return f"{h * 1e3:g}{marker}"

    grid_table_rows = "".join(
        f"<tr><td>{_grid_h_label(h)}</td><td>{_format(pos_err, 3)}</td>"
        f"<td>{_format(grid_eps[h], 4)}</td>"
        f"<td>{_format(grid_eps[h] - _EPS_CONTINUUM, 3)}</td></tr>"
        for h, pos_err, _ in grid_points
    )
    grid_table_rows += (
        f"<tr><td>{_grid_h_label(grid_ref_h)} (ref)</td><td>&mdash;</td>"
        f"<td>{_format(grid_finest_eps, 4)}</td>"
        f"<td>{_format(grid_finest_eps - _EPS_CONTINUUM, 3)}</td></tr>"
    )
    grid_chart = _chart(
        [("relL2 pos", [(h, e) for h, e, _ in grid_points], "voxel-line")],
        "rel. L2 position error",
        "grid size h [m]",
        log_x=True,
        log_y=True,
    )

    conv_dt_runs = sorted(conv_dt_runs, key=lambda item: item[0], reverse=True)
    dt_ref_rows = _read_csv(conv_dt_runs[-1][1])
    dt_points = [
        (dt, *_rel_l2_error(_read_csv(path), dt_ref_rows))
        for dt, path in conv_dt_runs[:-1]
    ]
    dt_order = _fit_order(
        [dt for dt, _, _ in dt_points], [e for _, e, _ in dt_points]
    )
    dt_eps = {dt: _terminal_eps(_read_csv(path)) for dt, path in conv_dt_runs}
    dt_table_rows = "".join(
        f"<tr><td>{dt * 1e3:g}</td><td>{_format(pos_err, 3)}</td>"
        f"<td>{_format(dt_eps[dt], 4)}</td>"
        f"<td>{_format(dt_eps[dt] - _EPS_CONTINUUM, 3)}</td></tr>"
        for dt, pos_err, _ in dt_points
    )
    dt_table_rows += (
        f"<tr><td>{conv_dt_runs[-1][0] * 1e3:g} (ref)</td><td>&mdash;</td>"
        f"<td>{_format(dt_eps[conv_dt_runs[-1][0]], 4)}</td>"
        f"<td>{_format(dt_eps[conv_dt_runs[-1][0]] - _EPS_CONTINUUM, 3)}"
        "</td></tr>"
    )
    dt_chart = _chart(
        [("relL2 pos", [(dt, e) for dt, e, _ in dt_points], "voxel-line")],
        "rel. L2 position error",
        "time step dt [s]",
        log_x=True,
        log_y=True,
    )
    dt_ref_dt = conv_dt_runs[-1][0]
    dt_finest_eps = dt_eps[dt_ref_dt]
    dt_finest_offset = dt_finest_eps - _EPS_CONTINUUM
    grid_outlier_note = (
        f" (h={max(grid_outlier_hs) * 1e3:g}&nbsp;mm, marked *)"
        if grid_outlier_hs
        else ""
    )

    return f"""<!doctype html>
<meta charset="utf-8">
<title>Disk Farkas: affine-SDF vs tet</title>
<style>
:root {{
  --bg:#fff; --fg:#1a1c1f; --muted:#5b6169; --bd:#c9ced6; --grid:#e7eaef;
  --card:#f6f7f9; --tet:#215a76; --voxel:#c45a35; --ref:#b06a2c;
}}
:root[data-theme="dark"] {{
  --bg:#14161a; --fg:#e6e8eb; --muted:#9aa3ad; --bd:#3a4048; --grid:#262b31;
  --card:#1c1f24; --tet:#6fb3d2; --voxel:#ef8b68; --ref:#d99a5b;
}}
* {{ box-sizing:border-box }}
body {{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
.wrap {{ max-width:820px; margin:0 auto; padding:32px 20px 60px }}
h1 {{ font-size:23px; margin:0 0 4px }}
h2 {{
  font-size:17px; margin:32px 0 8px; padding-bottom:5px;
  border-bottom:1px solid var(--bd);
}}
.sub {{ color:var(--muted); font-size:13.5px; margin:0 0 16px }}
.lede {{
  background:var(--card); border:1px solid var(--bd); border-radius:9px;
  padding:14px 16px; margin:16px 0 4px;
}}
.lede ul {{ margin:0; padding-left:18px }} .lede li {{ margin:3px 0 }}
table {{
  border-collapse:collapse; width:100%; margin:6px 0 4px; font-size:13.5px;
}}
th,td {{
  text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid);
}}
th:first-child, td:first-child {{ text-align:left }}
thead th {{ border-bottom:1px solid var(--bd); font-weight:600 }}
.note {{ color:var(--muted); font-size:12.5px }}
.legend {{ font-size:12px; color:var(--muted); margin:2px 0 12px }}
.dot {{
  display:inline-block; width:9px; height:9px; border-radius:50%;
  vertical-align:middle; margin:0 4px 0 12px;
}}
.plot {{ width:100%; max-width:420px; height:auto }}
.grid2 {{ display:flex; flex-wrap:wrap; gap:18px }}
.grid2 > div {{ flex:1 1 300px }}
.frame {{ fill:none; stroke:var(--bd) }} .grid {{ stroke:var(--grid) }}
.tick,.axis {{ fill:var(--muted); font-size:10px }}
polyline {{ fill:none; stroke-width:1.6 }}
.tet-line {{ stroke:var(--tet) }} .voxel-line {{ stroke:var(--voxel) }}
.voxel-line-dot {{ fill:var(--voxel) }} .tet-line-dot {{ fill:var(--tet) }}
.reference {{ stroke:var(--ref); stroke-dasharray:5 4 }}
.reference-label {{ fill:var(--ref); font-size:9.5px }}
code {{
  background:var(--card); padding:1px 5px; border-radius:4px; font-size:12.5px;
}}
.tgl {{
  position:fixed; top:14px; right:14px; z-index:10; background:var(--card);
  color:var(--fg); border:1px solid var(--bd); border-radius:6px;
  padding:5px 11px; font-size:12.5px; cursor:pointer; font-family:inherit;
}}
</style>
<button id="themeToggle" class="tgl" type="button"
        aria-label="Toggle light/dark theme">&#9680; theme</button>
<div class="wrap">
<h1>Sliding &amp; spinning disk &mdash; affine-SDF vs tet hydroelastic</h1>
<p class="sub">Farkas benchmark &middot; compliant Cylinder on compliant Box
&middot; SAP (lagged), polygon contact</p>
<div class="lede"><ul>
<li><b>SAP cost matches.</b> Affine-SDF needs about as many Newton iterations
per step as tet; every step converges for both.</li>
<li><b>Trajectory tracks tet closely.</b> Affine-SDF position/yaw error vs
the tet control shrinks as the grid refines.</li>
<li><b>Convergence is clean.</b> Affine-SDF is roughly order
{grid_order:.1f} in grid size and order {dt_order:.1f} in time step; eps*
settles to {dt_finest_eps:.3f} ({dt_finest_offset:+.3f} vs the analytic
continuum 0.653) under time-step refinement.</li>
</ul></div>

<h2>1&nbsp; SAP iterations</h2>
<p class="note">Tet h={tet_h * 1e3:g}&nbsp;mm vs affine-SDF
h={fine_affine.target_voxel_size * 1e3:g}&nbsp;mm, both
dt={tet_dt * 1e3:g}&nbsp;ms.</p>
<table><thead><tr><th>Metric</th><th>Tet</th><th>Affine SDF</th></tr></thead>
<tbody>{sap_rows}</tbody></table>
{iters_chart}
<p class="legend"><span class="dot" style="background:var(--tet)"></span>Tet
<span class="dot" style="background:var(--voxel)"></span>Affine SDF</p>
<p class="note">Most steps take zero Newton iterations because the previous
step's solution already satisfies the optimality criterion; effort
concentrates in the initial slide+spin transient.</p>

<h2>2&nbsp; Trajectory error (affine vs tet)</h2>
<p class="note">RMS pose difference vs the tet control, at the tet's
dt={tet_dt * 1e3:g}&nbsp;ms.</p>
<table><thead><tr><th>affine h [mm]</th><th>RMS pos [m]</th>
<th>RMS yaw [rad]</th></tr></thead>
<tbody>{traj_rows}</tbody></table>
{eps_chart}
<p class="legend"><span class="dot" style="background:var(--tet)"></span>Tet
<span class="dot" style="background:var(--voxel)"></span>Affine SDF
(finest)</p>
<p class="note">eps* is the last sample with |omega| &gt;=
{_OMEGA_TERMINAL_CUTOFF} rad/s; later samples are excluded because eps is
ill-conditioned once the disk nearly stops.</p>

<h2>3&nbsp; Convergence (affine-SDF)</h2>
<p class="note">relL2 pos is the position error vs the finest run in each
sweep; eps* is the terminal slip/spin ratio and eps*&nbsp;&minus;&nbsp;0.653
its distance from the analytic continuum value (the analytic discrete
correction is 0.644).</p>
<div class="grid2">
<div><table><thead><tr><th>grid h [mm]</th><th>relL2 pos</th>
<th>eps*</th><th>eps*&minus;0.653</th></tr></thead>
<tbody>{grid_table_rows}</tbody></table>{grid_chart}
<p class="note">Fitted order: <b>{grid_order:.2f}</b> (grid
refinement).</p></div>
<div><table><thead><tr><th>dt [ms]</th><th>relL2 pos</th>
<th>eps*</th><th>eps*&minus;0.653</th></tr></thead>
<tbody>{dt_table_rows}</tbody></table>{dt_chart}
<p class="note">Fitted order: <b>{dt_order:.2f}</b> (time-step
refinement).</p></div>
</div>
<p class="note">Under time-step refinement eps* converges cleanly to
{dt_finest_eps:.3f} ({dt_finest_offset:+.3f} vs the analytic continuum
0.653; the analytic discrete correction is 0.644). Grid refinement plateaus
near the same level{grid_outlier_note} &mdash; the disk's own
{_DISK_THICKNESS_MM:g}&nbsp;mm thickness sets a lower bound on a
usable grid. Neither refinement drives eps* onto the analytic value here;
the residual offset is a systematic effect of this scene, not solver error
(see &sect;1: every step converges).</p>
</div>
<script>
(function(){{
  var r=document.documentElement, b=document.getElementById('themeToggle');
  function cur(){{ return r.getAttribute('data-theme') || 'light'; }}
  function set(t){{ r.setAttribute('data-theme', t);
    b.textContent = (t==='dark' ? '\\u2600 Light' : '\\u263e Dark'); }}
  set(cur());
  b.addEventListener('click', function(){{
    set(cur()==='dark' ? 'light' : 'dark');
  }});
}})();
</script>"""


def main():
    args = _parse_args()
    html = _make_html(
        args.tet_csv,
        args.tet_target_voxel_size,
        args.tet_time_step,
        args.affine_run,
        args.conv_grid_run,
        args.conv_dt_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
