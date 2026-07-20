"""Interactively displays the affine-SDF Farkas disk contact surface."""

import argparse
from pathlib import Path

import pyvista as pv


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path(
            "tools/hydro_compare/out/sdf_disk_farkas/"
            "lagged_voxel_sdf_h0p625mm_dt1ms_surface.vtk"
        ),
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="render off-screen to this PNG instead of opening a window",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    surface = pv.read(args.surface)
    if "pressure" not in surface.point_data:
        raise RuntimeError(f"{args.surface} has no point-data pressure array")
    surface.point_data["pressure_kPa"] = surface.point_data["pressure"] / 1e3

    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.010874868),
        direction=(0.0, 0.0, 1.0),
        radius=0.01213,
        height=0.00175,
        resolution=128,
        capping=True,
    )
    box = pv.Box(bounds=(-0.1, 0.1, -0.1, 0.1, -0.01, 0.01))

    plotter = pv.Plotter(
        off_screen=args.screenshot is not None,
        window_size=(1200, 900),
    )
    plotter.set_background("white")
    maximum = max(float(surface.point_data["pressure_kPa"].max()), 1e-12)
    plotter.add_mesh(
        surface,
        scalars="pressure_kPa",
        clim=(0.0, maximum),
        cmap="viridis",
        show_edges=True,
        edge_color="black",
        line_width=0.7,
        scalar_bar_args={"title": "Pressure [kPa]"},
    )
    plotter.add_mesh(
        cylinder,
        style="wireframe",
        color="tomato",
        opacity=0.55,
        line_width=2,
        label="Cylinder",
    )
    plotter.add_mesh(
        box,
        style="wireframe",
        color="deepskyblue",
        opacity=0.25,
        line_width=1,
        label="Box",
    )
    plotter.add_text(
        "Farkas disk: primitive-affine voxel-SDF contact\n"
        f"{surface.n_points} vertices, {surface.n_cells} polygons",
        font_size=12,
    )
    plotter.add_legend(loc="lower left")
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera(bounds=(-0.018, 0.018, -0.018, 0.018, 0.007, 0.013))

    if args.screenshot is None:
        plotter.show(title="Affine-SDF Farkas disk contact")
    else:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(args.screenshot))
        print(f"Wrote {args.screenshot}")


if __name__ == "__main__":
    main()
