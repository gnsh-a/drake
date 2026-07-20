# Affine-SDF disk Farkas comparison

This package ports the sliding-and-spinning disk benchmark from the `mc`
worktree and runs the same compliant Cylinder-on-compliant-Box scene with
either Drake's tetrahedral hydroelastic representation or the primitive-affine
voxel-SDF representation. Both representations use zero hydroelastic margin,
polygon contact, the same initial overlap, and the same SAP (lagged) time grid.

Run the tet control, affine-SDF runs at target voxel sizes 2.5, 1.25, and
0.625 mm, a 0.5 ms time-step check at the finest grid, and their first
output-frame contact surfaces:

```bash
python3 tools/sdf_disk_farkas/run_comparison.py
```

Each case writes a trajectory CSV plus companion `<stem>_sap_stats.csv` and
`<stem>_tamsi_stats.csv` solver-statistics files (the SAP stats are populated
under the lagged approximation, TAMSI under tamsi). Analyze the CSVs directly.

The generated files are written beneath the already-ignored local artifact
directory `tools/hydro_compare/out/sdf_disk_farkas/`. The driver never clears
that directory; it only overwrites its explicitly named CSV and VTK outputs.

Open the affine-SDF contact surface interactively:

```bash
uv run --with pyvista python tools/sdf_disk_farkas/view_surface.py
```

Each trajectory CSV samples the terminal slip/spin statistic `eps = |v| /
(|omega| R)` per frame; treat as terminal the last sample with
`|omega| >= 0.1 rad/s`, because `eps` is ill-conditioned after the disk nearly
stops.
