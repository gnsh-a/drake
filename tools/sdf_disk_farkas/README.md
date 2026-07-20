# Affine-SDF disk Farkas comparison

This package ports the sliding-and-spinning disk benchmark from the `mc`
worktree and runs the same compliant Cylinder-on-compliant-Box scene with
either Drake's tetrahedral hydroelastic representation or the primitive-affine
voxel-SDF representation. Both representations use zero hydroelastic margin,
polygon contact, the same initial overlap, and the same SAP (lagged) time grid.

Generate the tet control, affine-SDF runs at target voxel sizes 2.5, 1.25, and
0.625 mm, a 0.5 ms time-step check at the finest grid, their first output-frame
contact surfaces, and the self-contained HTML report:

```bash
python3 tools/sdf_disk_farkas/run_comparison.py
```

The generated files are written beneath the already-ignored local artifact
directory `tools/hydro_compare/out/sdf_disk_farkas/`. The driver never clears
that directory; it only overwrites its explicitly named CSV, VTK, and HTML
outputs.

Open the affine-SDF contact surface interactively:

```bash
uv run --with pyvista python tools/sdf_disk_farkas/view_surface.py
```

The report compares spatial-refinement increments, contact-surface complexity,
normal-load consistency, contact invariants, and the finest-grid time-step
sensitivity. It defines the terminal slip/spin statistic as the last sample
with `|omega| >= 0.1 rad/s`; later samples are excluded because
`eps = |v| / (|omega| R)` is ill-conditioned after the disk nearly stops.
