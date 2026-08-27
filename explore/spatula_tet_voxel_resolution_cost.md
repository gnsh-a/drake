# Spatula Tet and Voxel-SDF Resolution Cost

The spatula-slip experiment contains three compliant collision geometries:
two identical bubble-finger ellipsoids and one spatula-handle cylinder.

| Representation | Bubble setting | Spatula setting | Each bubble | Spatula | Total elements |
|---|---:|---:|---:|---:|---:|
| Tet | 40 mm resolution hint | 5 mm resolution hint | 128 tets | 95 tets | **351 tets** |
| Voxel coarse | 40 mm voxel width | 5 mm voxel width | 36 cells | 792 cells | **864 cells** |
| Voxel fine | 20 mm voxel width | 2.5 mm voxel width | 175 cells | 6,336 cells | **6,686 cells** |
| Voxel finer | 10 mm voxel width | 1.25 mm voxel width | 1,053 cells | 50,688 cells | **52,794 cells** |

The tet values are meshing hints, not uniform tetrahedron widths. Voxel widths
are the exact cubic cell widths.

## Observed Contact Counts

Each trajectory contains 751 samples over 30 seconds.

| Representation | Samples with two spatula contacts | Samples with zero spatula contacts | Accumulated spatula contacts |
|---|---:|---:|---:|
| Tet | 751 | 0 | 1,502 |
| Voxel coarse | 583 | 168 | 1,166 |
| Voxel fine | 751 | 0 | 1,502 |
| Voxel finer | 751 | 0 | 1,502 |

The coarse voxel-SDF trajectory loses both spatula contacts at 23.32 seconds.
Tet, fine voxel-SDF, and finer voxel-SDF retain both contacts for the complete
trajectory.

Contact-surface count is an outcome, not a direct computational-cost measure.
Voxel cells and tetrahedra are also not equivalent units. A runtime comparison
would additionally require per-case wall time and preferably the number of
generated contact polygons or visited candidate cells.
