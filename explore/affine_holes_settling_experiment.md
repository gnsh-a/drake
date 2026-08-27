# Effect of Primitive-Affine Contact Holes on Settling

## Aim

Determine whether primitive-affine contact holes merely affect visualization or
materially change contact force, dynamics, penetration, and equilibrium.

## Experiment

We compared tet and primitive-affine contact for identical sphere-sphere systems
using lagged SAP, gravity, Hunt-Crossley damping, and zero friction.

We used two initial conditions:

- Released from a 1 mm gap.
- Released already compressed at `delta = 20.2 mm`.

The second condition tests whether equilibrium depends on which side of the
voxel-boundary defect the system starts.

The sphere-box experiment serves as a general tet-versus-affine dynamics
control. The aligned sphere-sphere experiment is the targeted test because it
deliberately drives the contact plane through the verified hole-producing voxel
boundary.

## Findings

- Tet remained one continuous surface and settled at `delta ≈ 20.000 mm` from
  both initial conditions.
- Near `delta = 20 mm`, the affine surface fragmented into 93 connected
  components and had about 11.2% less contact area than tet.
- Affine settled to two different equilibria under the same load:

  ```text
  Released from gap:       19.459 mm
  Started at 20.2 mm:      20.132 mm
  Difference:               0.674 mm = 6.74% of h
  ```

- Both representations still balanced the same `28.372 N` load and the falling
  cases settled in approximately `0.43 s`.
- The sphere-box comparison produced a smaller, single equilibrium-penetration
  shift of about `0.268 mm = 2.68% of h`, without the same severe fragmentation.

## Boundary-Coverage Verification

The boundary-aligned affine contact surface is not simply one disk counted
twice. Only the lower-ID sphere's grid is traversed; the other sphere is
evaluated analytically at each host-cell center. The competing polygons
therefore come from two adjacent cells of the same host grid, not independently
from both spheres' grids.

For the `R = 1 m`, `h = 0.1 m`, `delta = 0.2 m` diagnostic, the exact
equal-pressure plane is `x = 0.9`, between host cells `i = 18` and `i = 19`:

```text
left voxel i=18       shared face       right voxel i=19
     [0.8, 0.9]          x=0.9              [0.9, 1.0]
```

Each cell independently constructs a tangent-affine sphere field. At a
representative hole, the resulting equal-pressure roots were

```text
left-cell root  = 0.90152398  -> outside the left voxel
right-cell root = 0.89847602  -> outside the right voxel
```

Neither cell accepts a polygon there, so the surface is missing. Elsewhere the
inequalities reverse: the left root lies inside the left voxel and the right
root lies inside the right voxel. Both polygons are then accepted, producing
two slightly separated sheets.

The production kernel suppresses an exact coincident boundary copy only when
its geometry, pressure, normal, and pressure gradients agree with an already
accepted polygon. The sphere sheets generally differ in position or field data,
so they are not equivalent duplicates and are both retained.

The dedicated three-dimensional boundary diagnostic was rerun successfully. It
measured:

```text
polygons:             120
missing coverage:     57.127%
single coverage:       0.000%
double coverage:      42.873%
```

The diagnostic compared this local cell-pair classification with the complete
production contact surface at more than 100,000 projected points and found zero
mismatches. Replacing the two independent local planes with one shared plane
changed the result to 0% missing, 100% single coverage, and 0% overlap. This
confirms that the gap/overlap pattern is caused by neighboring local-plane
mismatch rather than duplicate suppression or rasterization.

The double-covered regions contribute twice to surface integration, but they do
not compensate for the missing regions:

```text
affine integrated pressure: 2.510789 MN
tet integrated pressure:    2.837220 MN
difference:                 -11.505%
```

Thus the precise description is:

```text
Not: the entire contact disk is duplicated in both voxels.

Yes: adjacent voxels independently create inconsistent sheets.
     Some locations receive two sheets.
     Other locations receive no sheet.
```

## Inference

The holes are not merely visual. The underlying cell-local affine discontinuity
changes the force-penetration relation, makes it non-monotonic, and creates
history-dependent equilibrium.

Final force alone hides the problem because any equilibrium must balance
gravity. Penetration, contact-surface connectivity, and sensitivity to initial
conditions reveal it.

The experiment does not isolate holes from the associated pressure and normal
errors. They are all consequences of the same discontinuous primitive-affine
approximation.
