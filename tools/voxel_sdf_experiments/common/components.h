#pragma once

#include <vector>

#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

/* Returns the vertex-coincidence tolerance used to decide whether two faces
 touch, scaled by the largest vertex coordinate magnitude in `surface`. */
double DefaultComponentTolerance(const SurfaceView& surface);

/* Labels each face of `surface` with the connected component it belongs to.
 Two faces are connected when they have vertices within `tolerance` of each
 other; coincident-but-duplicated vertices therefore still connect, because
 faces are grouped by vertex position rather than by vertex index.

 The returned vector is indexed by face and holds an arbitrary but consistent
 integer per component, so faces sharing a value share a component. Conforming
 meshes yield one component; a representation that emits independent per-cell
 faces yields many. */
std::vector<int> CalcFaceComponentIds(const SurfaceView& surface,
                                      double tolerance);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
