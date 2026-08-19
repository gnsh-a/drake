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

/* Fragmentation of one sampled contact surface.

 `largest_area_fraction` is the largest component's share of the total area and
 `num_components` is how many components there are. Both come from a single
 labelling pass because labelling is the expensive part, and they are reported
 together because neither is conclusive alone: a count that collapsed to one
 would leave the fraction reading a perfect 1.0, and a fraction near 1.0 with a
 large count means one component dominates a shower of slivers. */
struct ComponentStats {
  double largest_area_fraction{};
  int num_components{};
};

/* Returns the fragmentation of `surface`, whose faces sum to `total_area`.
 An empty surface has no components and a zero fraction. */
ComponentStats CalcComponentStats(const SurfaceView& surface,
                                  double total_area);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
