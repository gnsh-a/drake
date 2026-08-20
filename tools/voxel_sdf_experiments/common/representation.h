#pragma once

#include <string_view>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

enum class Representation {
  kTet,
  kPlaneClip,
  kMarchingCubes,
  // Marching cubes with the patch rim solved for instead of interpolated.
  kMarchingCubesExactRim,
};

Representation ParseRepresentation(std::string_view value);

/* True for either marching-cubes kernel. The two differ only in where they
 place the patch rim, so everything downstream -- the triangle surface type,
 the dual-grid offset -- is shared. */
bool IsMarchingCubes(Representation representation);
std::string_view to_string(Representation representation);

geometry::ProximityProperties MakeProperties(Representation representation,
                                             double resolution,
                                             double hydroelastic_modulus);

geometry::HydroelasticContactRepresentation SurfaceTypeFor(
    Representation representation);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
