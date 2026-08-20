#include "drake/tools/voxel_sdf_experiments/common/representation.h"

#include <stdexcept>
#include <string>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

Representation ParseRepresentation(std::string_view value) {
  if (value == "tet") return Representation::kTet;
  if (value == "plane_clip") return Representation::kPlaneClip;
  if (value == "marching_cubes") return Representation::kMarchingCubes;
  if (value == "marching_cubes_exact_rim") {
    return Representation::kMarchingCubesExactRim;
  }
  throw std::logic_error("Unknown representation '" + std::string(value) +
                         "'; expected tet, plane_clip, marching_cubes, or "
                         "marching_cubes_exact_rim");
}

bool IsMarchingCubes(Representation representation) {
  return representation == Representation::kMarchingCubes ||
         representation == Representation::kMarchingCubesExactRim;
}

std::string_view to_string(Representation representation) {
  switch (representation) {
    case Representation::kTet:
      return "tet";
    case Representation::kPlaneClip:
      return "plane_clip";
    case Representation::kMarchingCubes:
      return "marching_cubes";
    case Representation::kMarchingCubesExactRim:
      return "marching_cubes_exact_rim";
  }
  throw std::logic_error("Invalid Representation value");
}

geometry::ProximityProperties MakeProperties(Representation representation,
                                             double resolution,
                                             double hydroelastic_modulus) {
  geometry::ProximityProperties properties;
  if (representation == Representation::kTet) {
    geometry::AddCompliantHydroelasticProperties(
        resolution, hydroelastic_modulus, &properties);
    return properties;
  }
  geometry::VoxelSdfExtractionMethod extraction_method =
      geometry::VoxelSdfExtractionMethod::kPlaneClip;
  if (representation == Representation::kMarchingCubes) {
    extraction_method = geometry::VoxelSdfExtractionMethod::kMarchingCubes;
  } else if (representation == Representation::kMarchingCubesExactRim) {
    extraction_method =
        geometry::VoxelSdfExtractionMethod::kMarchingCubesExactRim;
  }
  geometry::AddCompliantHydroelasticVoxelSdfProperties(
      resolution, hydroelastic_modulus,
      geometry::VoxelSdfEvaluationMode::kPrimitiveSdf, extraction_method,
      &properties);
  return properties;
}

geometry::HydroelasticContactRepresentation SurfaceTypeFor(
    Representation representation) {
  return IsMarchingCubes(representation)
             ? geometry::HydroelasticContactRepresentation::kTriangle
             : geometry::HydroelasticContactRepresentation::kPolygon;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
