#pragma once

#include <filesystem>
#include <string_view>

#include "drake/common/eigen_types.h"
#include "drake/tools/voxel_sdf_experiments/common/metrics.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

enum class Scene { kSphereSphere, kSphereBox };

Scene ParseScene(std::string_view value);
std::string_view to_string(Scene scene);

struct FrozenSurfaceConfig {
  Scene scene{Scene::kSphereSphere};
  Representation representation{Representation::kPlaneClip};
  double penetration{0.0199};
  double voxel_width{0.01};
  double tet_resolution_hint{0.01};
  double radius{0.1};
  Eigen::Vector3d box_half_widths{0.2, 0.2, 0.1};
  double modulus_a{1.0e8};
  double modulus_b{1.0e8};
  Eigen::Vector3d grid_rpy_deg{Eigen::Vector3d::Zero()};
  /* When non-empty, the contact surface is also written here as a legacy VTK
   POLYDATA file. Empty by default so sweeps do not emit meshes unasked. */
  std::filesystem::path mesh_output;
};

Metrics RunOne(const FrozenSurfaceConfig& config);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
