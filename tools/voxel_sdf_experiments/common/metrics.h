#pragma once

#include "drake/common/eigen_types.h"
#include "drake/tools/voxel_sdf_experiments/common/reference.h"
#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

struct Metrics {
  double surface_distance_rms{};
  double surface_distance_max{};

  double normal_force{};
  double reference_force{};
  double normal_force_relative_error{};

  double pressure_error_rms{};
  double pressure_error_max{};
  double peak_pressure{};
  double reference_peak_pressure{};
  double peak_pressure_relative_error{};

  double projected_area{};
  double reference_area{};
  double area_relative_error{};
  double patch_radius{};
  double reference_patch_radius{};
  double patch_radius_relative_error{};

  int num_faces{};
  int num_vertices{};

  Eigen::Vector3d centroid_W{Eigen::Vector3d::Zero()};
  double centroid_position_error{};

  double largest_component_area_fraction{};
};

Metrics CalcMetrics(const SurfaceView& surface, const Reference& reference);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
