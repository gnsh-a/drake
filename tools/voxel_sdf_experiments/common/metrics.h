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
  /* Raw summed face area, with no projection onto the reference normal. It
   exceeds projected_area by however much the surface tilts away from that
   normal, so comparing the two isolates face orientation from face size.
   It is compared against the reference's unprojected surface area, so unlike
   area_relative_error it does not confound tilt with discretization error. */
  double total_area{};
  double reference_surface_area{};
  double total_area_relative_error{};
  double patch_radius{};
  double reference_patch_radius{};
  double patch_radius_relative_error{};

  int num_faces{};
  int num_vertices{};

  Eigen::Vector3d centroid_W{Eigen::Vector3d::Zero()};
  double centroid_position_error{};

  /* Pressure-weighted centre of pressure, as distinct from the area-weighted
   centroid above. The two coincide on a flat patch but not on a curved one, so
   this is measured against the reference's pressure-weighted centroid rather
   than its area-weighted one. */
  Eigen::Vector3d center_of_pressure_W{Eigen::Vector3d::Zero()};
  double center_of_pressure_error{};

  /* Magnitude of the moment of the contact traction about the reference
   centroid. Axisymmetry makes the true value zero, so whatever is reported is
   entirely discretization asymmetry. Also given normalized by the reference
   force and patch radius, which makes it comparable across resolutions. */
  double spurious_moment{};
  double spurious_moment_normalized{};

  double largest_component_area_fraction{};
  int num_components{};
};

Metrics CalcMetrics(const SurfaceView& surface, const Reference& reference);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
