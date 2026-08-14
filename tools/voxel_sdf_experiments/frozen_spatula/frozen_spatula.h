#pragma once

#include <array>
#include <filesystem>
#include <limits>
#include <string_view>

#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

enum class SpatulaPose { kDemo, kFirstTouch };

SpatulaPose ParseSpatulaPose(std::string_view value);
std::string_view to_string(SpatulaPose pose);

struct FrozenSpatulaConfig {
  SpatulaPose pose{SpatulaPose::kDemo};
  /* Inward translation from the selected pose along the approach direction. */
  double penetration{0.0};
  /* Multiplier on each collision geometry's SDF resolution hint. */
  double resolution_scale{1.0};
  /* Common resolution hint for both shapes in FineTetReference. */
  double fine_tet_resolution_hint{4.8828125e-6};
  /* When non-empty, one VTK file per contacting representation is written. */
  std::filesystem::path mesh_output_dir;
};

struct FineSpatulaReferenceValues {
  bool available{false};
  double resolution_hint{};
  double force{};
  double projected_area{};
  Eigen::Vector3d centroid_C{Eigen::Vector3d::Zero()};
  double construction_wall_time{};
};

struct FrozenSpatulaMetrics {
  bool in_contact{false};
  bool is_triangle{false};
  int num_vertices{};
  int num_faces{};
  double total_surface_area{};
  double projected_area{};
  double reference_projected_area{std::numeric_limits<double>::quiet_NaN()};
  double area_relative_error{std::numeric_limits<double>::quiet_NaN()};
  double pressure_integral{};
  double force_norm{};
  double reference_force{std::numeric_limits<double>::quiet_NaN()};
  double force_relative_error{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Vector3d force_on_cylinder_C{Eigen::Vector3d::Zero()};
  double normal_force{};
  double transverse_force_norm{};
  double min_pressure{};
  double max_pressure{};
  double surface_distance_rms{std::numeric_limits<double>::quiet_NaN()};
  double surface_distance_max{std::numeric_limits<double>::quiet_NaN()};
  double pressure_error_rms{std::numeric_limits<double>::quiet_NaN()};
  double pressure_error_max{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Vector3d centroid_C{Eigen::Vector3d::Zero()};
  double centroid_position_error{std::numeric_limits<double>::quiet_NaN()};
  int connected_components{};
  double largest_component_area_fraction{};
  bool has_nonfinite{false};
  bool has_negative_pressure{false};
};

struct FrozenSpatulaRepresentationResult {
  Representation representation{Representation::kTet};
  FrozenSpatulaMetrics metrics;
};

struct FrozenSpatulaResult {
  FrozenSpatulaConfig config;
  math::RigidTransformd X_CE_demo;
  math::RigidTransformd X_CE_first_touch;
  math::RigidTransformd X_CE;
  Eigen::Vector3d approach_direction_C{Eigen::Vector3d::Zero()};
  double demo_directional_penetration{};
  double realized_directional_penetration{};
  double rigid_signed_distance{};
  double ellipsoid_base_resolution{};
  double cylinder_base_resolution{};
  double ellipsoid_resolution{};
  double cylinder_resolution{};
  double ellipsoid_modulus{};
  double cylinder_modulus{};
  FineSpatulaReferenceValues reference;
  std::array<FrozenSpatulaRepresentationResult, 3> representations;
};

/* Loads the gripper and spatula SDFormat assets and runs all three contact
 representations at one frozen pose. */
FrozenSpatulaResult RunFrozenSpatula(const FrozenSpatulaConfig& config);

/* Constructs only the fine-tet reference at the requested pose. This is the
 narrow seam used by the opt-in accuracy-floor measurement gtest. */
FineSpatulaReferenceValues MeasureFrozenSpatulaReference(
    const FrozenSpatulaConfig& config, double fine_tet_resolution_hint);

std::string_view FrozenSpatulaCsvHeader();
void EmitFrozenSpatulaCsv(const std::filesystem::path& path,
                          const FrozenSpatulaResult& result);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
