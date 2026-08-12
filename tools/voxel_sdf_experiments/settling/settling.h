#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

#include "drake/common/eigen_types.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

enum class SettlingScene { kSphereSphere, kSphereBox };

SettlingScene ParseSettlingScene(std::string_view value);
std::string_view to_string(SettlingScene scene);

struct SettlingConfig {
  SettlingScene scene{SettlingScene::kSphereSphere};
  Representation representation{Representation::kPlaneClip};
  double mass{0.0};
  double voxel_width{0.01};
  double tet_resolution_hint{0.01};
  double hydroelastic_modulus{1.0e5};
  double initial_gap{0.001};
  double time_step{1.0e-4};
  double dissipation{3.0};
  double duration{0.0};
  double settling_window{0.0};
  Eigen::Vector3d grid_rpy_deg{Eigen::Vector3d::Zero()};
  std::filesystem::path output;
  std::filesystem::path trajectory;
  std::filesystem::path mesh_output;
};

struct SettlingDerived {
  double weight{};
  double analytic_equilibrium_penetration{};
  double contact_stiffness{};
  double natural_period{};
  double duration{};
  double settling_window{};
  double patch_radius{};
  double elements_across_patch{};
  double contact_plane_voxel_phase{};
};

struct SettlingResult {
  SettlingDerived derived;
  double equilibrium_penetration{};
  double penetration_error{};
  double penetration_relative_error{};
  double mean_support_force{};
  double mean_support_force_relative_error{};
  double penetration_span{};
  double max_abs_axial_velocity{};
  double max_lateral_offset{};
  double max_angular_speed{};
  double mean_faces{};
  double mean_contact_area{};
  double largest_component_area_fraction{};
  double max_penetration{};
  std::optional<double> first_contact_time;
  std::optional<double> first_contact_loss_time;
  int64_t steps{};
  double wall_time{};
  bool axial_velocity_settled{};
  bool penetration_span_settled{};
  bool settled{};
};

/* The default load is chosen from the analytic force at 19.9 mm. It is
 scene-dependent, so direct binary invocations of both scenes remain at the
 same operating penetration without transcribed force constants. */
double DefaultSettlingMass(SettlingScene scene, double hydroelastic_modulus);

SettlingDerived CalcSettlingDerived(const SettlingConfig& config);
std::string_view SettlingCsvHeader();
SettlingResult RunSettling(const SettlingConfig& config);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
