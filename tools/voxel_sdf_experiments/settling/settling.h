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

/* The penetration the default load is chosen to produce. The value is
 0.08 / 3, picked so the equal-pressure plane never lands on a voxel cell
 boundary: over a dyadic ladder delta / h is 8/3 times a power of two, and a
 1/3 offset is invariant under halving h, so the plane stays 1/3 of a cell off
 a boundary at every rung. Pass a different target to put it deliberately on
 one -- 0.02 m sits exactly on an affine cell boundary at every dyadic rung.
 No single value does that for marching cubes, whose degeneracy is at
 delta = (2i + 1) h: halving h turns an odd multiple into an even one. */
constexpr double kDefaultEquilibriumPenetration = 0.08 / 3.0;

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
  /* Positive starts above contact; negative starts at the corresponding
   penetration. */
  double initial_gap{0.001};
  /* Constrain the moving sphere to world-z translation. This isolates the
   axial force equilibrium from the sphere-on-sphere lateral instability. */
  bool axial_only{false};
  double time_step{1.0e-4};
  double dissipation{3.0};
  double duration{0.0};
  double settling_window{0.0};
  /* Duration and window in natural periods, used when the corresponding
   absolute time above is zero. The body reaches its final penetration to
   within a micrometre by about three periods, so the default carries a wide
   margin; shortening it is the cheapest way to buy sweep breadth. */
  double duration_periods{15.0};
  double settling_window_periods{1.25};
  /* Write every Nth trajectory sample. Asking for a trajectory at all forces
   the surface view and its connected-component pass to run at every step
   rather than only inside the settled window, which is the dominant cost of a
   trajectory run; striding buys that back. Comparison against a reference
   requires every run in a study to use the same stride and time step. */
  int trajectory_stride{1};
  Eigen::Vector3d grid_rpy_deg{Eigen::Vector3d::Zero()};
  std::filesystem::path output;
  std::filesystem::path trajectory;
  std::filesystem::path mesh_output;
  /* Optional directory for color frames rendered by Drake during the actual
   * simulation. Empty disables rendering. */
  std::filesystem::path frames_dir;
  double frame_period{0.01};
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
  double faces_span{};
  double mean_contact_area{};
  double contact_area_span{};
  double largest_component_area_fraction{};
  double mean_num_components{};
  /* SAP effort over the settled window. A representation that makes the
   contact problem harder to solve shows up here rather than in the
   equilibrium, which is pinned to the load. */
  double mean_sap_iters{};
  double max_sap_iters{};
  double sap_nonconverged_steps{};
  double max_sap_momentum_residual{};
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
double DefaultSettlingMass(
    SettlingScene scene, double hydroelastic_modulus,
    double target_penetration = kDefaultEquilibriumPenetration);

SettlingDerived CalcSettlingDerived(const SettlingConfig& config);
std::string_view SettlingCsvHeader();
SettlingResult RunSettling(const SettlingConfig& config);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
