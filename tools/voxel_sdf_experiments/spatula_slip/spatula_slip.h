#pragma once

#include <filesystem>
#include <limits>
#include <string_view>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

struct SpatulaSlipConfig {
  Representation representation{Representation::kPlaneClip};
  /* Multiplier on the parsed resolution hint of all three compliant shapes. */
  double resolution_scale{1.0};
  double time_step{0.04};
  double sample_period{0.04};
  double duration{30.0};
  double stiction_tolerance{1.0e-4};
  double gripper_force{1.5};
  double amplitude{5.0};
  double duty_cycle{0.5};
  double period{3.0};
  std::filesystem::path output;
  /* Optional directory for color frames rendered by Drake during the actual
   * simulation. Empty disables rendering. */
  std::filesystem::path frames_dir;
};

struct SpatulaSlipRow {
  double time{};
  Eigen::Vector3d position_WB{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quaternion_WB{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d handle_axis_W{Eigen::Vector3d::UnitZ()};
  double left_finger_position{};
  double right_finger_position{};
  int point_contacts{};
  int hydro_contacts{};
  int spatula_contacts{};
  Eigen::Vector3d left_contact_force_W{Eigen::Vector3d::Zero()};
  double left_handle_axis_torque{};
  double left_contact_area{};
  int left_surface_faces{};
  int left_num_components{};
  double left_largest_component_area_fraction{};
  Eigen::Vector3d right_contact_force_W{Eigen::Vector3d::Zero()};
  double right_handle_axis_torque{};
  double right_contact_area{};
  int right_surface_faces{};
  int right_num_components{};
  double right_largest_component_area_fraction{};
  /* Effort the solver spent on the step ending at this sample. This is the
   paper's only frictional stick-slip scene, so it is the only one where the
   count has room to vary; the other two studies found it pinned at one. */
  int sap_iters{};
  int sap_line_search_iters{};
  bool sap_converged{true};
};

struct SpatulaSlipResult {
  double ellipsoid_base_resolution{};
  double cylinder_base_resolution{};
  double ellipsoid_resolution{};
  double cylinder_resolution{};
  int contact_samples{};
  int two_finger_contact_samples{};
  int triangle_surface_samples{};
  double max_position_displacement{};
  double max_handle_axis_swing{};
  double max_contact_force{};
  double max_abs_handle_axis_torque{};
  double max_contact_area{};
  /* Whether the gripper still held the spatula at the end, and when it first
   stopped holding it. A dropped run has no meaningful trajectory error: its
   position diverges as free fall, so an RMS against a reference measures the
   length of the fall rather than an accuracy. Analysis must branch on this
   rather than on the size of the number. */
  bool held{true};
  double first_contact_loss_time{std::numeric_limits<double>::quiet_NaN()};
  /* How far the spatula slid down the gripper's handle axis while held. This
   is the quantity the task is about, and unlike a trajectory RMS it stays
   comparable when one representation drops the object partway through. */
  double handle_axis_slip{};
  int max_num_components{};
  double min_largest_component_area_fraction{1.0};
  int max_sap_iters{};
  int sap_nonconverged_steps{};
  std::vector<SpatulaSlipRow> rows;
};

/* Runs one dynamic representation of the gripper-spatula slip scene. */
SpatulaSlipResult RunSpatulaSlip(const SpatulaSlipConfig& config);

std::string_view SpatulaSlipCsvHeader();

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
