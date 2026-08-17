#pragma once

#include <filesystem>
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
  Eigen::Vector3d right_contact_force_W{Eigen::Vector3d::Zero()};
  double right_handle_axis_torque{};
  double right_contact_area{};
  int right_surface_faces{};
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
  std::vector<SpatulaSlipRow> rows;
};

/* Runs one dynamic representation of the gripper-spatula slip scene. */
SpatulaSlipResult RunSpatulaSlip(const SpatulaSlipConfig& config);

std::string_view SpatulaSlipCsvHeader();

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
