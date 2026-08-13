#pragma once

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/name_value.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

struct DiskSceneConfig {
  struct Disk {
    double radius{};
    double thickness{};
    double hydroelastic_modulus{};
    Eigen::Vector3d initial_position{Eigen::Vector3d::Zero()};
    std::optional<Eigen::Vector3d> initial_linear_velocity;
    std::optional<Eigen::Vector3d> initial_angular_velocity;
    std::optional<double> density;

    template <typename Archive>
    void Serialize(Archive* archive) {
      archive->Visit(DRAKE_NVP(radius));
      archive->Visit(DRAKE_NVP(thickness));
      archive->Visit(DRAKE_NVP(hydroelastic_modulus));
      archive->Visit(DRAKE_NVP(initial_position));
      archive->Visit(DRAKE_NVP(initial_linear_velocity));
      archive->Visit(DRAKE_NVP(initial_angular_velocity));
      archive->Visit(DRAKE_NVP(density));
    }
  };

  struct Box {
    Eigen::Vector3d full_size{Eigen::Vector3d::Zero()};
    double hydroelastic_modulus{};

    template <typename Archive>
    void Serialize(Archive* archive) {
      archive->Visit(DRAKE_NVP(full_size));
      archive->Visit(DRAKE_NVP(hydroelastic_modulus));
    }
  };

  struct Material {
    double friction{};
    std::optional<double> relaxation_time;

    template <typename Archive>
    void Serialize(Archive* archive) {
      archive->Visit(DRAKE_NVP(friction));
      archive->Visit(DRAKE_NVP(relaxation_time));
    }
  };

  struct Mesh {
    double sdf_target_voxel_size{};

    template <typename Archive>
    void Serialize(Archive* archive) {
      archive->Visit(DRAKE_NVP(sdf_target_voxel_size));
    }
  };

  Disk disk;
  Box box;
  Material material;
  Mesh mesh;

  template <typename Archive>
  void Serialize(Archive* archive) {
    archive->Visit(DRAKE_NVP(disk));
    archive->Visit(DRAKE_NVP(box));
    archive->Visit(DRAKE_NVP(material));
    archive->Visit(DRAKE_NVP(mesh));
  }
};

struct DiskFarkasConfig {
  DiskSceneConfig scene;
  Representation representation{Representation::kPlaneClip};
  double resolution{0.0025};
  double time_step{1.25e-4};
  double settle_time{0.05};
  double settle_time_step{6.25e-5};
  double frames_per_second{2000.0};
  int num_frames{300};
  std::filesystem::path output;
};

struct DiskFarkasRow {
  double time{};
  Eigen::Vector3d position_WD{Eigen::Vector3d::Zero()};
  double qx{};
  double qy{};
  double qz{};
  double qw{};
  Eigen::Vector3d angular_velocity_WD{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_velocity_WD{Eigen::Vector3d::Zero()};
  double angular_speed{};
  double linear_speed{};
  int point_contacts{};
  int hydro_contacts{};
  Eigen::Vector3d contact_force_W{Eigen::Vector3d::Zero()};
  double contact_area{};
  int surface_vertices{};
  int surface_faces{};
  double normal_force_z{};
  double friction_force_x{};
  double friction_force_y{};
  double friction_torque_z{};
  double eps{};
  bool post_kick{};
};

struct DiskFarkasResult {
  double mass{};
  double weight{};
  bool contact_acquired{};
  int hydro_contact_samples{};
  int post_kick_contact_samples{};
  double settled_normal_load{};
  double settled_normal_load_relative_error{};
  bool used_distinct_settle_plant{};
  int transferred_discrete_state_groups{};
  bool state_transfer_exact{};
  double initial_epsilon{};
  double terminal_epsilon{};
  double initial_linear_speed{};
  double final_linear_speed{};
  double initial_spin_speed{};
  double final_spin_speed{};
  std::vector<DiskFarkasRow> rows;
};

DiskSceneConfig LoadDiskScene(const std::filesystem::path& path);
std::string_view DiskFarkasCsvHeader();
DiskFarkasResult RunDiskFarkas(const DiskFarkasConfig& config);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
