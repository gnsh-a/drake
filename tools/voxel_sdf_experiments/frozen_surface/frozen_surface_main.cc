#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <gflags/gflags.h>

#include "drake/tools/voxel_sdf_experiments/common/emit.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"
#include "drake/tools/voxel_sdf_experiments/frozen_surface/frozen_surface.h"

DEFINE_string(scene, "sphere_sphere", "Scene: sphere_sphere or sphere_box.");
DEFINE_string(representation, "plane_clip",
              "Representation: tet, plane_clip, or marching_cubes.");
DEFINE_double(penetration, 0.0199, "Commanded penetration in meters.");
DEFINE_double(voxel_width, 0.01, "Voxel-SDF cell width in meters.");
DEFINE_double(tet_resolution_hint, 0.01,
              "Tet hydroelastic resolution hint in meters.");
DEFINE_double(radius, 0.1, "Sphere radius in meters.");
DEFINE_double(box_half_width_x, 0.2, "Box x half-width in meters.");
DEFINE_double(box_half_width_y, 0.2, "Box y half-width in meters.");
DEFINE_double(box_half_width_z, 0.1, "Box z half-width in meters.");
DEFINE_double(hydroelastic_modulus_a, 1.0e8,
              "Hydroelastic modulus for body A in pascals.");
DEFINE_double(hydroelastic_modulus_b, 1.0e8,
              "Hydroelastic modulus for body B in pascals.");
DEFINE_double(grid_roll_deg, 0.0, "Body-A grid roll in degrees.");
DEFINE_double(grid_pitch_deg, 0.0, "Body-A grid pitch in degrees.");
DEFINE_double(grid_yaw_deg, 0.0, "Body-A grid yaw in degrees.");
DEFINE_string(output, "frozen_surface.csv", "Per-run output CSV path.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int DoMain() {
  FrozenSurfaceConfig config;
  config.scene = ParseScene(FLAGS_scene);
  config.representation = ParseRepresentation(FLAGS_representation);
  config.penetration = FLAGS_penetration;
  config.voxel_width = FLAGS_voxel_width;
  config.tet_resolution_hint = FLAGS_tet_resolution_hint;
  config.radius = FLAGS_radius;
  config.box_half_widths = Eigen::Vector3d(
      FLAGS_box_half_width_x, FLAGS_box_half_width_y, FLAGS_box_half_width_z);
  config.modulus_a = FLAGS_hydroelastic_modulus_a;
  config.modulus_b = FLAGS_hydroelastic_modulus_b;
  config.grid_rpy_deg = Eigen::Vector3d(
      FLAGS_grid_roll_deg, FLAGS_grid_pitch_deg, FLAGS_grid_yaw_deg);
  const Metrics metrics = RunOne(config);

  RunRecord record;
  record.metadata = {
      .scene = std::string(to_string(config.scene)),
      .representation = std::string(to_string(config.representation)),
      .penetration = config.penetration,
      .voxel_width = config.voxel_width,
      .tet_resolution_hint = config.tet_resolution_hint,
      .radius = config.radius,
      .box_half_widths = config.box_half_widths,
      .modulus_a = config.modulus_a,
      .modulus_b = config.modulus_b,
      .grid_rpy_deg = config.grid_rpy_deg,
  };
  record.provenance = ReadGitProvenance();
  record.metrics = metrics;
  EmitCsv(std::filesystem::path(FLAGS_output), record);
  std::cout << "wrote " << FLAGS_output << ": faces=" << metrics.num_faces
            << ", force_relative_error=" << metrics.normal_force_relative_error
            << ", surface_rms_m=" << metrics.surface_distance_rms << '\n';
  return 0;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Run one frozen voxel-SDF contact-surface query.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::voxel_sdf_experiments::DoMain();
  } catch (const std::exception& error) {
    std::cerr << "frozen_surface: " << error.what() << '\n';
    return 1;
  }
}
