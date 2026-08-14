#include <exception>
#include <filesystem>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/tools/voxel_sdf_experiments/frozen_spatula/frozen_spatula.h"

DEFINE_string(pose, "demo", "Frozen pose: demo or first_touch.");
DEFINE_double(penetration, 0.0,
              "Inward translation in meters from the selected pose.");
DEFINE_double(resolution_scale, 1.0,
              "Scale on the SDF-provided resolution hint of each shape.");
DEFINE_double(fine_tet_resolution_hint, 4.8828125e-6,
              "Common fine-tet reference resolution hint in meters.");
DEFINE_string(
    output, "tools/voxel_sdf_experiments/out/frozen_spatula/frozen_spatula.csv",
    "Three-row output CSV path.");
DEFINE_string(mesh_output_dir, "",
              "When set, write one VTK contact surface per representation.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int DoMain() {
  FrozenSpatulaConfig config;
  config.pose = ParseSpatulaPose(FLAGS_pose);
  config.penetration = FLAGS_penetration;
  config.resolution_scale = FLAGS_resolution_scale;
  config.fine_tet_resolution_hint = FLAGS_fine_tet_resolution_hint;
  config.mesh_output_dir = std::filesystem::path(FLAGS_mesh_output_dir);
  const FrozenSpatulaResult result = RunFrozenSpatula(config);
  EmitFrozenSpatulaCsv(std::filesystem::path(FLAGS_output), result);
  std::cout << "wrote " << FLAGS_output << ": pose=" << to_string(config.pose)
            << ", rigid_distance_m=" << result.rigid_signed_distance
            << ", reference="
            << (result.reference.available ? "available" : "unavailable")
            << '\n';
  for (const FrozenSpatulaRepresentationResult& representation :
       result.representations) {
    const FrozenSpatulaMetrics& metrics = representation.metrics;
    std::cout << "  " << to_string(representation.representation)
              << ": contact=" << metrics.in_contact
              << ", triangles=" << metrics.is_triangle
              << ", faces=" << metrics.num_faces
              << ", force_relative_error=" << metrics.force_relative_error
              << ", area_relative_error=" << metrics.area_relative_error
              << '\n';
  }
  if (!FLAGS_mesh_output_dir.empty()) {
    std::cout << "wrote meshes under " << FLAGS_mesh_output_dir << '\n';
  }
  return 0;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Compare tet, plane-clip, and marching-cubes contact surfaces for the "
      "frozen gripper-spatula cylinder/ellipsoid pair.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::voxel_sdf_experiments::DoMain();
  } catch (const std::exception& error) {
    std::cerr << "frozen_spatula: " << error.what() << '\n';
    return 1;
  }
}
