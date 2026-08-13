#include <exception>
#include <filesystem>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/tools/voxel_sdf_experiments/common/representation.h"
#include "drake/tools/voxel_sdf_experiments/disk_farkas/disk_farkas.h"

DEFINE_string(scene, "tools/voxel_sdf_experiments/disk_farkas/disk_plane.yaml",
              "Disk scene YAML path.");
DEFINE_string(representation, "plane_clip",
              "Representation: tet, plane_clip, or marching_cubes.");
DEFINE_double(resolution, 0.0,
              "Tet resolution hint or voxel width in meters; 0 uses YAML.");
DEFINE_double(time_step, 1.25e-4, "Measurement-plant time step in seconds.");
DEFINE_double(settle_time, 0.05, "Pre-kick settling duration in seconds.");
DEFINE_double(settle_time_step, 6.25e-5,
              "Fixed settling-plant time step in seconds.");
DEFINE_double(fps, 2000.0, "Trajectory samples per second.");
DEFINE_int32(num_frames, 300, "Number of post-kick trajectory frames.");
DEFINE_string(output,
              "tools/voxel_sdf_experiments/out/disk_farkas/disk_farkas.csv",
              "Trajectory CSV path.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int DoMain() {
  DiskFarkasConfig config;
  config.scene = LoadDiskScene(FLAGS_scene);
  config.representation = ParseRepresentation(FLAGS_representation);
  config.resolution = FLAGS_resolution > 0.0
                          ? FLAGS_resolution
                          : config.scene.mesh.sdf_target_voxel_size;
  config.time_step = FLAGS_time_step;
  config.settle_time = FLAGS_settle_time;
  config.settle_time_step = FLAGS_settle_time_step;
  config.frames_per_second = FLAGS_fps;
  config.num_frames = FLAGS_num_frames;
  config.output = std::filesystem::path(FLAGS_output);

  const DiskFarkasResult result = RunDiskFarkas(config);
  std::cout << "wrote " << config.output
            << ": representation=" << to_string(config.representation)
            << ", settled_load_over_mg="
            << result.settled_normal_load / result.weight
            << ", eps=" << result.initial_epsilon << " -> "
            << result.terminal_epsilon << '\n';
  return 0;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Run the free six-DOF sliding-and-spinning disk infrastructure demo.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::voxel_sdf_experiments::DoMain();
  } catch (const std::exception& error) {
    std::cerr << "disk_farkas: " << error.what() << '\n';
    return 1;
  }
}
