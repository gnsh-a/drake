#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
DEFINE_string(init_linear_velocity, "",
              "Override the scene's post-settle slide kick, as \"x,y,z\" in "
              "m/s. Together with --init_angular_velocity this is what lets a "
              "driver sweep the initial slip-to-spin ratio without minting a "
              "YAML per case.");
DEFINE_string(init_angular_velocity, "",
              "Override the scene's post-settle spin kick, as \"x,y,z\" in "
              "rad/s.");
DEFINE_string(box_full_size, "",
              "Override the ground box size, as \"x,y,z\" in meters. A large "
              "initial slip-to-spin ratio slides the disk further, so the "
              "default 0.2 m ground is not wide enough for the upper end of an "
              "eps0 sweep. Widening it costs memory: the voxel grid is sized "
              "by the box, not by the disk.");
DEFINE_string(output,
              "tools/voxel_sdf_experiments/out/disk_farkas/disk_farkas.csv",
              "Trajectory CSV path.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

/* Parses an "x,y,z" triple. Returns nullopt for an empty string, which is how
 every override flag spells "leave the scene value alone". */
std::optional<Eigen::Vector3d> ParseOptionalVector3(std::string_view name,
                                                    const std::string& value) {
  if (value.empty()) return std::nullopt;
  Eigen::Vector3d parsed;
  size_t start = 0;
  for (int index = 0; index < 3; ++index) {
    const size_t comma = value.find(',', start);
    const bool last = (index == 2);
    if (last != (comma == std::string::npos)) {
      throw std::runtime_error(std::string(name) +
                               " expects exactly three comma-separated "
                               "values, for example 0.2,0,0");
    }
    parsed[index] = std::stod(value.substr(start, comma - start));
    start = last ? start : comma + 1;
  }
  if (!parsed.allFinite()) {
    throw std::runtime_error(std::string(name) + " must be finite");
  }
  return parsed;
}

int DoMain() {
  DiskFarkasConfig config;
  config.scene = LoadDiskScene(FLAGS_scene);
  config.representation = ParseRepresentation(FLAGS_representation);
  config.resolution = FLAGS_resolution > 0.0
                          ? FLAGS_resolution
                          : config.scene.mesh.sdf_target_voxel_size;
  if (const auto velocity = ParseOptionalVector3("--init_linear_velocity",
                                                 FLAGS_init_linear_velocity)) {
    config.scene.disk.initial_linear_velocity = *velocity;
  }
  if (const auto rate = ParseOptionalVector3("--init_angular_velocity",
                                             FLAGS_init_angular_velocity)) {
    config.scene.disk.initial_angular_velocity = *rate;
  }
  if (const auto size =
          ParseOptionalVector3("--box_full_size", FLAGS_box_full_size)) {
    if (!(size->minCoeff() > 0.0)) {
      throw std::runtime_error("--box_full_size entries must be positive");
    }
    config.scene.box.full_size = *size;
  }
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
