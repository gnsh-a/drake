#include <exception>
#include <filesystem>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/tools/voxel_sdf_experiments/common/representation.h"
#include "drake/tools/voxel_sdf_experiments/spatula_slip/spatula_slip.h"

DEFINE_string(representation, "plane_clip",
              "Representation: tet, plane_clip, or marching_cubes.");
DEFINE_double(resolution_scale, 1.0,
              "Scale on every parsed compliant resolution hint.");
DEFINE_double(time_step, 0.04, "Discrete MultibodyPlant time step in seconds.");
DEFINE_double(sample_period, 0.04,
              "Trajectory sample period; an integer multiple of time_step.");
DEFINE_double(duration, 30.0, "Simulation duration in seconds.");
DEFINE_double(stiction_tolerance, 1.0e-4,
              "MultibodyPlant stiction tolerance in m/s.");
DEFINE_double(gripper_force, 1.5, "Baseline opposing finger force in N.");
DEFINE_double(amplitude, 5.0, "Square-wave finger-force amplitude in N.");
DEFINE_double(duty_cycle, 0.5, "Square-wave duty cycle.");
DEFINE_double(period, 3.0, "Square-wave period in seconds.");
DEFINE_string(output,
              "tools/voxel_sdf_experiments/out/spatula_slip/spatula_slip.csv",
              "Trajectory CSV path; empty disables CSV output.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int DoMain() {
  SpatulaSlipConfig config;
  config.representation = ParseRepresentation(FLAGS_representation);
  config.resolution_scale = FLAGS_resolution_scale;
  config.time_step = FLAGS_time_step;
  config.sample_period = FLAGS_sample_period;
  config.duration = FLAGS_duration;
  config.stiction_tolerance = FLAGS_stiction_tolerance;
  config.gripper_force = FLAGS_gripper_force;
  config.amplitude = FLAGS_amplitude;
  config.duty_cycle = FLAGS_duty_cycle;
  config.period = FLAGS_period;
  config.output = std::filesystem::path(FLAGS_output);
  const SpatulaSlipResult result = RunSpatulaSlip(config);
  std::cout << "representation=" << to_string(config.representation)
            << ", scale=" << config.resolution_scale
            << ", samples=" << result.rows.size()
            << ", contact_samples=" << result.contact_samples
            << ", two_finger_contact_samples="
            << result.two_finger_contact_samples
            << ", max_handle_axis_swing_rad=" << result.max_handle_axis_swing
            << ", max_contact_force_n=" << result.max_contact_force << '\n';
  if (!config.output.empty()) std::cout << "wrote " << config.output << '\n';
  return 0;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Run one dynamic gripper-spatula slip trajectory using tet, affine "
      "plane clipping, or marching cubes.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::voxel_sdf_experiments::DoMain();
  } catch (const std::exception& error) {
    std::cerr << "spatula_slip: " << error.what() << '\n';
    return 1;
  }
}
