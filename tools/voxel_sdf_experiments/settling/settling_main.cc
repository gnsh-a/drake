#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include <gflags/gflags.h>

#include "drake/tools/voxel_sdf_experiments/common/representation.h"
#include "drake/tools/voxel_sdf_experiments/settling/settling.h"

DEFINE_string(scene, "sphere_sphere", "Scene: sphere_sphere or sphere_box.");
DEFINE_string(representation, "plane_clip",
              "Representation: tet, plane_clip, or marching_cubes.");
DEFINE_double(mass, 0.0,
              "Free-sphere mass in kilograms. When omitted, the scene's "
              "analytic 19.9 mm equilibrium load is used.");
DEFINE_double(voxel_width, 0.01, "Voxel-SDF cell width in meters.");
DEFINE_double(tet_resolution_hint, 0.01,
              "Tet hydroelastic resolution hint in meters.");
DEFINE_double(hydroelastic_modulus, 1.0e5,
              "Hydroelastic modulus for both bodies in pascals.");
DEFINE_double(initial_gap, 0.001, "Initial gap above touching in meters.");
DEFINE_double(time_step, 1.0e-4, "Discrete plant time step in seconds.");
DEFINE_double(dissipation, 3.0, "Hunt-Crossley dissipation in seconds/meter.");
DEFINE_double(duration, 0.0,
              "Simulation duration in seconds; 0 selects 15 natural periods.");
DEFINE_double(settling_window, 0.0,
              "Final aggregation window in seconds; 0 selects 1.25 periods.");
DEFINE_double(grid_roll_deg, 0.0, "Moving-sphere grid roll in degrees.");
DEFINE_double(grid_pitch_deg, 0.0, "Moving-sphere grid pitch in degrees.");
DEFINE_double(grid_yaw_deg, 0.0, "Moving-sphere grid yaw in degrees.");
DEFINE_string(output, "settling.csv", "One-row summary CSV path.");
DEFINE_string(trajectory, "", "Optional per-step trajectory CSV path.");
DEFINE_string(mesh_output, "",
              "Optional settled contact-surface VTK POLYDATA path.");

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int DoMain(bool mass_was_set) {
  SettlingConfig config;
  config.scene = ParseSettlingScene(FLAGS_scene);
  config.representation = ParseRepresentation(FLAGS_representation);
  config.hydroelastic_modulus = FLAGS_hydroelastic_modulus;
  config.mass = mass_was_set ? FLAGS_mass
                             : DefaultSettlingMass(config.scene,
                                                   config.hydroelastic_modulus);
  config.voxel_width = FLAGS_voxel_width;
  config.tet_resolution_hint = FLAGS_tet_resolution_hint;
  config.initial_gap = FLAGS_initial_gap;
  config.time_step = FLAGS_time_step;
  config.dissipation = FLAGS_dissipation;
  config.duration = FLAGS_duration;
  config.settling_window = FLAGS_settling_window;
  config.grid_rpy_deg = Eigen::Vector3d(
      FLAGS_grid_roll_deg, FLAGS_grid_pitch_deg, FLAGS_grid_yaw_deg);
  config.output = std::filesystem::path(FLAGS_output);
  config.trajectory = std::filesystem::path(FLAGS_trajectory);
  config.mesh_output = std::filesystem::path(FLAGS_mesh_output);

  const SettlingResult result = RunSettling(config);
  std::cout << "wrote " << FLAGS_output << ": scene=" << to_string(config.scene)
            << ", representation=" << to_string(config.representation)
            << ", m_hat=" << result.derived.elements_across_patch
            << ", equilibrium_penetration_m=" << result.equilibrium_penetration
            << ", penetration_error_m=" << result.penetration_error
            << ", support_force_relative_error="
            << result.mean_support_force_relative_error
            << ", first_contact_time_s=";
  if (result.first_contact_time.has_value()) {
    std::cout << *result.first_contact_time;
  } else {
    std::cout << "n/a";
  }
  std::cout << '\n';
  if (!FLAGS_trajectory.empty()) {
    std::cout << "wrote trajectory " << FLAGS_trajectory << '\n';
  }
  if (!FLAGS_mesh_output.empty()) {
    std::cout << "wrote mesh " << FLAGS_mesh_output << '\n';
  }
  return 0;
}

bool CommandLineSetsMass(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--mass" || argument.starts_with("--mass=") ||
        argument == "-mass" || argument.starts_with("-mass=")) {
      return true;
    }
  }
  return false;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  const bool mass_was_set =
      drake::tools::voxel_sdf_experiments::CommandLineSetsMass(argc, argv);
  gflags::SetUsageMessage("Run one free-body lagged-SAP settling experiment.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::voxel_sdf_experiments::DoMain(mass_was_set);
  } catch (const std::exception& error) {
    std::cerr << "settling: " << error.what() << '\n';
    return 1;
  }
}
