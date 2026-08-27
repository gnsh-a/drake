#include "drake/tools/voxel_sdf_experiments/settling/settling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/render/render_camera.h"
#include "drake/geometry/render_vtk/factory.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_statistics.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/event_status.h"
#include "drake/systems/sensors/image_writer.h"
#include "drake/systems/sensors/rgbd_sensor.h"
#include "drake/tools/voxel_sdf_experiments/common/components.h"
#include "drake/tools/voxel_sdf_experiments/common/emit.h"
#include "drake/tools/voxel_sdf_experiments/common/mesh_export.h"
#include "drake/tools/voxel_sdf_experiments/common/reference.h"
#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Vector3d;
using geometry::Box;
using geometry::ContactSurface;
using geometry::ProximityProperties;
using geometry::RenderEngineVtkParams;
using geometry::Sphere;
using geometry::render::ColorRenderCamera;
using geometry::render::DepthRenderCamera;
using math::RigidTransformd;
using math::RotationMatrixd;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::ContactModel;
using multibody::ContactResults;
using multibody::CoulombFriction;
using multibody::DiscreteContactApproximation;
using multibody::MultibodyPlant;
using multibody::PrismaticJoint;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::SpatialVelocity;
using multibody::contact_solvers::internal::SapStatistics;
using systems::Context;
using systems::DiagramBuilder;
using systems::EventStatus;
using systems::Simulator;
using systems::sensors::ImageRgba8U;
using systems::sensors::RgbdSensor;
using systems::sensors::SaveToPng;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kGravity = 9.81;
constexpr double kRadius = 0.1;
const Vector3d kBoxHalfWidths{0.2, 0.2, 0.1};
/* The default load is the one whose analytic equilibrium lands here. The value
 is 0.08 / 3, chosen so the equal-pressure plane never lands on a voxel cell
 boundary: the affine kernel degenerates when delta = 2 i h and marching cubes,
 which samples the dual grid, when delta = (2 i + 1) h, so any delta that is an
 integer multiple of h is degenerate for one of them. Since delta / h is 8/3
 times a power of two over a dyadic ladder, and a 1/3 offset is invariant under
 halving h, this one value stays 1/3 of a cell off a boundary at every rung.
 A boundary-aligned penetration is a deliberate robustness axis, not a default;
 see contact_plane_voxel_phase in the emitted row. */
constexpr double kDegreesToRadians = kPi / 180.0;
constexpr std::string_view kSettlingCsvHeader =
    "schema_version,git_commit,git_dirty,scene,representation,mass_kg,"
    "voxel_width_m,tet_resolution_hint_m,hydroelastic_modulus_pa,"
    "initial_gap_m,time_step_s,dissipation_s_m,duration_input_s,"
    "settling_window_input_s,trajectory_stride,"
    "grid_roll_deg,grid_pitch_deg,grid_yaw_deg,"
    "weight_N,analytic_equilibrium_penetration_m,contact_stiffness_N_m,"
    "natural_period_s,duration_s,settling_window_s,patch_radius_m,"
    "elements_across_patch,contact_plane_voxel_phase,"
    "equilibrium_penetration_m,penetration_error_m,"
    "penetration_relative_error,mean_support_force_N,"
    "mean_support_force_relative_error,penetration_span_m,"
    "max_abs_axial_velocity_m_s,max_lateral_offset_m,"
    "max_angular_speed_rad_s,mean_faces,faces_span,mean_contact_area_m2,"
    "contact_area_span_m2,largest_component_area_fraction,"
    "mean_num_components,mean_sap_iters,max_sap_iters,"
    "sap_nonconverged_steps,max_sap_momentum_residual,max_penetration_m,"
    "first_contact_time_s,first_contact_loss_time_s,steps,wall_time_s,"
    "axial_velocity_settled,penetration_span_settled,settled";
constexpr std::string_view kTrajectoryCsvHeader =
    "time_s,penetration_m,support_force_N,axial_velocity_m_s,"
    "lateral_offset_m,angular_speed_rad_s,faces,contact_area_m2,"
    "largest_component_area_fraction,num_components,sap_iters,"
    "sap_line_search_iters,sap_optimality,sap_cost,sap_momentum_residual,"
    "contact";

void ThrowUnlessFinitePositive(double value, std::string_view name) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and strictly positive");
  }
}

void ThrowUnlessFiniteNonnegative(double value, std::string_view name) {
  if (!(std::isfinite(value) && value >= 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and nonnegative");
  }
}

ReferenceFactory MakeReferenceFactory(SettlingScene scene, double modulus) {
  if (scene == SettlingScene::kSphereSphere) {
    return [modulus](double penetration) {
      return std::make_unique<AnalyticPlane>(kRadius, penetration, modulus,
                                             modulus);
    };
  }
  return [modulus](double penetration) {
    return std::make_unique<AnalyticParaboloid>(kRadius, penetration, modulus,
                                                modulus);
  };
}

void ValidateConfig(const SettlingConfig& config) {
  ThrowUnlessFinitePositive(config.mass, "mass");
  ThrowUnlessFinitePositive(config.voxel_width, "voxel_width");
  ThrowUnlessFinitePositive(config.tet_resolution_hint, "tet_resolution_hint");
  ThrowUnlessFinitePositive(config.hydroelastic_modulus,
                            "hydroelastic_modulus");
  ThrowUnlessFinitePositive(config.duration_periods, "duration_periods");
  if (config.trajectory_stride < 1) {
    throw std::logic_error("trajectory_stride must be at least one");
  }
  ThrowUnlessFinitePositive(config.settling_window_periods,
                            "settling_window_periods");
  if (!(std::isfinite(config.initial_gap) &&
        config.initial_gap > -2.0 * kRadius)) {
    throw std::logic_error(
        "initial_gap must be finite and greater than -2 * radius");
  }
  ThrowUnlessFinitePositive(config.time_step, "time_step");
  ThrowUnlessFinitePositive(config.frame_period, "frame_period");
  ThrowUnlessFiniteNonnegative(config.dissipation, "dissipation");
  ThrowUnlessFiniteNonnegative(config.duration, "duration");
  ThrowUnlessFiniteNonnegative(config.settling_window, "settling_window");
  if (!config.grid_rpy_deg.allFinite()) {
    throw std::logic_error("grid RPY angles must be finite");
  }
}

RigidTransformd MakeCameraPose() {
  const Vector3d p_WC(0.52, -0.58, 0.38);
  const Vector3d p_WT(0.0, 0.0, 0.05);
  const Vector3d Cz_W = (p_WT - p_WC).normalized();
  const Vector3d Cx_W = -Vector3d::UnitZ().cross(Cz_W).normalized();
  const Vector3d Cy_W = Cz_W.cross(Cx_W).normalized();
  return RigidTransformd(
      RotationMatrixd::MakeFromOrthonormalColumns(Cx_W, Cy_W, Cz_W), p_WC);
}

ProximityProperties MakeContactProperties(const SettlingConfig& config) {
  const double resolution = config.representation == Representation::kTet
                                ? config.tet_resolution_hint
                                : config.voxel_width;
  ProximityProperties properties = MakeProperties(
      config.representation, resolution, config.hydroelastic_modulus);
  geometry::AddContactMaterial(config.dissipation, {},
                               CoulombFriction<double>(0.0, 0.0), &properties);
  return properties;
}

struct Sample {
  double time{};
  double penetration{};
  double support_force{};
  double axial_velocity{};
  double lateral_offset{};
  double angular_speed{};
  int faces{};
  double contact_area{};
  double largest_component_area_fraction{};
  int num_components{};
  int sap_iters{};
  int sap_line_search_iters{};
  bool sap_optimality{};
  bool sap_cost{};
  double sap_momentum_residual{};
  bool sap_valid{};
  bool contact{};
};

struct SettledAccumulator {
  int64_t samples{};
  double penetration_sum{};
  double penetration_min{std::numeric_limits<double>::infinity()};
  double penetration_max{-std::numeric_limits<double>::infinity()};
  double support_force_sum{};
  double max_abs_axial_velocity{};
  double max_lateral_offset{};
  double max_angular_speed{};
  double faces_sum{};
  int faces_min{std::numeric_limits<int>::max()};
  int faces_max{std::numeric_limits<int>::lowest()};
  double contact_area_sum{};
  double contact_area_min{std::numeric_limits<double>::infinity()};
  double contact_area_max{-std::numeric_limits<double>::infinity()};
  double largest_component_area_fraction_sum{};
  double num_components_sum{};
  int64_t sap_samples{};
  double sap_iters_sum{};
  double sap_iters_max{};
  double sap_nonconverged{};
  double sap_momentum_residual_max{};

  void Add(const Sample& sample) {
    ++samples;
    penetration_sum += sample.penetration;
    penetration_min = std::min(penetration_min, sample.penetration);
    penetration_max = std::max(penetration_max, sample.penetration);
    support_force_sum += sample.support_force;
    max_abs_axial_velocity =
        std::max(max_abs_axial_velocity, std::abs(sample.axial_velocity));
    max_lateral_offset = std::max(max_lateral_offset, sample.lateral_offset);
    max_angular_speed = std::max(max_angular_speed, sample.angular_speed);
    faces_sum += sample.faces;
    faces_min = std::min(faces_min, sample.faces);
    faces_max = std::max(faces_max, sample.faces);
    contact_area_sum += sample.contact_area;
    contact_area_min = std::min(contact_area_min, sample.contact_area);
    contact_area_max = std::max(contact_area_max, sample.contact_area);
    largest_component_area_fraction_sum +=
        sample.largest_component_area_fraction;
    num_components_sum += sample.num_components;
    if (sample.sap_valid) {
      ++sap_samples;
      sap_iters_sum += sample.sap_iters;
      sap_iters_max = std::max<double>(sap_iters_max, sample.sap_iters);
      // SAP stops on either criterion; failing both means it ran out of
      // iterations at this step.
      if (!sample.sap_optimality && !sample.sap_cost) ++sap_nonconverged;
      sap_momentum_residual_max =
          std::max(sap_momentum_residual_max, sample.sap_momentum_residual);
    }
  }
};

class TrajectoryWriter final {
 public:
  explicit TrajectoryWriter(const std::filesystem::path& path) {
    if (path.empty()) return;
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    output_.open(path);
    if (!output_) {
      throw std::runtime_error("Unable to create trajectory CSV '" +
                               path.string() + "'");
    }
    output_ << std::setprecision(std::numeric_limits<double>::max_digits10);
    output_ << kTrajectoryCsvHeader << '\n';
  }

  void Write(const Sample& sample) {
    if (!output_.is_open()) return;
    output_ << sample.time << ',' << sample.penetration << ','
            << sample.support_force << ',' << sample.axial_velocity << ','
            << sample.lateral_offset << ',' << sample.angular_speed << ','
            << sample.faces << ',' << sample.contact_area << ','
            << sample.largest_component_area_fraction << ','
            << sample.num_components << ',' << sample.sap_iters << ','
            << sample.sap_line_search_iters << ','
            << (sample.sap_optimality ? "true" : "false") << ','
            << (sample.sap_cost ? "true" : "false") << ','
            << sample.sap_momentum_residual << ','
            << (sample.contact ? "true" : "false") << '\n';
    if (!output_) {
      throw std::runtime_error("Failed while writing trajectory CSV");
    }
  }

 private:
  std::ofstream output_;
};

void WriteOptional(std::ostream& output, const std::optional<double>& value) {
  if (value.has_value()) output << *value;
}

GitProvenance ReadGitProvenanceOrUnknown() {
  try {
    return ReadGitProvenance();
  } catch (const std::exception&) {
    // Bazel tests execute from a runfiles tree rather than a Git checkout.
    return {.commit = "unknown", .dirty = true};
  }
}

void WriteSettlingCsv(const std::filesystem::path& path,
                      const SettlingConfig& config,
                      const SettlingResult& result) {
  if (path.empty()) return;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to create summary CSV '" + path.string() +
                             "'");
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  const GitProvenance provenance = ReadGitProvenanceOrUnknown();
  const SettlingDerived& derived = result.derived;
  output << kSettlingCsvHeader << '\n';
  output << "1," << provenance.commit << ','
         << (provenance.dirty ? "true" : "false") << ','
         << to_string(config.scene) << ',' << to_string(config.representation)
         << ',' << config.mass << ',' << config.voxel_width << ','
         << config.tet_resolution_hint << ',' << config.hydroelastic_modulus
         << ',' << config.initial_gap << ',' << config.time_step << ','
         << config.dissipation << ',' << config.duration << ','
         << config.settling_window << ',' << config.trajectory_stride << ','
         << config.grid_rpy_deg.x() << ',' << config.grid_rpy_deg.y() << ','
         << config.grid_rpy_deg.z() << ',' << derived.weight << ','
         << derived.analytic_equilibrium_penetration << ','
         << derived.contact_stiffness << ',' << derived.natural_period << ','
         << derived.duration << ',' << derived.settling_window << ','
         << derived.patch_radius << ',' << derived.elements_across_patch << ','
         << derived.contact_plane_voxel_phase << ','
         << result.equilibrium_penetration << ',' << result.penetration_error
         << ',' << result.penetration_relative_error << ','
         << result.mean_support_force << ','
         << result.mean_support_force_relative_error << ','
         << result.penetration_span << ',' << result.max_abs_axial_velocity
         << ',' << result.max_lateral_offset << ',' << result.max_angular_speed
         << ',' << result.mean_faces << ',' << result.faces_span << ','
         << result.mean_contact_area << ',' << result.contact_area_span << ','
         << result.largest_component_area_fraction << ','
         << result.mean_num_components << ',' << result.mean_sap_iters << ','
         << result.max_sap_iters << ',' << result.sap_nonconverged_steps << ','
         << result.max_sap_momentum_residual << ',' << result.max_penetration
         << ',';
  WriteOptional(output, result.first_contact_time);
  output << ',';
  WriteOptional(output, result.first_contact_loss_time);
  output << ',' << result.steps << ',' << result.wall_time << ','
         << (result.axial_velocity_settled ? "true" : "false") << ','
         << (result.penetration_span_settled ? "true" : "false") << ','
         << (result.settled ? "true" : "false") << '\n';
  if (!output) {
    throw std::runtime_error("Failed while writing summary CSV '" +
                             path.string() + "'");
  }
}

}  // namespace

SettlingScene ParseSettlingScene(std::string_view value) {
  if (value == "sphere_sphere") return SettlingScene::kSphereSphere;
  if (value == "sphere_box") return SettlingScene::kSphereBox;
  throw std::logic_error("Unknown scene '" + std::string(value) +
                         "'; expected sphere_sphere or sphere_box");
}

std::string_view to_string(SettlingScene scene) {
  switch (scene) {
    case SettlingScene::kSphereSphere:
      return "sphere_sphere";
    case SettlingScene::kSphereBox:
      return "sphere_box";
  }
  throw std::logic_error("Invalid SettlingScene value");
}

double DefaultSettlingMass(SettlingScene scene, double hydroelastic_modulus,
                           double target_penetration) {
  ThrowUnlessFinitePositive(hydroelastic_modulus, "hydroelastic_modulus");
  ThrowUnlessFinitePositive(target_penetration, "target_penetration");
  return ForceAtPenetration(MakeReferenceFactory(scene, hydroelastic_modulus),
                            target_penetration) /
         kGravity;
}

SettlingDerived CalcSettlingDerived(const SettlingConfig& config) {
  ValidateConfig(config);
  SettlingDerived result;
  result.weight = config.mass * kGravity;
  const ReferenceFactory factory =
      MakeReferenceFactory(config.scene, config.hydroelastic_modulus);
  result.analytic_equilibrium_penetration =
      EquilibriumPenetration(factory, result.weight, kRadius);
  const double penetration = result.analytic_equilibrium_penetration;
  result.contact_stiffness = StiffnessAtPenetration(factory, penetration);
  result.natural_period =
      2.0 * kPi * std::sqrt(config.mass / result.contact_stiffness);
  result.duration = config.duration == 0.0
                        ? config.duration_periods * result.natural_period
                        : config.duration;
  result.settling_window =
      config.settling_window == 0.0
          ? config.settling_window_periods * result.natural_period
          : config.settling_window;
  if (!(result.settling_window <= result.duration)) {
    throw std::logic_error("settling_window must not exceed duration");
  }
  result.patch_radius =
      std::sqrt(kRadius * penetration - penetration * penetration / 4.0);
  const double resolution = config.representation == Representation::kTet
                                ? config.tet_resolution_hint
                                : config.voxel_width;
  result.elements_across_patch = 2.0 * result.patch_radius / resolution;
  if (config.representation == Representation::kTet) {
    result.contact_plane_voxel_phase = std::numeric_limits<double>::quiet_NaN();
  } else {
    double boundary_coordinate = penetration / (2.0 * config.voxel_width);
    if (config.representation == Representation::kMarchingCubes) {
      boundary_coordinate -= 0.5;
    }
    result.contact_plane_voxel_phase = std::remainder(boundary_coordinate, 1.0);
  }
  return result;
}

std::string_view SettlingCsvHeader() {
  return kSettlingCsvHeader;
}

SettlingResult RunSettling(const SettlingConfig& config) {
  const SettlingDerived derived = CalcSettlingDerived(config);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, config.time_step);
  plant.set_contact_model(ContactModel::kHydroelastic);
  plant.set_discrete_contact_approximation(
      DiscreteContactApproximation::kLagged);
  plant.set_contact_surface_representation(
      SurfaceTypeFor(config.representation));
  plant.SetUseSampledOutputPorts(false);
  plant.mutable_gravity_field().set_gravity_vector(-kGravity *
                                                   Vector3d::UnitZ());

  const SpatialInertia<double> free_inertia =
      SpatialInertia<double>::SolidSphereWithMass(config.mass, kRadius);
  const RigidBody<double>& free_body =
      plant.AddRigidBody("free_sphere", free_inertia);
  const PrismaticJoint<double>* axial_joint = nullptr;
  if (config.axial_only) {
    axial_joint = &plant.AddJoint<PrismaticJoint>(
        "axial_translation", plant.world_body(), std::nullopt, free_body,
        std::nullopt, Vector3d::UnitZ());
  }
  const math::RollPitchYawd grid_rpy(
      kDegreesToRadians * config.grid_rpy_deg.x(),
      kDegreesToRadians * config.grid_rpy_deg.y(),
      kDegreesToRadians * config.grid_rpy_deg.z());
  const RigidTransformd X_BG(grid_rpy, Vector3d::Zero());
  const Sphere sphere(kRadius);
  // Register the moving sphere first. For equal-resolution voxel pairs, its
  // lower GeometryId makes its body-fixed grid the traversed host grid.
  plant.RegisterCollisionGeometry(free_body, X_BG, sphere, "free_sphere",
                                  MakeContactProperties(config));
  plant.RegisterVisualGeometry(free_body, X_BG, sphere, "free_sphere_visual",
                               Eigen::Vector4d(0.20, 0.48, 0.88, 0.65));
  if (config.scene == SettlingScene::kSphereSphere) {
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(),
                                    sphere, "anchored_sphere",
                                    MakeContactProperties(config));
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransformd(), sphere,
                                 "anchored_sphere_visual",
                                 Eigen::Vector4d(0.72, 0.75, 0.80, 0.45));
  } else {
    const Box box(2.0 * kBoxHalfWidths.x(), 2.0 * kBoxHalfWidths.y(),
                  2.0 * kBoxHalfWidths.z());
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(), box,
                                    "anchored_box",
                                    MakeContactProperties(config));
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransformd(), box,
                                 "anchored_box_visual",
                                 Eigen::Vector4d(0.72, 0.75, 0.80, 0.45));
  }
  plant.Finalize();
  if (config.axial_only) {
    if (plant.num_positions() != 1 || plant.num_velocities() != 1) {
      throw std::runtime_error(
          "Axial settling must have exactly one position and velocity");
    }
  } else if (plant.num_positions() != 7 || plant.num_velocities() != 6) {
    throw std::runtime_error("Free settling must retain all six DOF");
  }

  const RgbdSensor* camera{};
  if (!config.frames_dir.empty()) {
    constexpr std::string_view kRendererName = "settling_video_renderer";
    RenderEngineVtkParams renderer_params;
    renderer_params.default_clear_color = Vector3d::Ones();
    scene_graph.AddRenderer(std::string(kRendererName),
                            geometry::MakeRenderEngineVtk(renderer_params));
    const ColorRenderCamera color_camera{{std::string(kRendererName),
                                          {512, 512, 7.0 * kPi / 30.0},
                                          {0.05, 3.0},
                                          {}},
                                         false};
    const DepthRenderCamera depth_camera{color_camera.core(), {0.05, 3.0}};
    camera = builder.AddSystem<RgbdSensor>(scene_graph.world_frame_id(),
                                           MakeCameraPose(), color_camera,
                                           depth_camera);
    builder.Connect(scene_graph.get_query_output_port(),
                    camera->query_object_input_port());
  }

  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  const double initial_height = 2.0 * kRadius + config.initial_gap;
  if (axial_joint != nullptr) {
    axial_joint->set_translation(&plant_context, initial_height);
    axial_joint->set_translation_rate(&plant_context, 0.0);
  } else {
    plant.SetFreeBodyPose(&plant_context, free_body,
                          RigidTransformd(Vector3d(0.0, 0.0, initial_height)));
    plant.SetFreeBodySpatialVelocity(&plant_context, free_body,
                                     SpatialVelocity<double>::Zero());
  }

  TrajectoryWriter trajectory_writer(config.trajectory);
  SettledAccumulator settled;
  std::optional<double> first_contact_time;
  std::optional<double> first_contact_loss_time;
  double max_penetration = 0.0;
  const double settled_start = derived.duration - derived.settling_window;
  if (!config.frames_dir.empty()) {
    std::filesystem::create_directories(config.frames_dir);
    std::filesystem::create_directories(config.frames_dir / "contact_surfaces");
  }

  Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(0.0);
  int64_t monitor_calls = 0;
  int frame_index = 0;
  simulator.set_monitor([&](const Context<double>& root_context) {
    const bool write_trajectory =
        !config.trajectory.empty() &&
        (monitor_calls++ % config.trajectory_stride == 0);
    const Context<double>& current_plant_context =
        plant.GetMyContextFromRoot(root_context);
    const RigidTransformd& X_WB =
        free_body.EvalPoseInWorld(current_plant_context);
    const SpatialVelocity<double>& V_WB =
        free_body.EvalSpatialVelocityInWorld(current_plant_context);
    Sample sample;
    sample.time = root_context.get_time();
    const bool write_frame =
        camera != nullptr && sample.time + 0.5 * config.time_step >=
                                 frame_index * config.frame_period;
    sample.penetration = 2.0 * kRadius - X_WB.translation().z();
    sample.axial_velocity = V_WB.translational().z();
    sample.lateral_offset = X_WB.translation().head<2>().norm();
    sample.angular_speed = V_WB.rotational().norm();
    const bool in_settled_window =
        sample.time + 0.5 * config.time_step >= settled_start;

    // The statistics describe the solve that advances from this state, which
    // is the same state the surface measures below are taken at.
    const std::optional<SapStatistics> sap =
        plant.EvalSapSolverStatistics(current_plant_context);
    if (sap.has_value()) {
      sample.sap_valid = true;
      sample.sap_iters = sap->num_iters;
      sample.sap_line_search_iters = sap->num_line_search_iters;
      sample.sap_optimality = sap->optimality_criterion_reached;
      sample.sap_cost = sap->cost_criterion_reached;
      sample.sap_momentum_residual =
          sap->momentum_residual.empty() ? 0.0 : sap->momentum_residual.back();
    }

    const ContactResults<double>& contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            current_plant_context);
    if (contacts.num_point_pair_contacts() != 0) {
      throw std::runtime_error(
          "Strict hydroelastic settling unexpectedly produced point contact");
    }
    if (contacts.num_hydroelastic_contacts() > 1) {
      throw std::runtime_error("Expected at most one hydroelastic contact");
    }
    if (contacts.num_hydroelastic_contacts() == 1) {
      sample.contact = true;
      const auto& contact = contacts.hydroelastic_contact_info(0);
      sample.support_force = std::abs(contact.F_Ac_W().translational().z());
      const ContactSurface<double>& surface = contact.contact_surface();
      const bool expect_triangle =
          config.representation == Representation::kMarchingCubes;
      if (surface.is_triangle() != expect_triangle) {
        throw std::runtime_error(
            "Contact surface type did not match the representation");
      }
      if (in_settled_window || write_trajectory || write_frame) {
        const SurfaceView view = MakeSurfaceView(surface);
        sample.faces = view.faces.size();
        for (const Face& face : view.faces) sample.contact_area += face.area;
        const ComponentStats components =
            CalcComponentStats(view, sample.contact_area);
        sample.largest_component_area_fraction =
            components.largest_area_fraction;
        sample.num_components = components.num_components;
        if (write_frame) {
          WriteSurfaceVtk(
              config.frames_dir / "contact_surfaces" /
                  fmt::format("frame_{:04d}.vtk", frame_index),
              view,
              fmt::format("{} settling contact t={}",
                          to_string(config.representation), sample.time));
        }
      }
    }

    if (sample.contact && !first_contact_time.has_value()) {
      first_contact_time = sample.time;
    } else if (!sample.contact && first_contact_time.has_value() &&
               !first_contact_loss_time.has_value()) {
      first_contact_loss_time = sample.time;
    }
    max_penetration = std::max(max_penetration, sample.penetration);
    if (in_settled_window) settled.Add(sample);
    if (write_trajectory) trajectory_writer.Write(sample);
    if (write_frame) {
      const Context<double>& camera_context =
          camera->GetMyContextFromRoot(root_context);
      const ImageRgba8U& image =
          camera->color_image_output_port().Eval<ImageRgba8U>(camera_context);
      SaveToPng(image, (config.frames_dir /
                        fmt::format("frame_{:04d}.png", frame_index))
                           .string());
      ++frame_index;
    }
    return EventStatus::Succeeded();
  });

  const auto wall_start = std::chrono::steady_clock::now();
  simulator.Initialize();
  simulator.ResetStatistics();
  simulator.AdvanceTo(derived.duration);
  const double wall_time = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - wall_start)
                               .count();
  if (settled.samples == 0) {
    throw std::runtime_error("The settled window contained no samples");
  }

  SettlingResult result;
  result.derived = derived;
  result.equilibrium_penetration = settled.penetration_sum / settled.samples;
  result.penetration_error =
      result.equilibrium_penetration - derived.analytic_equilibrium_penetration;
  result.penetration_relative_error =
      result.penetration_error / derived.analytic_equilibrium_penetration;
  result.mean_support_force = settled.support_force_sum / settled.samples;
  result.mean_support_force_relative_error =
      (result.mean_support_force - derived.weight) / derived.weight;
  result.penetration_span = settled.penetration_max - settled.penetration_min;
  result.max_abs_axial_velocity = settled.max_abs_axial_velocity;
  result.max_lateral_offset = settled.max_lateral_offset;
  result.max_angular_speed = settled.max_angular_speed;
  result.mean_faces = settled.faces_sum / settled.samples;
  result.faces_span = settled.faces_max - settled.faces_min;
  result.mean_contact_area = settled.contact_area_sum / settled.samples;
  result.contact_area_span =
      settled.contact_area_max - settled.contact_area_min;
  result.largest_component_area_fraction =
      settled.largest_component_area_fraction_sum / settled.samples;
  result.mean_num_components = settled.num_components_sum / settled.samples;
  if (settled.sap_samples > 0) {
    result.mean_sap_iters = settled.sap_iters_sum / settled.sap_samples;
    result.max_sap_iters = settled.sap_iters_max;
    result.sap_nonconverged_steps = settled.sap_nonconverged;
    result.max_sap_momentum_residual = settled.sap_momentum_residual_max;
  }
  result.max_penetration = max_penetration;
  result.first_contact_time = first_contact_time;
  result.first_contact_loss_time = first_contact_loss_time;
  result.steps = simulator.get_num_steps_taken();
  result.wall_time = wall_time;
  result.axial_velocity_settled = result.max_abs_axial_velocity < 1.0e-3;
  result.penetration_span_settled = result.penetration_span < 1.0e-4;
  result.settled =
      result.axial_velocity_settled && result.penetration_span_settled;

  if (!config.mesh_output.empty()) {
    const Context<double>& final_plant_context =
        plant.GetMyContextFromRoot(simulator.get_context());
    const ContactResults<double>& final_contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            final_plant_context);
    if (final_contacts.num_hydroelastic_contacts() != 1) {
      throw std::runtime_error(
          "Cannot write a settled mesh because the run ended without contact");
    }
    const SurfaceView final_surface = MakeSurfaceView(
        final_contacts.hydroelastic_contact_info(0).contact_surface());
    WriteSurfaceVtk(
        config.mesh_output, final_surface,
        fmt::format("{} {} settled h={}", to_string(config.scene),
                    to_string(config.representation), config.voxel_width));
  }
  WriteSettlingCsv(config.output, config, result);
  return result;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
