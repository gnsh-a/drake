#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "drake/common/name_value.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_statistics.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/event_status.h"

namespace drake {
namespace tools {
namespace hydro_compare {
namespace {

using Eigen::Vector3d;
using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::Box;
using geometry::ProximityProperties;
using geometry::Sphere;
using math::RigidTransformd;
using multibody::AddMultibodyPlant;
using multibody::ContactResults;
using multibody::CoulombFriction;
using multibody::MultibodyPlantConfig;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::SpatialVelocity;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::EventStatus;
using systems::Simulator;

using SapStatistics = multibody::contact_solvers::internal::SapStatistics;

constexpr double kDefaultDensity = 1000.0;

struct SphereConfig {
  double radius{};
  double hydroelastic_modulus{};
  Vector3d initial_position{Vector3d::Zero()};
  std::optional<Vector3d> initial_linear_velocity;
  std::optional<Vector3d> initial_angular_velocity;
  std::optional<double> density;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(radius));
    a->Visit(DRAKE_NVP(hydroelastic_modulus));
    a->Visit(DRAKE_NVP(initial_position));
    a->Visit(DRAKE_NVP(initial_linear_velocity));
    a->Visit(DRAKE_NVP(initial_angular_velocity));
    a->Visit(DRAKE_NVP(density));
  }
};

struct BoxConfig {
  Vector3d full_size{Vector3d::Zero()};
  double hydroelastic_modulus{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(full_size));
    a->Visit(DRAKE_NVP(hydroelastic_modulus));
  }
};

struct MaterialConfig {
  double friction{};
  std::optional<double> relaxation_time;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(friction));
    a->Visit(DRAKE_NVP(relaxation_time));
  }
};

struct MeshConfig {
  double sdf_target_voxel_size{};

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(sdf_target_voxel_size));
  }
};

struct SceneConfig {
  SphereConfig sphere;
  BoxConfig box;
  MaterialConfig material;
  MeshConfig mesh;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(sphere));
    a->Visit(DRAKE_NVP(box));
    a->Visit(DRAKE_NVP(material));
    a->Visit(DRAKE_NVP(mesh));
  }
};

struct Args {
  std::string scene{"tools/hydro_compare/sphere_box_hydro.yaml"};
  std::string output{
      "tools/hydro_compare/out/run_hydro_contact/"
      "drake_hydro_contact_reference_cc.csv"};
  int num_frames{600};
  double fps{120.0};
  int substeps{4};
  std::optional<double> time_step;
};

struct Row {
  double time{};
  Vector3d p_WS{Vector3d::Zero()};
  double qx{};
  double qy{};
  double qz{};
  double qw{};
  Vector3d v_WS{Vector3d::Zero()};
  double linear_speed{};
  int point_contacts{};
  int hydro_contacts{};
  double contact_area{};
};

struct SapStatRow {
  double time{};
  int num_hydro_contacts{};
  int num_iters{};
  int num_line_search_iters{};
  bool optimality_reached{};
  bool cost_reached{};
  double final_momentum_residual{};
};

std::string ConsumeValue(int* i, int argc, char* argv[]) {
  if (*i + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + argv[*i]);
  }
  ++(*i);
  return argv[*i];
}

Args ParseArgs(int argc, char* argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--scene") {
      args.scene = ConsumeValue(&i, argc, argv);
    } else if (arg == "--output") {
      args.output = ConsumeValue(&i, argc, argv);
    } else if (arg == "--num-frames") {
      args.num_frames = std::stoi(ConsumeValue(&i, argc, argv));
    } else if (arg == "--fps") {
      args.fps = std::stod(ConsumeValue(&i, argc, argv));
    } else if (arg == "--substeps") {
      args.substeps = std::stoi(ConsumeValue(&i, argc, argv));
    } else if (arg == "--time-step") {
      args.time_step = std::stod(ConsumeValue(&i, argc, argv));
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (args.num_frames < 0) {
    throw std::runtime_error("--num-frames must be non-negative.");
  }
  if (args.fps <= 0.0) {
    throw std::runtime_error("--fps must be positive.");
  }
  if (args.substeps <= 0) {
    throw std::runtime_error("--substeps must be positive.");
  }
  if (args.time_step.has_value() && *args.time_step <= 0.0) {
    throw std::runtime_error("--time-step must be positive.");
  }
  return args;
}

void ValidateScene(const SceneConfig& scene) {
  if (scene.sphere.radius <= 0.0) {
    throw std::runtime_error("sphere.radius must be positive.");
  }
  if (scene.sphere.hydroelastic_modulus <= 0.0) {
    throw std::runtime_error("sphere.hydroelastic_modulus must be positive.");
  }
  if (scene.sphere.density.has_value() && *scene.sphere.density <= 0.0) {
    throw std::runtime_error("sphere.density must be positive.");
  }
  if (scene.box.full_size.minCoeff() <= 0.0) {
    throw std::runtime_error("box.full_size entries must be positive.");
  }
  if (scene.box.hydroelastic_modulus <= 0.0) {
    throw std::runtime_error("box.hydroelastic_modulus must be positive.");
  }
  if (scene.material.friction < 0.0) {
    throw std::runtime_error("material.friction must be non-negative.");
  }
  if (scene.material.relaxation_time.has_value() &&
      *scene.material.relaxation_time < 0.0) {
    throw std::runtime_error("material.relaxation_time must be non-negative.");
  }
  if (scene.mesh.sdf_target_voxel_size <= 0.0) {
    throw std::runtime_error("mesh.sdf_target_voxel_size must be positive.");
  }
}

ProximityProperties MakeProximityProperties(
    double hydroelastic_modulus, double resolution_hint, double friction,
    std::optional<double> relaxation_time) {
  ProximityProperties properties;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &properties);
  AddContactMaterial(std::nullopt, std::nullopt,
                     CoulombFriction<double>(friction, friction), &properties);
  if (relaxation_time.has_value()) {
    properties.AddProperty("material", "relaxation_time", *relaxation_time);
  }
  return properties;
}

Row SampleRow(const multibody::MultibodyPlant<double>& plant,
              const RigidBody<double>& sphere_body,
              const Context<double>& root_context, double time) {
  const Context<double>& plant_context =
      plant.GetMyContextFromRoot(root_context);
  const RigidTransformd X_WS =
      plant.EvalBodyPoseInWorld(plant_context, sphere_body);
  const SpatialVelocity<double> V_WS =
      plant.EvalBodySpatialVelocityInWorld(plant_context, sphere_body);
  const ContactResults<double>& contacts =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          plant_context);

  Row row;
  row.time = time;
  row.p_WS = X_WS.translation();
  const Eigen::Quaterniond q_WS = X_WS.rotation().ToQuaternion();
  row.qx = q_WS.x();
  row.qy = q_WS.y();
  row.qz = q_WS.z();
  row.qw = q_WS.w();
  row.v_WS = V_WS.translational();
  row.linear_speed = row.v_WS.norm();
  row.point_contacts = contacts.num_point_pair_contacts();
  row.hydro_contacts = contacts.num_hydroelastic_contacts();

  for (int i = 0; i < contacts.num_hydroelastic_contacts(); ++i) {
    const auto& info = contacts.hydroelastic_contact_info(i);
    row.contact_area += info.contact_surface().total_area();
  }
  return row;
}

void WriteHeader(std::ofstream* out) {
  *out << "time,x,y,z,qx,qy,qz,qw,vx,vy,vz,linear_speed,point_contacts,"
          "hydro_contacts,contact_area\n";
}

void WriteRow(std::ofstream* out, const Row& row) {
  *out << row.time << "," << row.p_WS.x() << "," << row.p_WS.y() << ","
       << row.p_WS.z() << "," << row.qx << "," << row.qy << "," << row.qz
       << "," << row.qw << "," << row.v_WS.x() << "," << row.v_WS.y() << ","
       << row.v_WS.z() << "," << row.linear_speed << ","
       << row.point_contacts << "," << row.hydro_contacts << ","
       << row.contact_area << "\n";
}

std::filesystem::path MakeSapStatsPath(
    const std::filesystem::path& output_path) {
  std::filesystem::path sap_stats_path = output_path;
  sap_stats_path.replace_filename(output_path.stem().string() +
                                  "_sap_stats.csv");
  return sap_stats_path;
}

void WriteSapStats(const std::vector<SapStatRow>& rows,
                   const std::filesystem::path& output_path) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  if (!out.good()) {
    throw std::runtime_error("Could not write " + output_path.string());
  }
  out.precision(17);
  out << "time,num_hydro_contacts,num_iters,num_line_search_iters,"
         "optimality_reached,cost_reached,final_momentum_residual\n";
  for (const SapStatRow& row : rows) {
    out << row.time << "," << row.num_hydro_contacts << "," << row.num_iters
        << "," << row.num_line_search_iters << ","
        << (row.optimality_reached ? 1 : 0) << ","
        << (row.cost_reached ? 1 : 0) << ","
        << row.final_momentum_residual << "\n";
  }
}

int DoMain(int argc, char* argv[]) {
  const Args args = ParseArgs(argc, argv);
  const SceneConfig scene = yaml::LoadYamlFile<SceneConfig>(args.scene);
  ValidateScene(scene);

  const double frame_dt = 1.0 / args.fps;
  const double sim_dt =
      args.time_step.value_or(frame_dt / static_cast<double>(args.substeps));
  const double density = scene.sphere.density.value_or(kDefaultDensity);
  const double radius = scene.sphere.radius;
  const double mass = density * 4.0 / 3.0 * M_PI * radius * radius * radius;

  DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = sim_dt;
  plant_config.contact_model = "hydroelastic";
  plant_config.discrete_contact_approximation = "sap";
  plant_config.contact_surface_representation = "polygon";
  auto plant_scene_graph = AddMultibodyPlant(plant_config, &builder);
  auto& plant = plant_scene_graph.plant;

  const RigidBody<double>& sphere_body = plant.AddRigidBody(
      "sphere", SpatialInertia<double>::SolidSphereWithMass(mass, radius));
  ProximityProperties sphere_props = MakeProximityProperties(
      scene.sphere.hydroelastic_modulus, scene.mesh.sdf_target_voxel_size,
      scene.material.friction, scene.material.relaxation_time);
  plant.RegisterCollisionGeometry(sphere_body, RigidTransformd::Identity(),
                                  Sphere(radius), "sphere_collision",
                                  std::move(sphere_props));

  ProximityProperties box_props = MakeProximityProperties(
      scene.box.hydroelastic_modulus, scene.mesh.sdf_target_voxel_size,
      scene.material.friction, scene.material.relaxation_time);
  plant.RegisterCollisionGeometry(
      plant.world_body(), RigidTransformd::Identity(),
      Box(scene.box.full_size.x(), scene.box.full_size.y(),
          scene.box.full_size.z()),
      "box_collision", std::move(box_props));

  plant.Finalize();
  std::unique_ptr<Diagram<double>> diagram = builder.Build();
  std::unique_ptr<Context<double>> context = diagram->CreateDefaultContext();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());
  plant.SetFreeBodyPose(&plant_context, sphere_body,
                        RigidTransformd(scene.sphere.initial_position));
  plant.SetFreeBodySpatialVelocity(
      &plant_context, sphere_body,
      SpatialVelocity<double>(
          scene.sphere.initial_angular_velocity.value_or(Vector3d::Zero()),
          scene.sphere.initial_linear_velocity.value_or(Vector3d::Zero())));

  const std::filesystem::path output_path(args.output);
  const std::filesystem::path sap_stats_path = MakeSapStatsPath(output_path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  if (!out.good()) {
    throw std::runtime_error("Could not write " + output_path.string());
  }
  out.precision(17);
  WriteHeader(&out);

  Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_target_realtime_rate(0.0);

  std::vector<SapStatRow> sap_stats;
  simulator.set_monitor([&plant,
                         &sap_stats](const Context<double>& root_context) {
    const Context<double>& plant_ctx = plant.GetMyContextFromRoot(root_context);
    const std::optional<SapStatistics> stats =
        plant.EvalSapSolverStatistics(plant_ctx);
    if (!stats.has_value()) {
      return EventStatus::Succeeded();
    }
    const ContactResults<double>& contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            plant_ctx);
    SapStatRow row;
    row.time = root_context.get_time();
    row.num_hydro_contacts = contacts.num_hydroelastic_contacts();
    row.num_iters = stats->num_iters;
    row.num_line_search_iters = stats->num_line_search_iters;
    row.optimality_reached = stats->optimality_criterion_reached;
    row.cost_reached = stats->cost_criterion_reached;
    row.final_momentum_residual = stats->momentum_residual.empty()
                                      ? 0.0
                                      : stats->momentum_residual.back();
    sap_stats.push_back(row);
    return EventStatus::Succeeded();
  });

  simulator.Initialize();

  Row final = SampleRow(plant, sphere_body, simulator.get_context(), 0.0);
  WriteRow(&out, final);
  for (int frame = 1; frame <= args.num_frames; ++frame) {
    const double time = frame * frame_dt;
    simulator.AdvanceTo(time);
    final = SampleRow(plant, sphere_body, simulator.get_context(), time);
    WriteRow(&out, final);
  }

  WriteSapStats(sap_stats, sap_stats_path);
  return 0;
}

}  // namespace
}  // namespace hydro_compare
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::tools::hydro_compare::DoMain(argc, argv);
}
