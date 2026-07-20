#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/name_value.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_statistics.h"
#include "drake/multibody/contact_solvers/tamsi_solver_statistics.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/event_status.h"

namespace drake {
namespace tools {
namespace hydro_compare {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;
using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::Box;
using geometry::Cylinder;
using geometry::ProximityProperties;
using math::RigidTransformd;
using multibody::AddMultibodyPlant;
using multibody::ContactResults;
using multibody::CoulombFriction;
using multibody::MultibodyPlantConfig;
using multibody::PlanarJoint;
using multibody::RigidBody;
using multibody::SpatialForce;
using multibody::SpatialInertia;
using multibody::SpatialVelocity;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::EventStatus;
using systems::Simulator;

using SapStatistics = multibody::contact_solvers::internal::SapStatistics;
using TamsiStatistics = multibody::contact_solvers::internal::TamsiStatistics;

struct DiskConfig {
  double radius{};
  double thickness{};
  double hydroelastic_modulus{};
  Vector3d initial_position{Vector3d::Zero()};
  std::optional<Vector3d> initial_linear_velocity;
  std::optional<Vector3d> initial_angular_velocity;
  std::optional<double> density;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(radius));
    a->Visit(DRAKE_NVP(thickness));
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
  DiskConfig disk;
  BoxConfig box;
  MaterialConfig material;
  MeshConfig mesh;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(disk));
    a->Visit(DRAKE_NVP(box));
    a->Visit(DRAKE_NVP(material));
    a->Visit(DRAKE_NVP(mesh));
  }
};

struct Args {
  std::string scene{"tools/hydro_compare/disk_plane.yaml"};
  std::string output{
      "tools/hydro_compare/out/run_disk_contact/"
      "drake_disk_contact_reference_cc.csv"};
  int num_frames{600};
  double fps{200.0};
  int substeps{4};
  std::optional<double> time_step;
  std::string contact_approximation{"tamsi"};
  bool quiet{false};
};

struct Row {
  double time{};
  Vector3d p_WD{Vector3d::Zero()};
  double qx{};
  double qy{};
  double qz{};
  double qw{};
  Vector3d w_WD{Vector3d::Zero()};
  Vector3d v_WD{Vector3d::Zero()};
  double angular_speed{};
  double linear_speed{};
  int point_contacts{};
  int hydro_contacts{};
  Vector3d contact_force{Vector3d::Zero()};
  double contact_area{};
  double normal_force_z{};
  double friction_force_x{};
  double friction_force_y{};
  double friction_torque_z{};
  double eps{};
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

struct TamsiStatRow {
  double time{};
  int num_hydro_contacts{};
  int accepted_num_substeps{};
  int num_substep_attempts{};
  int num_solve_calls{};
  int total_iterations{};
  int max_iterations_per_solve{};
  double final_vt_residual{};
  multibody::TamsiSolverResult result{
      multibody::TamsiSolverResult::kMaxIterationsReached};
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
    } else if (arg == "--contact-approximation") {
      args.contact_approximation = ConsumeValue(&i, argc, argv);
    } else if (arg == "--quiet") {
      args.quiet = true;
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
  if (args.contact_approximation != "tamsi" &&
      args.contact_approximation != "lagged") {
    throw std::runtime_error(
        "--contact-approximation must be one of tamsi or lagged.");
  }
  return args;
}

void ValidateScene(const SceneConfig& scene) {
  if (scene.disk.radius <= 0.0) {
    throw std::runtime_error("disk.radius must be positive.");
  }
  if (scene.disk.thickness <= 0.0) {
    throw std::runtime_error("disk.thickness must be positive.");
  }
  if (scene.disk.hydroelastic_modulus <= 0.0) {
    throw std::runtime_error("disk.hydroelastic_modulus must be positive.");
  }
  if (scene.disk.density.has_value() && *scene.disk.density <= 0.0) {
    throw std::runtime_error("disk.density must be positive.");
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

bool ContainsGeometryId(const std::vector<geometry::GeometryId>& ids,
                        geometry::GeometryId id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

Row SampleRow(const multibody::MultibodyPlant<double>& plant,
              const RigidBody<double>& disk_body,
              const std::vector<geometry::GeometryId>& disk_geometries,
              double radius, const Context<double>& root_context,
              double time) {
  const Context<double>& plant_context =
      plant.GetMyContextFromRoot(root_context);
  const RigidTransformd X_WD =
      plant.EvalBodyPoseInWorld(plant_context, disk_body);
  const SpatialVelocity<double> V_WD =
      plant.EvalBodySpatialVelocityInWorld(plant_context, disk_body);
  const ContactResults<double>& contacts =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          plant_context);

  Row row;
  row.time = time;
  row.p_WD = X_WD.translation();
  const Eigen::Quaterniond q_WD = X_WD.rotation().ToQuaternion();
  row.qx = q_WD.x();
  row.qy = q_WD.y();
  row.qz = q_WD.z();
  row.qw = q_WD.w();
  row.w_WD = V_WD.rotational();
  row.v_WD = V_WD.translational();
  row.angular_speed = row.w_WD.norm();
  row.linear_speed = row.v_WD.norm();
  row.point_contacts = contacts.num_point_pair_contacts();
  row.hydro_contacts = contacts.num_hydroelastic_contacts();

  Vector3d net_force_W = Vector3d::Zero();
  Vector3d net_torque_WD = Vector3d::Zero();
  for (int i = 0; i < contacts.num_hydroelastic_contacts(); ++i) {
    const auto& info = contacts.hydroelastic_contact_info(i);
    const auto& surface = info.contact_surface();
    row.contact_area += surface.total_area();

    SpatialForce<double> F_C_W = info.F_Ac_W();
    const bool disk_is_A = ContainsGeometryId(disk_geometries, surface.id_M());
    if (!disk_is_A) {
      F_C_W = -F_C_W;
    }
    const SpatialForce<double> F_D_W =
        F_C_W.Shift(row.p_WD - surface.centroid());
    net_force_W += F_D_W.translational();
    net_torque_WD += F_D_W.rotational();
  }

  row.contact_force = net_force_W;
  row.normal_force_z = net_force_W.z();
  row.friction_force_x = net_force_W.x();
  row.friction_force_y = net_force_W.y();
  row.friction_torque_z = net_torque_WD.z();

  const double wz = std::abs(row.w_WD.z());
  row.eps = (wz > 1e-12) ? row.linear_speed / (wz * radius)
                         : std::numeric_limits<double>::quiet_NaN();
  return row;
}

void WriteHeader(std::ofstream* out) {
  *out << "time,x,y,z,qx,qy,qz,qw,wx,wy,wz,vx,vy,vz,angular_speed,"
          "linear_speed,point_contacts,hydro_contacts,contact_force_x,"
          "contact_force_y,contact_force_z,contact_area,normal_force_z,"
          "friction_force_x,friction_force_y,friction_torque_z,eps\n";
}

void WriteRow(std::ofstream* out, const Row& row) {
  *out << row.time << "," << row.p_WD.x() << "," << row.p_WD.y() << ","
       << row.p_WD.z() << "," << row.qx << "," << row.qy << "," << row.qz
       << "," << row.qw << "," << row.w_WD.x() << "," << row.w_WD.y() << ","
       << row.w_WD.z() << "," << row.v_WD.x() << "," << row.v_WD.y() << ","
       << row.v_WD.z() << "," << row.angular_speed << ","
       << row.linear_speed << "," << row.point_contacts << ","
       << row.hydro_contacts << "," << row.contact_force.x() << ","
       << row.contact_force.y() << "," << row.contact_force.z() << ","
       << row.contact_area << "," << row.normal_force_z << ","
       << row.friction_force_x << "," << row.friction_force_y << ","
       << row.friction_torque_z << "," << row.eps << "\n";
}

std::filesystem::path MakeSapStatsPath(
    const std::filesystem::path& output_path) {
  std::filesystem::path sap_stats_path = output_path;
  sap_stats_path.replace_filename(output_path.stem().string() +
                                  "_sap_stats.csv");
  return sap_stats_path;
}

std::filesystem::path MakeTamsiStatsPath(
    const std::filesystem::path& output_path) {
  std::filesystem::path tamsi_stats_path = output_path;
  tamsi_stats_path.replace_filename(output_path.stem().string() +
                                    "_tamsi_stats.csv");
  return tamsi_stats_path;
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

void WriteTamsiStats(const std::vector<TamsiStatRow>& rows,
                     const std::filesystem::path& output_path) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  if (!out.good()) {
    throw std::runtime_error("Could not write " + output_path.string());
  }
  out.precision(17);
  out << "time,num_hydro_contacts,accepted_num_substeps,"
         "num_substep_attempts,num_solve_calls,total_iterations,"
         "max_iterations_per_solve,final_vt_residual,result\n";
  for (const TamsiStatRow& row : rows) {
    out << row.time << "," << row.num_hydro_contacts << ","
        << row.accepted_num_substeps << "," << row.num_substep_attempts << ","
        << row.num_solve_calls << "," << row.total_iterations << ","
        << row.max_iterations_per_solve << "," << row.final_vt_residual << ","
        << static_cast<int>(row.result) << "\n";
  }
}

void CheckNoPointContact(const Row& row) {
  if (row.point_contacts > 0) {
    throw std::runtime_error(
        "Disk scene produced point contacts; expected strict hydroelastic "
        "contact.");
  }
}

int DoMain(int argc, char* argv[]) {
  const Args args = ParseArgs(argc, argv);
  const SceneConfig scene = yaml::LoadYamlFile<SceneConfig>(args.scene);
  ValidateScene(scene);

  const double frame_dt = 1.0 / args.fps;
  const double sim_dt =
      args.time_step.value_or(frame_dt / static_cast<double>(args.substeps));
  const double density = scene.disk.density.value_or(1000.0);
  const double radius = scene.disk.radius;
  const double mass =
      density * M_PI * radius * radius * scene.disk.thickness;

  DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = sim_dt;
  plant_config.contact_model = "hydroelastic";
  plant_config.discrete_contact_approximation = args.contact_approximation;
  plant_config.contact_surface_representation = "polygon";
  auto plant_scene_graph = AddMultibodyPlant(plant_config, &builder);
  auto& plant = plant_scene_graph.plant;

  const RigidBody<double>& disk_body = plant.AddRigidBody(
      "disk", SpatialInertia<double>::SolidCylinderWithMass(
                  mass, radius, scene.disk.thickness, Vector3d::UnitZ()));
  ProximityProperties disk_props = MakeProximityProperties(
      scene.disk.hydroelastic_modulus, scene.mesh.sdf_target_voxel_size,
      scene.material.friction, scene.material.relaxation_time);
  plant.RegisterCollisionGeometry(disk_body, RigidTransformd::Identity(),
                                  Cylinder(radius, scene.disk.thickness),
                                  "disk_collision", std::move(disk_props));

  const RigidTransformd X_WF(
      Vector3d(0.0, 0.0, scene.disk.initial_position.z()));
  const PlanarJoint<double>& disk_planar_joint = plant.AddJoint<PlanarJoint>(
      "disk_planar", plant.world_body(), X_WF, disk_body,
      RigidTransformd::Identity(), Vector3d::Zero());

  ProximityProperties box_props = MakeProximityProperties(
      scene.box.hydroelastic_modulus, scene.mesh.sdf_target_voxel_size,
      scene.material.friction, scene.material.relaxation_time);
  plant.RegisterCollisionGeometry(
      plant.world_body(), RigidTransformd::Identity(),
      Box(scene.box.full_size.x(), scene.box.full_size.y(),
          scene.box.full_size.z()),
      "box_collision", std::move(box_props));

  plant.Finalize();
  const std::vector<geometry::GeometryId> disk_geometries =
      plant.GetCollisionGeometriesForBody(disk_body);

  std::unique_ptr<Diagram<double>> diagram = builder.Build();
  std::unique_ptr<Context<double>> context = diagram->CreateDefaultContext();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());
  const Vector3d p0 = scene.disk.initial_position;
  const Vector3d v0 =
      scene.disk.initial_linear_velocity.value_or(Vector3d::Zero());
  const Vector3d w0 =
      scene.disk.initial_angular_velocity.value_or(Vector3d::Zero());
  disk_planar_joint.set_translation(&plant_context, Vector2d(p0.x(), p0.y()));
  disk_planar_joint.set_rotation(&plant_context, 0.0);
  disk_planar_joint.set_translational_velocity(&plant_context,
                                               Vector2d(v0.x(), v0.y()));
  disk_planar_joint.set_angular_velocity(&plant_context, w0.z());

  const std::filesystem::path output_path(args.output);
  const std::filesystem::path sap_stats_path = MakeSapStatsPath(output_path);
  const std::filesystem::path tamsi_stats_path =
      MakeTamsiStatsPath(output_path);
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
  std::vector<TamsiStatRow> tamsi_stats;
  simulator.set_monitor([&plant, &sap_stats,
                         &tamsi_stats](const Context<double>& root_context) {
    const Context<double>& plant_ctx = plant.GetMyContextFromRoot(root_context);
    const std::optional<SapStatistics> sap =
        plant.EvalSapSolverStatistics(plant_ctx);
    const std::optional<TamsiStatistics> tamsi =
        plant.EvalTamsiSolverStatistics(plant_ctx);
    if (!sap.has_value() && !tamsi.has_value()) {
      return EventStatus::Succeeded();
    }

    const ContactResults<double>& contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            plant_ctx);
    const int num_hydro_contacts = contacts.num_hydroelastic_contacts();
    const double time = root_context.get_time();

    if (sap.has_value()) {
      SapStatRow row;
      row.time = time;
      row.num_hydro_contacts = num_hydro_contacts;
      row.num_iters = sap->num_iters;
      row.num_line_search_iters = sap->num_line_search_iters;
      row.optimality_reached = sap->optimality_criterion_reached;
      row.cost_reached = sap->cost_criterion_reached;
      row.final_momentum_residual = sap->momentum_residual.empty()
                                        ? 0.0
                                        : sap->momentum_residual.back();
      sap_stats.push_back(row);
    }

    if (tamsi.has_value()) {
      TamsiStatRow row;
      row.time = time;
      row.num_hydro_contacts = num_hydro_contacts;
      row.accepted_num_substeps = tamsi->accepted_num_substeps;
      row.num_substep_attempts = tamsi->num_substep_attempts;
      row.num_solve_calls = tamsi->num_solve_calls;
      row.total_iterations = tamsi->total_iterations;
      row.max_iterations_per_solve = tamsi->max_iterations_per_solve;
      row.final_vt_residual = tamsi->final_vt_residual;
      row.result = tamsi->result;
      tamsi_stats.push_back(row);
    }
    return EventStatus::Succeeded();
  });

  simulator.Initialize();

  Row final = SampleRow(plant, disk_body, disk_geometries, radius,
                        simulator.get_context(), 0.0);
  CheckNoPointContact(final);
  WriteRow(&out, final);
  for (int frame = 1; frame <= args.num_frames; ++frame) {
    const double time = frame * frame_dt;
    simulator.AdvanceTo(time);
    final = SampleRow(plant, disk_body, disk_geometries, radius,
                      simulator.get_context(), time);
    CheckNoPointContact(final);
    WriteRow(&out, final);
  }

  WriteSapStats(sap_stats, sap_stats_path);
  WriteTamsiStats(tamsi_stats, tamsi_stats_path);
  if (!args.quiet) {
    std::cout << "wrote " << output_path << "\n";
    std::cout << "wrote " << sap_stats_path << "\n";
    std::cout << "wrote " << tamsi_stats_path << "\n";
  }
  return 0;
}

}  // namespace
}  // namespace hydro_compare
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::tools::hydro_compare::DoMain(argc, argv);
}
