#include "drake/tools/voxel_sdf_experiments/disk_farkas/disk_farkas.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/tools/voxel_sdf_experiments/common/components.h"
#include "drake/tools/voxel_sdf_experiments/common/emit.h"
#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Cylinder;
using geometry::ProximityProperties;
using math::RigidTransformd;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::ContactModel;
using multibody::ContactResults;
using multibody::CoulombFriction;
using multibody::DiscreteContactApproximation;
using multibody::MultibodyPlant;
using multibody::RigidBody;
using multibody::SpatialForce;
using multibody::SpatialInertia;
using multibody::SpatialVelocity;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kGravity = 9.81;
constexpr double kSpinThreshold = 0.1;
constexpr std::string_view kCsvHeader =
    "schema_version,git_commit,git_dirty,representation,resolution_m,"
    "time_step_s,settle_time_step_s,time_s,x_m,y_m,z_m,qx,qy,qz,qw,"
    "wx_rad_s,wy_rad_s,wz_rad_s,vx_m_s,vy_m_s,vz_m_s,angular_speed_rad_s,"
    "linear_speed_m_s,point_contacts,hydro_contacts,contact_force_x_N,"
    "contact_force_y_N,contact_force_z_N,contact_area_m2,surface_vertices,"
    "surface_faces,num_components,largest_component_area_fraction,"
    "normal_force_z_N,friction_force_x_N,friction_force_y_N,"
    "friction_torque_z_Nm,eps,post_kick";

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

void ValidateScene(const DiskSceneConfig& scene) {
  ThrowUnlessFinitePositive(scene.disk.radius, "disk.radius");
  ThrowUnlessFinitePositive(scene.disk.thickness, "disk.thickness");
  ThrowUnlessFinitePositive(scene.disk.hydroelastic_modulus,
                            "disk.hydroelastic_modulus");
  if (scene.disk.density.has_value()) {
    ThrowUnlessFinitePositive(*scene.disk.density, "disk.density");
  }
  if (!scene.disk.initial_position.allFinite()) {
    throw std::logic_error("disk.initial_position must be finite");
  }
  if (scene.disk.initial_linear_velocity.has_value() &&
      !scene.disk.initial_linear_velocity->allFinite()) {
    throw std::logic_error("disk.initial_linear_velocity must be finite");
  }
  if (scene.disk.initial_angular_velocity.has_value() &&
      !scene.disk.initial_angular_velocity->allFinite()) {
    throw std::logic_error("disk.initial_angular_velocity must be finite");
  }
  for (int axis = 0; axis < 3; ++axis) {
    ThrowUnlessFinitePositive(scene.box.full_size[axis], "box.full_size");
  }
  ThrowUnlessFinitePositive(scene.box.hydroelastic_modulus,
                            "box.hydroelastic_modulus");
  ThrowUnlessFiniteNonnegative(scene.material.friction, "material.friction");
  if (scene.material.relaxation_time.has_value()) {
    ThrowUnlessFiniteNonnegative(*scene.material.relaxation_time,
                                 "material.relaxation_time");
  }
  ThrowUnlessFinitePositive(scene.mesh.sdf_target_voxel_size,
                            "mesh.sdf_target_voxel_size");
}

void ValidateConfig(const DiskFarkasConfig& config) {
  ValidateScene(config.scene);
  ThrowUnlessFinitePositive(config.resolution, "resolution");
  ThrowUnlessFinitePositive(config.time_step, "time_step");
  ThrowUnlessFiniteNonnegative(config.settle_time, "settle_time");
  ThrowUnlessFinitePositive(config.settle_time_step, "settle_time_step");
  ThrowUnlessFinitePositive(config.frames_per_second, "frames_per_second");
  if (config.num_frames <= 0) {
    throw std::logic_error("num_frames must be positive");
  }
}

ProximityProperties MakeContactProperties(const DiskFarkasConfig& config,
                                          double modulus) {
  ProximityProperties properties =
      MakeProperties(config.representation, config.resolution, modulus);
  geometry::AddContactMaterial(
      std::nullopt, std::nullopt,
      CoulombFriction<double>(config.scene.material.friction,
                              config.scene.material.friction),
      &properties);
  if (config.scene.material.relaxation_time.has_value()) {
    properties.AddProperty("material", "relaxation_time",
                           *config.scene.material.relaxation_time);
  }
  return properties;
}

bool ContainsGeometryId(const std::vector<geometry::GeometryId>& ids,
                        geometry::GeometryId id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

struct BuiltScene {
  std::unique_ptr<Diagram<double>> diagram;
  const MultibodyPlant<double>* plant{};
  const RigidBody<double>* disk_body{};
  std::vector<geometry::GeometryId> disk_geometries;
  bool expect_triangle_surface{};
};

BuiltScene BuildScene(const DiskFarkasConfig& config, double time_step,
                      double mass) {
  DiagramBuilder<double> builder;
  MultibodyPlant<double>& plant =
      AddMultibodyPlantSceneGraph(&builder, time_step).plant;
  plant.set_contact_model(ContactModel::kHydroelastic);
  plant.set_discrete_contact_approximation(
      DiscreteContactApproximation::kLagged);
  plant.set_contact_surface_representation(
      SurfaceTypeFor(config.representation));
  plant.SetUseSampledOutputPorts(false);
  plant.mutable_gravity_field().set_gravity_vector(-kGravity *
                                                   Vector3d::UnitZ());

  const DiskSceneConfig& scene = config.scene;
  const RigidBody<double>& disk_body = plant.AddRigidBody(
      "disk",
      SpatialInertia<double>::SolidCylinderWithMass(
          mass, scene.disk.radius, scene.disk.thickness, Vector3d::UnitZ()));
  plant.RegisterCollisionGeometry(
      disk_body, RigidTransformd(),
      Cylinder(scene.disk.radius, scene.disk.thickness), "disk_collision",
      MakeContactProperties(config, scene.disk.hydroelastic_modulus));
  plant.RegisterCollisionGeometry(
      plant.world_body(), RigidTransformd(),
      Box(scene.box.full_size.x(), scene.box.full_size.y(),
          scene.box.full_size.z()),
      "box_collision",
      MakeContactProperties(config, scene.box.hydroelastic_modulus));
  plant.Finalize();
  if (plant.num_positions() != 7 || plant.num_velocities() != 6) {
    throw std::runtime_error("The disk must retain all six free-body DOFs");
  }

  BuiltScene result;
  result.disk_geometries = plant.GetCollisionGeometriesForBody(disk_body);
  result.diagram = builder.Build();
  result.plant = &plant;
  result.disk_body = &disk_body;
  result.expect_triangle_surface =
      config.representation == Representation::kMarchingCubes;
  return result;
}

DiskFarkasRow SampleRow(const BuiltScene& scene,
                        const Context<double>& root_context, double radius) {
  const Context<double>& plant_context =
      scene.plant->GetMyContextFromRoot(root_context);
  const RigidTransformd X_WD = scene.disk_body->EvalPoseInWorld(plant_context);
  const SpatialVelocity<double> V_WD =
      scene.disk_body->EvalSpatialVelocityInWorld(plant_context);
  const ContactResults<double>& contacts =
      scene.plant->get_contact_results_output_port()
          .Eval<ContactResults<double>>(plant_context);

  DiskFarkasRow row;
  row.time = root_context.get_time();
  row.position_WD = X_WD.translation();
  const Eigen::Quaterniond q_WD = X_WD.rotation().ToQuaternion();
  row.qx = q_WD.x();
  row.qy = q_WD.y();
  row.qz = q_WD.z();
  row.qw = q_WD.w();
  row.angular_velocity_WD = V_WD.rotational();
  row.linear_velocity_WD = V_WD.translational();
  row.angular_speed = row.angular_velocity_WD.norm();
  row.linear_speed = row.linear_velocity_WD.norm();
  row.point_contacts = contacts.num_point_pair_contacts();
  row.hydro_contacts = contacts.num_hydroelastic_contacts();

  Vector3d net_force_W = Vector3d::Zero();
  Vector3d net_torque_WD = Vector3d::Zero();
  for (int i = 0; i < contacts.num_hydroelastic_contacts(); ++i) {
    const auto& info = contacts.hydroelastic_contact_info(i);
    const auto& surface = info.contact_surface();
    if (surface.is_triangle() != scene.expect_triangle_surface) {
      throw std::runtime_error(
          "Contact surface type did not match the representation");
    }
    row.contact_area += surface.total_area();
    if (surface.is_triangle()) {
      row.surface_vertices += surface.tri_mesh_W().num_vertices();
      row.surface_faces += surface.tri_mesh_W().num_triangles();
    } else {
      row.surface_vertices += surface.poly_mesh_W().num_vertices();
      row.surface_faces += surface.poly_mesh_W().num_faces();
    }

    // Fragmentation is per surface, and the disk scene produces one, so the
    // count is summed and the fraction is taken from the largest patch seen.
    const SurfaceView view = MakeSurfaceView(surface);
    double view_area = 0.0;
    for (const Face& face : view.faces) view_area += face.area;
    const ComponentStats components = CalcComponentStats(view, view_area);
    row.num_components += components.num_components;
    row.largest_component_area_fraction = std::max(
        row.largest_component_area_fraction, components.largest_area_fraction);

    SpatialForce<double> F_C_W = info.F_Ac_W();
    const bool disk_is_body_a =
        ContainsGeometryId(scene.disk_geometries, surface.id_M());
    if (!disk_is_body_a) F_C_W = -F_C_W;
    const SpatialForce<double> F_D_W =
        F_C_W.Shift(row.position_WD - surface.centroid());
    net_force_W += F_D_W.translational();
    net_torque_WD += F_D_W.rotational();
  }
  row.contact_force_W = net_force_W;
  row.normal_force_z = net_force_W.z();
  row.friction_force_x = net_force_W.x();
  row.friction_force_y = net_force_W.y();
  row.friction_torque_z = net_torque_WD.z();

  const double wz = std::abs(row.angular_velocity_WD.z());
  row.eps = wz > 1e-12 ? row.linear_speed / (wz * radius)
                       : std::numeric_limits<double>::quiet_NaN();
  if (row.point_contacts != 0) {
    throw std::runtime_error(
        "Strict hydroelastic disk scene unexpectedly produced point contact");
  }
  if (row.hydro_contacts > 1) {
    throw std::runtime_error("Expected at most one hydroelastic contact");
  }
  return row;
}

GitProvenance ReadGitProvenanceOrUnknown() {
  try {
    return ReadGitProvenance();
  } catch (const std::exception&) {
    return {.commit = "unknown", .dirty = true};
  }
}

void WriteCsv(const DiskFarkasConfig& config,
              const std::vector<DiskFarkasRow>& rows) {
  if (config.output.empty()) return;
  if (config.output.has_parent_path()) {
    std::filesystem::create_directories(config.output.parent_path());
  }
  std::ofstream output(config.output);
  if (!output) {
    throw std::runtime_error("Unable to create trajectory CSV '" +
                             config.output.string() + "'");
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  const GitProvenance provenance = ReadGitProvenanceOrUnknown();
  output << kCsvHeader << '\n';
  for (const DiskFarkasRow& row : rows) {
    output << "1," << provenance.commit << ','
           << (provenance.dirty ? "true" : "false") << ','
           << to_string(config.representation) << ',' << config.resolution
           << ',' << config.time_step << ',' << config.settle_time_step << ','
           << row.time << ',' << row.position_WD.x() << ','
           << row.position_WD.y() << ',' << row.position_WD.z() << ',' << row.qx
           << ',' << row.qy << ',' << row.qz << ',' << row.qw << ','
           << row.angular_velocity_WD.x() << ',' << row.angular_velocity_WD.y()
           << ',' << row.angular_velocity_WD.z() << ','
           << row.linear_velocity_WD.x() << ',' << row.linear_velocity_WD.y()
           << ',' << row.linear_velocity_WD.z() << ',' << row.angular_speed
           << ',' << row.linear_speed << ',' << row.point_contacts << ','
           << row.hydro_contacts << ',' << row.contact_force_W.x() << ','
           << row.contact_force_W.y() << ',' << row.contact_force_W.z() << ','
           << row.contact_area << ',' << row.surface_vertices << ','
           << row.surface_faces << ',' << row.num_components << ','
           << row.largest_component_area_fraction << ',' << row.normal_force_z
           << ',' << row.friction_force_x << ',' << row.friction_force_y << ','
           << row.friction_torque_z << ',' << row.eps << ','
           << (row.post_kick ? "true" : "false") << '\n';
  }
  if (!output) {
    throw std::runtime_error("Failed while writing trajectory CSV '" +
                             config.output.string() + "'");
  }
}

bool DiscreteStatesAreExactlyEqual(const Context<double>& a,
                                   const Context<double>& b) {
  if (a.num_discrete_state_groups() != b.num_discrete_state_groups()) {
    return false;
  }
  for (int group = 0; group < a.num_discrete_state_groups(); ++group) {
    if (a.get_discrete_state(group).value() !=
        b.get_discrete_state(group).value()) {
      return false;
    }
  }
  return true;
}

}  // namespace

DiskSceneConfig LoadDiskScene(const std::filesystem::path& path) {
  DiskSceneConfig scene = yaml::LoadYamlFile<DiskSceneConfig>(path.string());
  ValidateScene(scene);
  return scene;
}

std::string_view DiskFarkasCsvHeader() {
  return kCsvHeader;
}

DiskFarkasResult RunDiskFarkas(const DiskFarkasConfig& config) {
  ValidateConfig(config);
  const DiskSceneConfig& scene = config.scene;
  const double mass = scene.disk.density.value_or(1000.0) * kPi *
                      scene.disk.radius * scene.disk.radius *
                      scene.disk.thickness;
  const double frame_step = 1.0 / config.frames_per_second;
  const BuiltScene measurement = BuildScene(config, config.time_step, mass);
  std::optional<BuiltScene> distinct_settle;
  if (config.settle_time_step != config.time_step) {
    distinct_settle = BuildScene(config, config.settle_time_step, mass);
  }
  const BuiltScene& settling =
      distinct_settle.has_value() ? *distinct_settle : measurement;

  DiskFarkasResult result;
  result.mass = mass;
  result.weight = mass * kGravity;
  result.used_distinct_settle_plant = distinct_settle.has_value();

  std::unique_ptr<Context<double>> settle_context =
      settling.diagram->CreateDefaultContext();
  Context<double>& settle_plant_context =
      settling.diagram->GetMutableSubsystemContext(*settling.plant,
                                                   settle_context.get());
  settling.plant->SetFreeBodyPose(&settle_plant_context, *settling.disk_body,
                                  RigidTransformd(scene.disk.initial_position));
  settling.plant->SetFreeBodySpatialVelocity(&settle_plant_context,
                                             *settling.disk_body,
                                             SpatialVelocity<double>::Zero());

  Simulator<double> settle_simulator(*settling.diagram,
                                     std::move(settle_context));
  settle_simulator.set_target_realtime_rate(0.0);
  settle_simulator.Initialize();
  const int settle_frames = static_cast<int>(
      std::llround(config.settle_time * config.frames_per_second));
  DiskFarkasRow row =
      SampleRow(settling, settle_simulator.get_context(), scene.disk.radius);
  result.rows.push_back(row);
  for (int frame = 1; frame <= settle_frames; ++frame) {
    settle_simulator.AdvanceTo(frame * frame_step);
    row =
        SampleRow(settling, settle_simulator.get_context(), scene.disk.radius);
    result.contact_acquired = result.contact_acquired || row.hydro_contacts > 0;
    result.hydro_contact_samples += row.hydro_contacts > 0 ? 1 : 0;
    result.rows.push_back(row);
  }
  result.settled_normal_load = std::abs(row.normal_force_z);
  result.settled_normal_load_relative_error =
      std::abs(result.settled_normal_load - result.weight) / result.weight;

  std::unique_ptr<Context<double>> measurement_context =
      measurement.diagram->CreateDefaultContext();
  Context<double>& measurement_plant_context =
      measurement.diagram->GetMutableSubsystemContext(
          *measurement.plant, measurement_context.get());
  const Context<double>& settled_plant_context =
      settling.plant->GetMyContextFromRoot(settle_simulator.get_context());
  measurement_plant_context.get_mutable_discrete_state().SetFrom(
      settled_plant_context.get_discrete_state());
  result.transferred_discrete_state_groups =
      settled_plant_context.num_discrete_state_groups();
  result.state_transfer_exact = DiscreteStatesAreExactlyEqual(
      settled_plant_context, measurement_plant_context);

  const Vector3d initial_linear_velocity =
      scene.disk.initial_linear_velocity.value_or(Vector3d::Zero());
  const Vector3d initial_angular_velocity =
      scene.disk.initial_angular_velocity.value_or(Vector3d::Zero());
  measurement.plant->SetFreeBodySpatialVelocity(
      &measurement_plant_context, *measurement.disk_body,
      SpatialVelocity<double>(initial_angular_velocity,
                              initial_linear_velocity));
  measurement_context->SetTime(settle_frames * frame_step);

  result.initial_linear_speed = initial_linear_velocity.norm();
  result.initial_spin_speed = std::abs(initial_angular_velocity.z());
  result.initial_epsilon =
      result.initial_spin_speed > 1e-12
          ? result.initial_linear_speed /
                (result.initial_spin_speed * scene.disk.radius)
          : std::numeric_limits<double>::quiet_NaN();

  Simulator<double> simulator(*measurement.diagram,
                              std::move(measurement_context));
  simulator.set_target_realtime_rate(0.0);
  simulator.Initialize();
  for (int frame = settle_frames + 1;
       frame <= settle_frames + config.num_frames; ++frame) {
    simulator.AdvanceTo(frame * frame_step);
    row = SampleRow(measurement, simulator.get_context(), scene.disk.radius);
    row.post_kick = true;
    result.contact_acquired = result.contact_acquired || row.hydro_contacts > 0;
    result.hydro_contact_samples += row.hydro_contacts > 0 ? 1 : 0;
    result.post_kick_contact_samples += row.hydro_contacts > 0 ? 1 : 0;
    if (std::isfinite(row.eps) &&
        std::abs(row.angular_velocity_WD.z()) >= kSpinThreshold) {
      result.terminal_epsilon = row.eps;
    }
    result.rows.push_back(row);
  }
  result.final_linear_speed = row.linear_speed;
  result.final_spin_speed = std::abs(row.angular_velocity_WD.z());
  WriteCsv(config, result.rows);
  return result;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
