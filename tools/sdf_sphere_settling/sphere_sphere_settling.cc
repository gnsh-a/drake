#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_string(representation, "both",
              "Contact representation: tet, affine, or both.");
DEFINE_double(time_step, 1.0e-4, "Discrete time step in seconds.");
DEFINE_double(duration, 1.0, "Simulation duration in seconds.");
DEFINE_double(grid_roll_deg, 0.0,
              "Initial roll of the moving sphere's body-fixed grid.");
DEFINE_double(grid_pitch_deg, 0.0,
              "Initial pitch of the moving sphere's body-fixed grid.");
DEFINE_double(grid_yaw_deg, 0.0,
              "Initial yaw of the moving sphere's body-fixed grid.");

namespace drake {
namespace tools {
namespace sdf_sphere_settling {
namespace {

using Eigen::Vector3d;
using geometry::ContactSurface;
using geometry::GeometryId;
using geometry::ProximityProperties;
using math::RigidTransformd;
using math::RollPitchYawd;
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

// This is the original R = 1 m, h = 0.1 m, delta = 0.2 m experiment
// scaled by ten in length and by one thousand in modulus. The dimensionless
// geometry and grid phase are unchanged, while the force is reduced from
// meganewtons to tens of newtons.
constexpr double kRadius = 0.1;
constexpr double kResolution = 0.01;
constexpr double kHydroelasticModulus = 1.0e5;
constexpr double kTargetPenetration = 0.02;
constexpr double kInitialGap = 0.001;
constexpr double kGravity = 9.81;
constexpr double kReferenceForce = 28.372195914103273;
constexpr double kMass = kReferenceForce / kGravity;
constexpr double kDegreesToRadians = 0.017453292519943295;

enum class Representation { kTet, kAffine };

struct BodyState {
  Vector3d position_W{};
  Vector3d translational_velocity_W{};
  Vector3d angular_velocity_W{};
  double penetration{};
  double axial_velocity{};
  double lateral_offset{};
  double angular_speed{};
};

struct Summary {
  Representation representation{};
  int num_steps{};
  int max_hydroelastic_contacts{};
  int max_point_contacts{};
  bool contact_observed{};
  std::optional<double> first_contact_time;
  std::optional<double> first_contact_loss_time;
  double min_contact_penetration{std::numeric_limits<double>::infinity()};
  double max_penetration{};
  double max_support_force{};
  double max_transverse_force{};
  double max_torque_about_com{};
  double max_lateral_offset{};
  double max_angular_speed{};
  std::optional<double> closest_target_error;
  double closest_target_time{};
  double closest_target_penetration{};
  double closest_target_support_force{};
  double closest_target_surface_area{};
  int closest_target_surface_faces{};
  double final_penetration{};
  double final_axial_velocity{};
  double final_lateral_offset{};
};

std::string_view ToString(Representation representation) {
  switch (representation) {
    case Representation::kTet:
      return "tet";
    case Representation::kAffine:
      return "affine";
  }
  throw std::logic_error("Unknown representation");
}

std::vector<Representation> ParseRepresentations(std::string_view value) {
  if (value == "tet") return {Representation::kTet};
  if (value == "affine") return {Representation::kAffine};
  if (value == "both") {
    return {Representation::kTet, Representation::kAffine};
  }
  throw std::runtime_error(
      "--representation must be tet, affine, or both; got '" +
      std::string(value) + "'");
}

void ValidateFlags() {
  if (!(std::isfinite(FLAGS_time_step) && FLAGS_time_step > 0.0)) {
    throw std::runtime_error("--time_step must be finite and positive");
  }
  if (!(std::isfinite(FLAGS_duration) && FLAGS_duration > 0.0)) {
    throw std::runtime_error("--duration must be finite and positive");
  }
  for (const auto& [value, name] :
       std::vector<std::pair<double, std::string_view>>{
           {FLAGS_grid_roll_deg, "--grid_roll_deg"},
           {FLAGS_grid_pitch_deg, "--grid_pitch_deg"},
           {FLAGS_grid_yaw_deg, "--grid_yaw_deg"}}) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string(name) + " must be finite");
    }
  }
}

ProximityProperties MakeProperties(Representation representation) {
  ProximityProperties properties;
  // kLagged uses Hunt-Crossley dissipation and ignores relaxation time. Set
  // Hunt-Crossley dissipation and friction to zero, and deliberately do not
  // add a relaxation-time property.
  geometry::AddContactMaterial(0.0, std::nullopt,
                               CoulombFriction<double>(0.0, 0.0), &properties);
  if (representation == Representation::kTet) {
    geometry::AddCompliantHydroelasticProperties(
        kResolution, kHydroelasticModulus, &properties);
  } else {
    geometry::AddCompliantHydroelasticVoxelSdfProperties(
        kResolution, kHydroelasticModulus,
        geometry::VoxelSdfEvaluationMode::kPrimitiveSdf, &properties);
  }
  return properties;
}

BodyState EvalBodyState(const MultibodyPlant<double>& plant,
                        const Context<double>& context,
                        const RigidBody<double>& moving_sphere) {
  BodyState state;
  state.position_W =
      plant.EvalBodyPoseInWorld(context, moving_sphere).translation();
  const SpatialVelocity<double> V_WB =
      plant.EvalBodySpatialVelocityInWorld(context, moving_sphere);
  state.translational_velocity_W = V_WB.translational();
  state.angular_velocity_W = V_WB.rotational();
  state.penetration = std::max(0.0, 2.0 * kRadius - state.position_W.norm());
  state.axial_velocity = state.translational_velocity_W.x();
  state.lateral_offset = state.position_W.tail<2>().norm();
  state.angular_speed = state.angular_velocity_W.norm();
  if (!state.position_W.allFinite() ||
      !state.translational_velocity_W.allFinite() ||
      !state.angular_velocity_W.allFinite()) {
    throw std::runtime_error("The moving-sphere state became non-finite");
  }
  return state;
}

SpatialForce<double> CalcForceOnMovingSphereAtCom(
    const multibody::HydroelasticContactInfo<double>& info,
    GeometryId moving_geometry_id, const Vector3d& p_WB) {
  const ContactSurface<double>& surface = info.contact_surface();
  SpatialForce<double> F_Bc_W;
  if (surface.id_M() == moving_geometry_id) {
    F_Bc_W = info.F_Ac_W();
  } else if (surface.id_N() == moving_geometry_id) {
    F_Bc_W = -info.F_Ac_W();
  } else {
    throw std::runtime_error(
        "Hydroelastic contact does not contain the moving sphere");
  }
  return F_Bc_W.Shift(p_WB - surface.centroid());
}

Summary RunSimulation(Representation representation) {
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, FLAGS_time_step);
  static_cast<void>(scene_graph);
  plant.set_contact_model(ContactModel::kHydroelastic);
  plant.set_discrete_contact_approximation(
      DiscreteContactApproximation::kLagged);
  plant.set_contact_surface_representation(
      geometry::HydroelasticContactRepresentation::kPolygon);
  plant.SetUseSampledOutputPorts(true);
  plant.mutable_gravity_field().set_gravity_vector(-kGravity *
                                                   Vector3d::UnitX());

  const geometry::Sphere sphere(kRadius);
  const ProximityProperties properties = MakeProperties(representation);
  const RigidBody<double>& moving_sphere = plant.AddRigidBody(
      "moving_sphere",
      SpatialInertia<double>::SolidSphereWithMass(kMass, kRadius));
  // Equal voxel widths traverse the lower-GeometryId grid. Register the moving
  // sphere first so the RPY flags rotate the grid that is actually traversed;
  // the anchored sphere remains the analytically queried primitive.
  const GeometryId moving_geometry_id = plant.RegisterCollisionGeometry(
      moving_sphere, RigidTransformd(), sphere, "moving_sphere", properties);
  plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(), sphere,
                                  "anchored_sphere", properties);
  plant.Finalize();

  std::unique_ptr<Diagram<double>> diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(0.0);
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(&simulator.get_mutable_context());
  const RollPitchYawd initial_grid_orientation(
      kDegreesToRadians * FLAGS_grid_roll_deg,
      kDegreesToRadians * FLAGS_grid_pitch_deg,
      kDegreesToRadians * FLAGS_grid_yaw_deg);
  plant.SetFreeBodyPose(
      &plant_context, moving_sphere,
      RigidTransformd(initial_grid_orientation,
                      (2.0 * kRadius + kInitialGap) * Vector3d::UnitX()));
  plant.SetFreeBodySpatialVelocity(&plant_context, moving_sphere,
                                   SpatialVelocity<double>::Zero());
  simulator.Initialize();

  Summary summary;
  summary.representation = representation;
  bool was_in_contact = false;
  const double time_tolerance = 1.0e-12 * std::max(1.0, FLAGS_duration);
  while (FLAGS_duration - simulator.get_context().get_time() > time_tolerance) {
    const double t_start = simulator.get_context().get_time();
    const Context<double>& start_context =
        plant.GetMyContextFromRoot(simulator.get_context());
    const BodyState start = EvalBodyState(plant, start_context, moving_sphere);

    const double t_end = std::min(t_start + FLAGS_time_step, FLAGS_duration);
    simulator.AdvanceTo(t_end);
    ++summary.num_steps;

    const Context<double>& end_context =
        plant.GetMyContextFromRoot(simulator.get_context());
    const BodyState end = EvalBodyState(plant, end_context, moving_sphere);
    summary.max_penetration =
        std::max({summary.max_penetration, start.penetration, end.penetration});
    summary.max_lateral_offset =
        std::max(summary.max_lateral_offset, end.lateral_offset);
    summary.max_angular_speed =
        std::max(summary.max_angular_speed, end.angular_speed);

    // With sampled output ports, the contact results emitted after this update
    // correspond to the pre-step state. Pair them with `start`, not `end`.
    const ContactResults<double>& contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            end_context);
    const int point_contacts = contacts.num_point_pair_contacts();
    const int hydro_contacts = contacts.num_hydroelastic_contacts();
    summary.max_point_contacts =
        std::max(summary.max_point_contacts, point_contacts);
    summary.max_hydroelastic_contacts =
        std::max(summary.max_hydroelastic_contacts, hydro_contacts);
    if (point_contacts != 0) {
      throw std::runtime_error("Unexpected point-contact fallback");
    }
    if (hydro_contacts > 1) {
      throw std::runtime_error(
          "Expected at most one sphere-sphere hydroelastic contact");
    }

    if (hydro_contacts == 1) {
      const auto& info = contacts.hydroelastic_contact_info(0);
      const ContactSurface<double>& surface = info.contact_surface();
      if (surface.is_triangle()) {
        throw std::runtime_error("Expected polygonal hydroelastic contact");
      }
      const SpatialForce<double> F_Bo_W = CalcForceOnMovingSphereAtCom(
          info, moving_geometry_id, start.position_W);
      const double support_force = F_Bo_W.translational().x();
      const double transverse_force = F_Bo_W.translational().tail<2>().norm();
      summary.contact_observed = true;
      if (!summary.first_contact_time.has_value()) {
        summary.first_contact_time = t_start;
      }
      summary.min_contact_penetration =
          std::min(summary.min_contact_penetration, start.penetration);
      summary.max_support_force =
          std::max(summary.max_support_force, support_force);
      summary.max_transverse_force =
          std::max(summary.max_transverse_force, transverse_force);
      summary.max_torque_about_com =
          std::max(summary.max_torque_about_com, F_Bo_W.rotational().norm());

      const double target_error =
          std::abs(start.penetration - kTargetPenetration);
      if (!summary.closest_target_error.has_value() ||
          target_error < *summary.closest_target_error) {
        summary.closest_target_error = target_error;
        summary.closest_target_time = t_start;
        summary.closest_target_penetration = start.penetration;
        summary.closest_target_support_force = support_force;
        summary.closest_target_surface_area = surface.total_area();
        summary.closest_target_surface_faces = surface.num_faces();
      }
      was_in_contact = true;
    } else if (was_in_contact && !summary.first_contact_loss_time.has_value()) {
      summary.first_contact_loss_time = t_start;
      was_in_contact = false;
    }
  }

  const Context<double>& final_context =
      plant.GetMyContextFromRoot(simulator.get_context());
  const BodyState final = EvalBodyState(plant, final_context, moving_sphere);
  summary.final_penetration = final.penetration;
  summary.final_axial_velocity = final.axial_velocity;
  summary.final_lateral_offset = final.lateral_offset;
  return summary;
}

void PrintOptional(std::string_view name, const std::optional<double>& value) {
  std::cout << "  " << name << ": ";
  if (value.has_value()) {
    std::cout << *value;
  } else {
    std::cout << "n/a";
  }
  std::cout << "\n";
}

void PrintSummary(const Summary& summary) {
  std::cout << "\n" << ToString(summary.representation) << ":\n";
  std::cout << "  steps: " << summary.num_steps << "\n";
  std::cout << "  contact_observed: " << std::boolalpha
            << summary.contact_observed << "\n";
  std::cout << "  max_hydroelastic_contacts: "
            << summary.max_hydroelastic_contacts << "\n";
  std::cout << "  max_point_contacts: " << summary.max_point_contacts << "\n";
  PrintOptional("first_contact_time_s", summary.first_contact_time);
  PrintOptional("first_contact_loss_time_s", summary.first_contact_loss_time);
  if (summary.contact_observed) {
    std::cout << "  min_contact_penetration_m: "
              << summary.min_contact_penetration << "\n";
  }
  std::cout << "  max_penetration_m: " << summary.max_penetration << "\n";
  std::cout << "  max_support_force_N: " << summary.max_support_force << "\n";
  std::cout << "  max_transverse_force_N: " << summary.max_transverse_force
            << "\n";
  std::cout << "  max_torque_about_com_Nm: " << summary.max_torque_about_com
            << "\n";
  std::cout << "  max_lateral_offset_m: " << summary.max_lateral_offset << "\n";
  std::cout << "  max_angular_speed_rad_s: " << summary.max_angular_speed
            << "\n";
  if (summary.closest_target_error.has_value()) {
    std::cout << "  closest_target_time_s: " << summary.closest_target_time
              << "\n";
    std::cout << "  closest_target_penetration_m: "
              << summary.closest_target_penetration << "\n";
    std::cout << "  closest_target_error_m: " << *summary.closest_target_error
              << "\n";
    std::cout << "  closest_target_support_force_N: "
              << summary.closest_target_support_force << "\n";
    std::cout << "  closest_target_surface_area_m2: "
              << summary.closest_target_surface_area << "\n";
    std::cout << "  closest_target_surface_faces: "
              << summary.closest_target_surface_faces << "\n";
  }
  std::cout << "  final_penetration_m: " << summary.final_penetration << "\n";
  std::cout << "  final_axial_velocity_m_s: " << summary.final_axial_velocity
            << "\n";
  std::cout << "  final_lateral_offset_m: " << summary.final_lateral_offset
            << "\n";
}

void PrintComparison(const Summary& tet, const Summary& affine) {
  if (!tet.closest_target_error.has_value() ||
      !affine.closest_target_error.has_value()) {
    return;
  }
  const double force_delta =
      affine.closest_target_support_force - tet.closest_target_support_force;
  const double area_delta =
      affine.closest_target_surface_area - tet.closest_target_surface_area;
  std::cout << "\ncomparison_at_delta_target:\n";
  std::cout << "  affine_minus_tet_force_N: " << force_delta << "\n";
  if (tet.closest_target_support_force != 0.0) {
    std::cout << "  affine_minus_tet_force_percent: "
              << 100.0 * force_delta / tet.closest_target_support_force << "\n";
  }
  std::cout << "  affine_minus_tet_area_m2: " << area_delta << "\n";
  if (tet.closest_target_surface_area != 0.0) {
    std::cout << "  affine_minus_tet_area_percent: "
              << 100.0 * area_delta / tet.closest_target_surface_area << "\n";
  }
}

int DoMain() {
  ValidateFlags();
  const std::vector<Representation> representations =
      ParseRepresentations(FLAGS_representation);

  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  std::cout << "scaled_sphere_sphere_lagged_sap:\n";
  std::cout << "  radius_m: " << kRadius << "\n";
  std::cout << "  resolution_m: " << kResolution << "\n";
  std::cout << "  target_penetration_m: " << kTargetPenetration << "\n";
  std::cout << "  hydroelastic_modulus_Pa: " << kHydroelasticModulus << "\n";
  std::cout << "  initial_gap_m: " << kInitialGap << "\n";
  std::cout << "  reference_force_N: " << kReferenceForce << "\n";
  std::cout << "  mass_kg: " << kMass << "\n";
  std::cout << "  time_step_s: " << FLAGS_time_step << "\n";
  std::cout << "  duration_s: " << FLAGS_duration << "\n";
  std::cout << "  hunt_crossley_dissipation: 0\n";
  std::cout << "  relaxation_time: not specified\n";
  std::cout << "  moving_grid_rpy_deg: [" << FLAGS_grid_roll_deg << ", "
            << FLAGS_grid_pitch_deg << ", " << FLAGS_grid_yaw_deg << "]\n";

  std::vector<Summary> summaries;
  for (const Representation representation : representations) {
    summaries.push_back(RunSimulation(representation));
    PrintSummary(summaries.back());
  }
  if (summaries.size() == 2) {
    PrintComparison(summaries[0], summaries[1]);
  }
  return 0;
}

}  // namespace
}  // namespace sdf_sphere_settling
}  // namespace tools
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Runs a scaled free sphere-sphere lagged-SAP comparison using tet and/or "
      "primitive-affine hydroelastic contact.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    return drake::tools::sdf_sphere_settling::DoMain();
  } catch (const std::exception& error) {
    std::cerr << "sphere_sphere_settling: " << error.what() << "\n";
    return 1;
  }
}
