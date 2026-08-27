#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/visualization/visualization_config.h"
#include "drake/visualization/visualization_config_functions.h"

// Parameters for squeezing the spatula.
DEFINE_double(gripper_force, 1.5,
              "The baseline force to be applied by the gripper. [N].");
DEFINE_double(amplitude, 5,
              "The amplitude of the oscillations "
              "carried out by the gripper. [N].");
DEFINE_double(duty_cycle, 0.5, "Duty cycle of the control signal.");
DEFINE_double(period, 3, "Period of the control signal. [s].");

// MultibodyPlant settings.
DEFINE_double(stiction_tolerance, 1e-4, "Default stiction tolerance. [m/s].");
DEFINE_double(mbp_discrete_update_period, 4.0e-2,
              "If zero, the plant is modeled as a continuous system. "
              "If positive, the period (in seconds) of the discrete updates "
              "for the plant modeled as a discrete system."
              "This parameter must be non-negative.");
DEFINE_string(contact_model, "hydroelastic",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'.");
DEFINE_string(contact_approximation, "lagged",
              "Discrete contact approximation. Options are: "
              "'sap', 'similar', 'lagged'");
DEFINE_string(hydroelastic_representation, "tet",
              "Compliant hydroelastic representation. Options are: "
              "'tet' or 'voxel_sdf'.");
DEFINE_double(tet_resolution_hint_scale, 1.0,
              "Scale applied to every parsed tet resolution hint. Requires "
              "--hydroelastic_representation=tet.");
DEFINE_double(voxel_sdf_width_scale, 1.0,
              "Scale applied to every parsed resolution hint when it becomes "
              "a voxel width. Requires --hydroelastic_representation="
              "voxel_sdf.");
DEFINE_bool(validate_voxel_sdf_contacts, false,
            "Validate both voxel-SDF bubble contacts and their finite contact "
            "data. Requires --hydroelastic_representation=voxel_sdf.");
DEFINE_string(trajectory_csv, "",
              "Output CSV for the minimal spatula trajectory and per-finger "
              "contact wrenches. Empty disables trajectory recording.");
DEFINE_bool(report_simulation_timing, false,
            "Report simulation-only wall time and realtime factor. Requires "
            "--realtime_rate=0, --visualize=false, and no trajectory CSV.");

// Simulator settings.
DEFINE_double(realtime_rate, 1,
              "Desired rate of the simulation compared to realtime."
              "A value of 1 indicates real time.");
DEFINE_double(simulation_sec, 30, "Number of seconds to simulate. [s].");
DEFINE_bool(visualize, true, "Enable Meshcat visualization and recording.");
// The following flags are only in effect in continuous mode.
DEFINE_double(accuracy, 1.0e-3, "The integration accuracy.");
DEFINE_double(max_time_step, 1.0e-2,
              "The maximum time step the integrator is allowed to take, [s].");
DEFINE_string(integration_scheme, "implicit_euler",
              "Integration scheme to be used. Available options are: "
              "'semi_explicit_euler','runge_kutta2','runge_kutta3',"
              "'implicit_euler'");

namespace drake {

using geometry::SceneGraph;
using math::RigidTransform;
using math::RollPitchYaw;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::PrismaticJoint;
using systems::ApplySimulatorConfig;
using systems::BasicVector;
using systems::Context;
using systems::SimulatorConfig;
namespace examples {
namespace spatula_slip_control {
namespace {

struct ContactGeometryIds {
  geometry::GeometryId left_finger;
  geometry::GeometryId right_finger;
  geometry::GeometryId spatula;
};

struct TrajectorySample {
  double time{};
  Eigen::Vector3d position_W{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quaternion_WB{Eigen::Quaterniond::Identity()};
  double left_finger_position{};
  double right_finger_position{};
  int spatula_contacts{};
  Eigen::Vector3d left_contact_force_W{Eigen::Vector3d::Zero()};
  double left_handle_axis_torque{};
  Eigen::Vector3d right_contact_force_W{Eigen::Vector3d::Zero()};
  double right_handle_axis_torque{};
};

geometry::GeometryId GetOnlyCollisionGeometry(
    const MultibodyPlant<double>& plant, const std::string& body_name,
    ModelInstanceIndex model_instance) {
  const auto& body = plant.GetBodyByName(body_name, model_instance);
  const auto& geometry_ids = plant.GetCollisionGeometriesForBody(body);
  if (geometry_ids.size() != 1) {
    throw std::logic_error("Expected body '" + body_name +
                           "' to have exactly one collision geometry");
  }
  return geometry_ids.front();
}

ContactGeometryIds FindContactGeometryIds(const MultibodyPlant<double>& plant,
                                          ModelInstanceIndex gripper_instance,
                                          ModelInstanceIndex spatula_instance) {
  return ContactGeometryIds{
      .left_finger = GetOnlyCollisionGeometry(plant, "left_finger_bubble",
                                              gripper_instance),
      .right_finger = GetOnlyCollisionGeometry(plant, "right_finger_bubble",
                                               gripper_instance),
      .spatula = GetOnlyCollisionGeometry(plant, "spatula", spatula_instance),
  };
}

std::vector<geometry::GeometryId> UpdateCompliantGeometryProperties(
    const std::array<ModelInstanceIndex, 2>& model_instances,
    double resolution_hint_scale, bool use_voxel_sdf,
    MultibodyPlant<double>* plant, SceneGraph<double>* scene_graph) {
  DRAKE_DEMAND(plant != nullptr);
  DRAKE_DEMAND(scene_graph != nullptr);
  DRAKE_DEMAND(plant->get_source_id().has_value());
  DRAKE_DEMAND(resolution_hint_scale > 0.0 &&
               std::isfinite(resolution_hint_scale));

  std::vector<geometry::GeometryId> updated_ids;
  for (ModelInstanceIndex model_instance : model_instances) {
    for (multibody::BodyIndex body_index :
         plant->GetBodyIndices(model_instance)) {
      const auto& body = plant->get_body(body_index);
      for (geometry::GeometryId geometry_id :
           plant->GetCollisionGeometriesForBody(body)) {
        const geometry::ProximityProperties* old_properties =
            scene_graph->model_inspector().GetProximityProperties(geometry_id);
        DRAKE_DEMAND(old_properties != nullptr);
        const auto hydroelastic_type = old_properties->GetPropertyOrDefault(
            geometry::internal::kHydroGroup,
            geometry::internal::kComplianceType,
            geometry::internal::HydroelasticType::kUndefined);
        if (hydroelastic_type !=
            geometry::internal::HydroelasticType::kCompliant) {
          continue;
        }

        // Preserve the modulus, dissipation, and friction while scaling the
        // parsed resolution hint.
        const double old_resolution_hint = old_properties->GetProperty<double>(
            geometry::internal::kHydroGroup, geometry::internal::kRezHint);
        const double resolution_hint =
            old_resolution_hint * resolution_hint_scale;
        if (!(resolution_hint > 0.0 && std::isfinite(resolution_hint))) {
          throw std::logic_error("The scaled resolution hint is invalid");
        }
        geometry::ProximityProperties new_properties(*old_properties);
        new_properties.UpdateProperty(geometry::internal::kHydroGroup,
                                      geometry::internal::kRezHint,
                                      resolution_hint);
        if (use_voxel_sdf) {
          new_properties.UpdateProperty(
              geometry::internal::kHydroGroup,
              geometry::internal::kCompliantRepresentation,
              std::string("voxel_sdf"));
          new_properties.UpdateProperty(
              geometry::internal::kHydroGroup,
              geometry::internal::kVoxelSdfEvaluationMode,
              geometry::VoxelSdfEvaluationMode::kPrimitiveAffine);
        }
        scene_graph->AssignRole(*plant->get_source_id(), geometry_id,
                                new_properties, geometry::RoleAssign::kReplace);
        updated_ids.push_back(geometry_id);
      }
    }
  }

  if (updated_ids.size() != 3) {
    throw std::logic_error(
        "Expected to update exactly two bubble Ellipsoids and one spatula "
        "Cylinder");
  }
  return updated_ids;
}

void ValidateVoxelSdfContactResults(
    const MultibodyPlant<double>& plant,
    const systems::Context<double>& plant_context,
    const std::vector<geometry::GeometryId>& voxel_geometry_ids) {
  DRAKE_DEMAND(voxel_geometry_ids.size() == 3);
  const auto& contact_results =
      plant.get_contact_results_output_port()
          .Eval<multibody::ContactResults<double>>(plant_context);
  if (contact_results.num_point_pair_contacts() != 0 ||
      contact_results.num_hydroelastic_contacts() != 2) {
    throw std::runtime_error(
        "Expected exactly two hydroelastic bubble contacts and no point "
        "contacts");
  }

  std::array<bool, 3> geometry_seen{};
  auto mark_geometry_seen = [&voxel_geometry_ids,
                             &geometry_seen](geometry::GeometryId geometry_id) {
    const auto iter = std::find(voxel_geometry_ids.begin(),
                                voxel_geometry_ids.end(), geometry_id);
    if (iter == voxel_geometry_ids.end()) {
      throw std::runtime_error(
          "Hydroelastic contact contains an unexpected geometry");
    }
    geometry_seen[iter - voxel_geometry_ids.begin()] = true;
  };

  for (int i = 0; i < contact_results.num_hydroelastic_contacts(); ++i) {
    const auto& contact = contact_results.hydroelastic_contact_info(i);
    const auto& surface = contact.contact_surface();
    mark_geometry_seen(surface.id_M());
    mark_geometry_seen(surface.id_N());
    if (surface.is_triangle() || surface.num_faces() == 0 ||
        surface.num_vertices() == 0 ||
        !(surface.total_area() > 0.0 && std::isfinite(surface.total_area())) ||
        !contact.F_Ac_W().get_coeffs().allFinite() ||
        contact.F_Ac_W().translational().norm() == 0.0) {
      throw std::runtime_error(
          "Voxel-SDF bubble contact has invalid surface or force data");
    }
    bool has_positive_pressure = false;
    for (int v = 0; v < surface.num_vertices(); ++v) {
      const double pressure = surface.poly_e_MN().EvaluateAtVertex(v);
      if (!(pressure >= 0.0 && std::isfinite(pressure))) {
        throw std::runtime_error(
            "Voxel-SDF bubble contact has invalid pressure data");
      }
      has_positive_pressure = has_positive_pressure || pressure > 0.0;
    }
    if (!has_positive_pressure) {
      throw std::runtime_error(
          "Voxel-SDF bubble contact has no positive pressure");
    }
  }

  if (!std::all_of(geometry_seen.begin(), geometry_seen.end(), [](bool seen) {
        return seen;
      })) {
    throw std::runtime_error(
        "The two bubble contacts do not cover all three voxel geometries");
  }
}

TrajectorySample MakeTrajectorySample(
    const MultibodyPlant<double>& plant, const SceneGraph<double>& scene_graph,
    const systems::Context<double>& plant_context,
    const multibody::RigidBody<double>& spatula_body,
    const PrismaticJoint<double>& left_joint,
    const PrismaticJoint<double>& right_joint,
    const ContactGeometryIds& geometry_ids,
    const RigidTransform<double>& X_BH) {
  TrajectorySample sample;
  sample.time = plant_context.get_time();
  const RigidTransform<double>& X_WB =
      plant.EvalBodyPoseInWorld(plant_context, spatula_body);
  sample.position_W = X_WB.translation();
  sample.quaternion_WB = X_WB.rotation().ToQuaternion();
  sample.left_finger_position = left_joint.get_translation(plant_context);
  sample.right_finger_position = right_joint.get_translation(plant_context);

  const RigidTransform<double> X_WH = X_WB * X_BH;
  const Eigen::Vector3d p_WH_W = X_WH.translation();
  const Eigen::Vector3d handle_axis_W =
      X_WH.rotation().matrix() * Eigen::Vector3d::UnitZ();
  const auto& contact_results =
      plant.get_contact_results_output_port()
          .Eval<multibody::ContactResults<double>>(plant_context);
  if (contact_results.num_point_pair_contacts() != 0) {
    throw std::runtime_error(
        "Trajectory recording encountered point-contact fallback");
  }
  bool left_seen = false;
  bool right_seen = false;
  for (int i = 0; i < contact_results.num_hydroelastic_contacts(); ++i) {
    const auto& contact = contact_results.hydroelastic_contact_info(i);
    const auto& surface = contact.contact_surface();
    if (surface.is_triangle()) {
      throw std::runtime_error(
          "Trajectory recording requires polygon contact surfaces");
    }

    const bool spatula_is_M = surface.id_M() == geometry_ids.spatula;
    const bool spatula_is_N = surface.id_N() == geometry_ids.spatula;
    if (!spatula_is_M && !spatula_is_N) {
      const bool is_finger_pair =
          (surface.id_M() == geometry_ids.left_finger &&
           surface.id_N() == geometry_ids.right_finger) ||
          (surface.id_M() == geometry_ids.right_finger &&
           surface.id_N() == geometry_ids.left_finger);
      if (is_finger_pair) {
        continue;
      }
      throw std::runtime_error(
          "At t=" + std::to_string(sample.time) +
          ", hydroelastic contact between '" +
          scene_graph.model_inspector().GetName(surface.id_M()) + "' and '" +
          scene_graph.model_inspector().GetName(surface.id_N()) +
          "' is not part of the expected spatula or finger-finger contacts");
    }
    if (spatula_is_M && spatula_is_N) {
      throw std::runtime_error(
          "Hydroelastic contact contains the spatula geometry twice");
    }
    ++sample.spatula_contacts;
    const geometry::GeometryId finger_id =
        spatula_is_M ? surface.id_N() : surface.id_M();
    bool* seen = nullptr;
    Eigen::Vector3d* contact_force_W = nullptr;
    double* handle_axis_torque = nullptr;
    if (finger_id == geometry_ids.left_finger) {
      seen = &left_seen;
      contact_force_W = &sample.left_contact_force_W;
      handle_axis_torque = &sample.left_handle_axis_torque;
    } else if (finger_id == geometry_ids.right_finger) {
      seen = &right_seen;
      contact_force_W = &sample.right_contact_force_W;
      handle_axis_torque = &sample.right_handle_axis_torque;
    } else {
      throw std::runtime_error(
          "At t=" + std::to_string(sample.time) +
          ", the spatula contacts unexpected geometry '" +
          scene_graph.model_inspector().GetName(finger_id) + "'");
    }
    if (*seen) {
      throw std::runtime_error(
          "At t=" + std::to_string(sample.time) +
          ", trajectory recording encountered a duplicate finger contact");
    }
    *seen = true;

    if (!surface.centroid().allFinite() ||
        !contact.F_Ac_W().get_coeffs().allFinite()) {
      throw std::runtime_error(
          "Hydroelastic contact contains non-finite surface or wrench data");
    }
    const multibody::SpatialForce<double> F_Sc_W =
        spatula_is_M ? contact.F_Ac_W() : -contact.F_Ac_W();
    const multibody::SpatialForce<double> F_Sh_W =
        F_Sc_W.Shift(p_WH_W - surface.centroid());
    *contact_force_W = F_Sh_W.translational();
    *handle_axis_torque = F_Sh_W.rotational().dot(handle_axis_W);
  }

  if (!sample.position_W.allFinite() ||
      !sample.quaternion_WB.coeffs().allFinite() ||
      !std::isfinite(sample.left_finger_position) ||
      !std::isfinite(sample.right_finger_position) ||
      !sample.left_contact_force_W.allFinite() ||
      !std::isfinite(sample.left_handle_axis_torque) ||
      !sample.right_contact_force_W.allFinite() ||
      !std::isfinite(sample.right_handle_axis_torque)) {
    throw std::runtime_error(
        "Trajectory recording produced a non-finite sample");
  }
  return sample;
}

void WriteTrajectoryCsv(const std::vector<TrajectorySample>& samples,
                        const std::string& path) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Cannot open trajectory CSV '" + path + "'");
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "representation,time_s,x_m,y_m,z_m,qw,qx,qy,qz,"
            "left_finger_m,right_finger_m,spatula_contacts,"
            "left_contact_force_x_N,left_contact_force_y_N,"
            "left_contact_force_z_N,left_handle_axis_torque_Nm,"
            "right_contact_force_x_N,right_contact_force_y_N,"
            "right_contact_force_z_N,right_handle_axis_torque_Nm\n";
  for (const TrajectorySample& sample : samples) {
    const Eigen::Quaterniond& q = sample.quaternion_WB;
    output << FLAGS_hydroelastic_representation << "," << sample.time << ","
           << sample.position_W.x() << "," << sample.position_W.y() << ","
           << sample.position_W.z() << "," << q.w() << "," << q.x() << ","
           << q.y() << "," << q.z() << "," << sample.left_finger_position << ","
           << sample.right_finger_position << "," << sample.spatula_contacts
           << "," << sample.left_contact_force_W.x() << ","
           << sample.left_contact_force_W.y() << ","
           << sample.left_contact_force_W.z() << ","
           << sample.left_handle_axis_torque << ","
           << sample.right_contact_force_W.x() << ","
           << sample.right_contact_force_W.y() << ","
           << sample.right_contact_force_W.z() << ","
           << sample.right_handle_axis_torque << "\n";
  }
  if (!output) {
    throw std::runtime_error("Failed while writing trajectory CSV '" + path +
                             "'");
  }
}

// We create a simple leaf system that outputs a square wave signal for our
// open loop controller. The Square system here supports an arbitrarily
// dimensional signal, but we will use a 2-dimensional signal for our gripper.
class Square final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Square);

  // Constructs a %Square system where different amplitudes, duty cycles,
  // periods, and phases can be applied to each square wave.
  //
  // @param[in] amplitudes the square wave amplitudes. (unitless)
  // @param[in] duty_cycles the square wave duty cycles.
  //                        (ratio of pulse duration to period of the waveform)
  // @param[in] periods the square wave periods. (seconds)
  // @param[in] phases the square wave phases. (radians)
  Square(const Eigen::VectorXd& amplitudes, const Eigen::VectorXd& duty_cycles,
         const Eigen::VectorXd& periods, const Eigen::VectorXd& phases)
      : amplitude_(amplitudes),
        duty_cycle_(duty_cycles),
        period_(periods),
        phase_(phases) {
    // Ensure the incoming vectors are all the same size.
    DRAKE_THROW_UNLESS(duty_cycles.size() == amplitudes.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == periods.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == phases.size());

    this->DeclareVectorOutputPort("Square Wave Output", duty_cycles.size(),
                                  &Square::CalcValueOutput);
  }

 private:
  void CalcValueOutput(const Context<double>& context,
                       BasicVector<double>* output) const {
    Eigen::VectorBlock<VectorX<double>> output_block =
        output->get_mutable_value();

    const double time = context.get_time();

    for (int i = 0; i < duty_cycle_.size(); ++i) {
      // Add phase offset.
      double t = time + (period_[i] * phase_[i] / (2 * M_PI));

      output_block[i] =
          amplitude_[i] *
          (t - floor(t / period_[i]) * period_[i] < duty_cycle_[i] * period_[i]
               ? 1
               : 0);
    }
  }

  const Eigen::VectorXd amplitude_;
  const Eigen::VectorXd duty_cycle_;
  const Eigen::VectorXd period_;
  const Eigen::VectorXd phase_;
};

int DoMain() {
  if (FLAGS_hydroelastic_representation != "tet" &&
      FLAGS_hydroelastic_representation != "voxel_sdf") {
    throw std::logic_error(
        "--hydroelastic_representation must be 'tet' or 'voxel_sdf'");
  }
  if (FLAGS_hydroelastic_representation == "voxel_sdf" &&
      FLAGS_contact_surface_representation != "polygon") {
    throw std::logic_error(
        "--hydroelastic_representation=voxel_sdf requires "
        "--contact_surface_representation=polygon");
  }
  if (!(FLAGS_tet_resolution_hint_scale > 0.0 &&
        std::isfinite(FLAGS_tet_resolution_hint_scale))) {
    throw std::logic_error(
        "--tet_resolution_hint_scale must be finite and positive");
  }
  if (FLAGS_hydroelastic_representation != "tet" &&
      FLAGS_tet_resolution_hint_scale != 1.0) {
    throw std::logic_error(
        "--tet_resolution_hint_scale requires "
        "--hydroelastic_representation=tet");
  }
  if (!(FLAGS_voxel_sdf_width_scale > 0.0 &&
        std::isfinite(FLAGS_voxel_sdf_width_scale))) {
    throw std::logic_error(
        "--voxel_sdf_width_scale must be finite and positive");
  }
  if (FLAGS_hydroelastic_representation != "voxel_sdf" &&
      FLAGS_voxel_sdf_width_scale != 1.0) {
    throw std::logic_error(
        "--voxel_sdf_width_scale requires "
        "--hydroelastic_representation=voxel_sdf");
  }
  if (FLAGS_validate_voxel_sdf_contacts &&
      FLAGS_hydroelastic_representation != "voxel_sdf") {
    throw std::logic_error(
        "--validate_voxel_sdf_contacts requires "
        "--hydroelastic_representation=voxel_sdf");
  }
  if (!FLAGS_trajectory_csv.empty() &&
      FLAGS_mbp_discrete_update_period <= 0.0) {
    throw std::logic_error(
        "--trajectory_csv requires a positive "
        "--mbp_discrete_update_period");
  }
  if (FLAGS_simulation_sec < 0.0) {
    throw std::logic_error("--simulation_sec must be non-negative");
  }
  if (FLAGS_report_simulation_timing) {
    if (FLAGS_realtime_rate != 0.0) {
      throw std::logic_error(
          "--report_simulation_timing requires --realtime_rate=0");
    }
    if (FLAGS_visualize) {
      throw std::logic_error(
          "--report_simulation_timing requires --visualize=false");
    }
    if (!FLAGS_trajectory_csv.empty()) {
      throw std::logic_error(
          "--report_simulation_timing requires an empty --trajectory_csv");
    }
    if (FLAGS_simulation_sec <= 0.0) {
      throw std::logic_error(
          "--report_simulation_timing requires --simulation_sec>0");
    }
  }

  // Construct a MultibodyPlant and a SceneGraph.
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_mbp_discrete_update_period;
  plant_config.stiction_tolerance = FLAGS_stiction_tolerance;
  plant_config.contact_model = FLAGS_contact_model;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;
  plant_config.contact_surface_representation =
      FLAGS_contact_surface_representation;
  plant_config.use_sampled_output_ports = FLAGS_trajectory_csv.empty();

  DRAKE_DEMAND(FLAGS_mbp_discrete_update_period >= 0);
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  // Parse the gripper and spatula models.
  multibody::Parser parser(&builder);
  const auto gripper_instances = parser.AddModelsFromUrl(
      "package://drake_models/wsg_50_description/sdf/"
      "schunk_wsg_50_hydro_bubble.sdf");
  const auto spatula_instances = parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/spatula_slip_control/"
      "spatula.sdf");
  DRAKE_THROW_UNLESS(gripper_instances.size() == 1);
  DRAKE_THROW_UNLESS(spatula_instances.size() == 1);
  const ContactGeometryIds contact_geometry_ids =
      FindContactGeometryIds(plant, gripper_instances[0], spatula_instances[0]);
  const RigidTransform<double> X_BH =
      scene_graph.model_inspector().GetPoseInFrame(
          contact_geometry_ids.spatula);
  std::vector<geometry::GeometryId> voxel_geometry_ids;
  if (FLAGS_hydroelastic_representation == "voxel_sdf") {
    voxel_geometry_ids = UpdateCompliantGeometryProperties(
        {gripper_instances[0], spatula_instances[0]},
        FLAGS_voxel_sdf_width_scale, true, &plant, &scene_graph);
  } else if (FLAGS_tet_resolution_hint_scale != 1.0) {
    UpdateCompliantGeometryProperties(
        {gripper_instances[0], spatula_instances[0]},
        FLAGS_tet_resolution_hint_scale, false, &plant, &scene_graph);
  }
  // Pose the gripper and weld it to the world.
  const math::RigidTransform<double> X_WF0 = math::RigidTransform<double>(
      math::RollPitchYaw(0.0, -1.57, 0.0), Eigen::Vector3d(0, 0, 0.25));
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("gripper"), X_WF0);
  plant.Finalize();

  // Construct the open loop square wave controller. To oscillate around a
  // constant force, we construct a ConstantVectorSource and combine it with
  // the square wave output using an Adder.
  const double f0 = FLAGS_gripper_force;

  const Eigen::Vector2d amplitudes(FLAGS_amplitude, -FLAGS_amplitude);
  const Eigen::Vector2d duty_cycles(FLAGS_duty_cycle, FLAGS_duty_cycle);
  const Eigen::Vector2d periods(FLAGS_period, FLAGS_period);
  const Eigen::Vector2d phases(0, 0);
  const auto& square_force =
      *builder.AddSystem<Square>(amplitudes, duty_cycles, periods, phases);
  const auto& constant_force =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector2d(f0, -f0));
  const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
  builder.Connect(square_force.get_output_port(), adder.get_input_port(0));
  builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));

  // Connect the output of the adder to the plant's actuation input.
  builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

  std::shared_ptr<geometry::Meshcat> meshcat;
  if (FLAGS_visualize) {
    meshcat = std::make_shared<geometry::Meshcat>();
    visualization::ApplyVisualizationConfig(
        visualization::VisualizationConfig{
            .default_proximity_color = geometry::Rgba{1, 0, 0, 0.25},
            .enable_alpha_sliders = true,
        },
        &builder, nullptr, nullptr, nullptr, meshcat);
  }

  // Construct a simulator.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

  SimulatorConfig sim_config;
  sim_config.integration_scheme = FLAGS_integration_scheme;
  sim_config.max_step_size = FLAGS_max_time_step;
  sim_config.accuracy = FLAGS_accuracy;
  sim_config.target_realtime_rate = FLAGS_realtime_rate;

  systems::Simulator<double> simulator(*diagram);
  ApplySimulatorConfig(sim_config, &simulator);

  // Set the initial conditions for the spatula pose and the gripper finger
  // positions.
  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);

  // Set spatula's free body pose.
  const math::RigidTransform<double> X_WF1 = math::RigidTransform<double>(
      math::RollPitchYaw(-0.4, 0.0, 1.57), Eigen::Vector3d(0.35, 0, 0.25));
  const auto& base_link = plant.GetBodyByName("spatula");
  plant.SetFreeBodyPose(&plant_context, base_link, X_WF1);

  // Set finger joint positions.
  const PrismaticJoint<double>& left_joint =
      plant.GetJointByName<PrismaticJoint>("left_finger_sliding_joint");
  left_joint.set_translation(&plant_context, -0.01);
  const PrismaticJoint<double>& right_joint =
      plant.GetJointByName<PrismaticJoint>("right_finger_sliding_joint");
  right_joint.set_translation(&plant_context, 0.01);

  // Simulate.
  simulator.Initialize();
  if (meshcat != nullptr) {
    meshcat->StartRecording();
  }
  std::vector<TrajectorySample> trajectory;
  if (FLAGS_trajectory_csv.empty()) {
    if (FLAGS_report_simulation_timing) {
      // Initialize() resets these statistics before it processes initialization
      // events. Reset once more here so the timer covers AdvanceTo() only.
      simulator.ResetStatistics();
    }
    const double initial_simulation_time = simulator.get_context().get_time();
    simulator.AdvanceTo(FLAGS_simulation_sec);
    if (FLAGS_report_simulation_timing) {
      const double simulated_seconds =
          simulator.get_context().get_time() - initial_simulation_time;
      const double realtime_factor = simulator.get_actual_realtime_rate();
      const double wall_seconds = simulated_seconds / realtime_factor;
      std::cout << std::setprecision(17)
                << "simulation_time_s: " << simulated_seconds << "\n"
                << "simulation_wall_time_s: " << wall_seconds << "\n"
                << "simulation_rtf: " << realtime_factor << "\n"
                << "simulation_steps: " << simulator.get_num_steps_taken()
                << "\n"
                << "simulation_discrete_updates: "
                << simulator.get_num_discrete_updates() << "\n"
                << "simulation_unrestricted_updates: "
                << simulator.get_num_unrestricted_updates() << "\n";
    }
  } else {
    trajectory.push_back(MakeTrajectorySample(
        plant, scene_graph, plant_context, base_link, left_joint, right_joint,
        contact_geometry_ids, X_BH));
    const double time_tolerance = 1.0e-12 * std::max(1.0, FLAGS_simulation_sec);
    while (FLAGS_simulation_sec - simulator.get_context().get_time() >
           time_tolerance) {
      const double next_time = std::min(
          simulator.get_context().get_time() + FLAGS_mbp_discrete_update_period,
          FLAGS_simulation_sec);
      simulator.AdvanceTo(next_time);
      trajectory.push_back(MakeTrajectorySample(
          plant, scene_graph, plant_context, base_link, left_joint, right_joint,
          contact_geometry_ids, X_BH));
    }
    WriteTrajectoryCsv(trajectory, FLAGS_trajectory_csv);
    std::cout << "trajectory_csv: " << FLAGS_trajectory_csv << "\n";
  }
  if (FLAGS_validate_voxel_sdf_contacts) {
    ValidateVoxelSdfContactResults(plant, plant_context, voxel_geometry_ids);
    if (!plant.GetPositionsAndVelocities(plant_context).allFinite()) {
      throw std::runtime_error(
          "Voxel-SDF simulation produced a non-finite plant state");
    }
  }
  if (meshcat != nullptr) {
    meshcat->StopRecording();

    // TODO(#19142) According to issue 19142, we can playback contact forces
    // and torques; however, contact surfaces are not recorded properly.
    // For now, we delete contact surfaces to prevent confusion in the
    // playback. Remove deletion when 19142 is resolved.
    meshcat->Delete(
        "/drake/contact_forces/hydroelastic/"
        "left_finger_bubble+spatula/contact_surface");
    meshcat->Delete(
        "/drake/contact_forces/hydroelastic/"
        "right_finger_bubble+spatula/contact_surface");
    meshcat->PublishRecording();
  }

  systems::PrintSimulatorStatistics(simulator);
  return 0;
}

}  // namespace
}  // namespace spatula_slip_control
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is an example of using the hydroelastic contact model with a\n"
      "robot gripper with compliant bubble fingers and a compliant spatula.\n"
      "The example poses the spatula in the closed grip of the gripper and\n"
      "uses an open loop square wave controller to perform a controlled\n"
      "rotational slip of the spatula while maintaining the spatula in\n"
      "the gripper's grasp. Use the MeshCat URL from the console log\n"
      "messages for visualization. See the README.md file for more\n"
      "information.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::spatula_slip_control::DoMain();
}
