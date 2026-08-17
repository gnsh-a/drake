#include "drake/tools/voxel_sdf_experiments/spatula_slip/spatula_slip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/common/drake_copyable.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/tools/voxel_sdf_experiments/common/emit.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Vector3d;
using geometry::GeometryId;
using geometry::ProximityProperties;
using geometry::SceneGraph;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::ContactModel;
using multibody::ContactResults;
using multibody::DiscreteContactApproximation;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::PrismaticJoint;
using multibody::RigidBody;
using multibody::SpatialForce;
using systems::BasicVector;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

constexpr double kNegativePressureTolerance = 1.0e-7;
constexpr std::string_view kCsvHeader =
    "schema_version,git_commit,git_dirty,representation,resolution_scale,"
    "ellipsoid_resolution_m,cylinder_resolution_m,time_step_s,"
    "sample_period_s,time_s,x_m,y_m,z_m,qw,qx,qy,qz,handle_axis_x,"
    "handle_axis_y,handle_axis_z,left_finger_m,right_finger_m,"
    "point_contacts,hydro_contacts,spatula_contacts,"
    "left_contact_force_x_n,left_contact_force_y_n,left_contact_force_z_n,"
    "left_handle_axis_torque_nm,left_contact_area_m2,left_surface_faces,"
    "right_contact_force_x_n,right_contact_force_y_n,"
    "right_contact_force_z_n,right_handle_axis_torque_nm,"
    "right_contact_area_m2,right_surface_faces";

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

void ThrowUnlessIntegerMultiple(double larger, double smaller,
                                std::string_view description) {
  const double ratio = larger / smaller;
  if (std::abs(ratio - std::round(ratio)) >
      1.0e-12 * std::max(1.0, std::abs(ratio))) {
    throw std::logic_error(std::string(description) +
                           " must be an integer multiple");
  }
}

void ValidateConfig(const SpatulaSlipConfig& config) {
  ThrowUnlessFinitePositive(config.resolution_scale, "resolution_scale");
  ThrowUnlessFinitePositive(config.time_step, "time_step");
  ThrowUnlessFinitePositive(config.sample_period, "sample_period");
  ThrowUnlessFinitePositive(config.duration, "duration");
  ThrowUnlessFinitePositive(config.stiction_tolerance, "stiction_tolerance");
  ThrowUnlessFiniteNonnegative(config.gripper_force, "gripper_force");
  ThrowUnlessFiniteNonnegative(config.amplitude, "amplitude");
  ThrowUnlessFinitePositive(config.period, "period");
  if (!(std::isfinite(config.duty_cycle) && config.duty_cycle > 0.0 &&
        config.duty_cycle < 1.0)) {
    throw std::logic_error("duty_cycle must be finite and in (0, 1)");
  }
  ThrowUnlessIntegerMultiple(config.sample_period, config.time_step,
                             "sample_period");
  ThrowUnlessIntegerMultiple(config.duration, config.sample_period, "duration");
}

class SquareForce final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SquareForce);

  SquareForce(double amplitude, double duty_cycle, double period)
      : amplitude_(amplitude), duty_cycle_(duty_cycle), period_(period) {
    this->DeclareVectorOutputPort("square_force", 2, &SquareForce::CalcOutput);
  }

 private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* output) const {
    const double cycle_time =
        context.get_time() - std::floor(context.get_time() / period_) * period_;
    const double value = cycle_time < duty_cycle_ * period_ ? amplitude_ : 0.0;
    output->get_mutable_value() = Eigen::Vector2d(value, -value);
  }

  const double amplitude_;
  const double duty_cycle_;
  const double period_;
};

GeometryId GetOnlyCollisionGeometry(const MultibodyPlant<double>& plant,
                                    std::string_view body_name,
                                    ModelInstanceIndex model_instance) {
  const RigidBody<double>& body =
      plant.GetBodyByName(std::string(body_name), model_instance);
  const std::vector<GeometryId>& ids =
      plant.GetCollisionGeometriesForBody(body);
  if (ids.size() != 1) {
    throw std::runtime_error(
        fmt::format("Expected one collision geometry on '{}'; got {}",
                    body_name, ids.size()));
  }
  return ids[0];
}

struct ContactGeometryIds {
  GeometryId left_finger;
  GeometryId right_finger;
  GeometryId spatula;
};

struct ResolutionData {
  double ellipsoid_base{};
  double cylinder_base{};
};

ResolutionData UpdateCompliantGeometryProperties(
    Representation representation, double resolution_scale,
    const std::array<ModelInstanceIndex, 2>& model_instances,
    MultibodyPlant<double>* plant, SceneGraph<double>* scene_graph) {
  if (plant == nullptr || scene_graph == nullptr ||
      !plant->get_source_id().has_value()) {
    throw std::logic_error("The parsed scene is missing its geometry source");
  }
  std::vector<double> ellipsoid_resolutions;
  std::vector<double> cylinder_resolutions;
  int updated = 0;
  for (ModelInstanceIndex model_instance : model_instances) {
    for (multibody::BodyIndex body_index :
         plant->GetBodyIndices(model_instance)) {
      const RigidBody<double>& body = plant->get_body(body_index);
      for (GeometryId geometry_id :
           plant->GetCollisionGeometriesForBody(body)) {
        const ProximityProperties* const old_properties =
            scene_graph->model_inspector().GetProximityProperties(geometry_id);
        if (old_properties == nullptr) continue;
        const auto compliance = old_properties->GetPropertyOrDefault(
            geometry::internal::kHydroGroup,
            geometry::internal::kComplianceType,
            geometry::internal::HydroelasticType::kUndefined);
        if (compliance != geometry::internal::HydroelasticType::kCompliant) {
          continue;
        }
        const double base_resolution = old_properties->GetProperty<double>(
            geometry::internal::kHydroGroup, geometry::internal::kRezHint);
        const double resolution = resolution_scale * base_resolution;
        ThrowUnlessFinitePositive(resolution, "scaled resolution hint");
        ProximityProperties properties(*old_properties);
        properties.UpdateProperty(geometry::internal::kHydroGroup,
                                  geometry::internal::kRezHint, resolution);
        if (representation != Representation::kTet) {
          properties.UpdateProperty(
              geometry::internal::kHydroGroup,
              geometry::internal::kCompliantRepresentation,
              std::string("voxel_sdf"));
          properties.UpdateProperty(
              geometry::internal::kHydroGroup,
              geometry::internal::kVoxelSdfEvaluationMode,
              geometry::VoxelSdfEvaluationMode::kPrimitiveSdf);
          properties.UpdateProperty(
              geometry::internal::kHydroGroup,
              geometry::internal::kVoxelSdfExtractionMethod,
              representation == Representation::kPlaneClip
                  ? geometry::VoxelSdfExtractionMethod::kPlaneClip
                  : geometry::VoxelSdfExtractionMethod::kMarchingCubes);
        }
        scene_graph->AssignRole(*plant->get_source_id(), geometry_id,
                                properties, geometry::RoleAssign::kReplace);
        if (body.name() == "spatula") {
          cylinder_resolutions.push_back(base_resolution);
        } else if (body.name() == "left_finger_bubble" ||
                   body.name() == "right_finger_bubble") {
          ellipsoid_resolutions.push_back(base_resolution);
        }
        ++updated;
      }
    }
  }
  if (updated != 3 || ellipsoid_resolutions.size() != 2 ||
      cylinder_resolutions.size() != 1 ||
      ellipsoid_resolutions[0] != ellipsoid_resolutions[1]) {
    throw std::runtime_error(
        "Expected two equal-resolution compliant Ellipsoids and one "
        "compliant Cylinder");
  }
  return {.ellipsoid_base = ellipsoid_resolutions[0],
          .cylinder_base = cylinder_resolutions[0]};
}

struct BuiltScene {
  std::unique_ptr<Diagram<double>> diagram;
  const MultibodyPlant<double>* plant{};
  const SceneGraph<double>* scene_graph{};
  const RigidBody<double>* spatula_body{};
  const PrismaticJoint<double>* left_joint{};
  const PrismaticJoint<double>* right_joint{};
  ContactGeometryIds geometry_ids;
  RigidTransformd X_BH;
  ResolutionData resolutions;
  bool expect_triangles{};
};

BuiltScene BuildScene(const SpatulaSlipConfig& config) {
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, config.time_step);
  plant.set_contact_model(ContactModel::kHydroelasticsOnly);
  plant.set_discrete_contact_approximation(
      DiscreteContactApproximation::kLagged);
  // This is the load-bearing three-way dispatch. Plane clipping is polygon
  // only and marching cubes is triangle only.
  plant.set_contact_surface_representation(
      SurfaceTypeFor(config.representation));
  plant.set_stiction_tolerance(config.stiction_tolerance);
  plant.SetUseSampledOutputPorts(false);

  multibody::Parser parser(&builder);
  const std::vector<ModelInstanceIndex> gripper_instances =
      parser.AddModelsFromUrl(
          "package://drake_models/wsg_50_description/sdf/"
          "schunk_wsg_50_hydro_bubble.sdf");
  const std::vector<ModelInstanceIndex> spatula_instances =
      parser.AddModelsFromUrl(
          "package://drake/examples/hydroelastic/spatula_slip_control/"
          "spatula.sdf");
  if (gripper_instances.size() != 1 || spatula_instances.size() != 1) {
    throw std::runtime_error("Expected one gripper and one spatula model");
  }
  const ModelInstanceIndex gripper_instance = gripper_instances[0];
  const ModelInstanceIndex spatula_instance = spatula_instances[0];
  const ContactGeometryIds geometry_ids{
      .left_finger = GetOnlyCollisionGeometry(plant, "left_finger_bubble",
                                              gripper_instance),
      .right_finger = GetOnlyCollisionGeometry(plant, "right_finger_bubble",
                                               gripper_instance),
      .spatula = GetOnlyCollisionGeometry(plant, "spatula", spatula_instance)};
  const RigidTransformd X_BH =
      scene_graph.model_inspector().GetPoseInFrame(geometry_ids.spatula);
  const ResolutionData resolutions = UpdateCompliantGeometryProperties(
      config.representation, config.resolution_scale,
      {gripper_instance, spatula_instance}, &plant, &scene_graph);

  const RigidTransformd X_WG(RollPitchYawd(0.0, -1.57, 0.0),
                             Vector3d(0.0, 0.0, 0.25));
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("gripper"), X_WG);
  plant.Finalize();

  const auto& square = *builder.AddSystem<SquareForce>(
      config.amplitude, config.duty_cycle, config.period);
  const auto& constant =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector2d(config.gripper_force, -config.gripper_force));
  const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
  builder.Connect(square.get_output_port(), adder.get_input_port(0));
  builder.Connect(constant.get_output_port(), adder.get_input_port(1));
  builder.Connect(adder.get_output_port(), plant.get_actuation_input_port());

  BuiltScene result;
  result.plant = &plant;
  result.scene_graph = &scene_graph;
  result.spatula_body = &plant.GetBodyByName("spatula", spatula_instance);
  result.left_joint = &plant.GetJointByName<PrismaticJoint>(
      "left_finger_sliding_joint", gripper_instance);
  result.right_joint = &plant.GetJointByName<PrismaticJoint>(
      "right_finger_sliding_joint", gripper_instance);
  result.geometry_ids = geometry_ids;
  result.X_BH = X_BH;
  result.resolutions = resolutions;
  result.expect_triangles =
      config.representation == Representation::kMarchingCubes;
  result.diagram = builder.Build();
  return result;
}

void ValidateSurface(const geometry::ContactSurface<double>& surface,
                     const multibody::HydroelasticContactInfo<double>& info,
                     bool expect_triangles) {
  if (surface.is_triangle() != expect_triangles || surface.num_faces() <= 0 ||
      surface.num_vertices() <= 0 ||
      !(std::isfinite(surface.total_area()) && surface.total_area() > 0.0) ||
      !surface.centroid().allFinite() ||
      !info.F_Ac_W().get_coeffs().allFinite()) {
    throw std::runtime_error("Hydroelastic surface or wrench is invalid");
  }
  bool has_positive_pressure = false;
  for (int vertex = 0; vertex < surface.num_vertices(); ++vertex) {
    const double pressure = surface.is_triangle()
                                ? surface.tri_e_MN().EvaluateAtVertex(vertex)
                                : surface.poly_e_MN().EvaluateAtVertex(vertex);
    if (!(std::isfinite(pressure) && pressure >= -kNegativePressureTolerance)) {
      throw std::runtime_error("Hydroelastic pressure is invalid");
    }
    has_positive_pressure = has_positive_pressure || pressure > 0.0;
  }
  if (!has_positive_pressure) {
    throw std::runtime_error("Hydroelastic surface has no positive pressure");
  }
}

SpatulaSlipRow SampleRow(const BuiltScene& scene,
                         const Context<double>& root_context) {
  const Context<double>& plant_context =
      scene.plant->GetMyContextFromRoot(root_context);
  SpatulaSlipRow row;
  row.time = root_context.get_time();
  const RigidTransformd X_WB =
      scene.spatula_body->EvalPoseInWorld(plant_context);
  row.position_WB = X_WB.translation();
  row.quaternion_WB = X_WB.rotation().ToQuaternion();
  const RigidTransformd X_WH = X_WB * scene.X_BH;
  row.handle_axis_W = X_WH.rotation().matrix() * Vector3d::UnitZ();
  row.left_finger_position = scene.left_joint->get_translation(plant_context);
  row.right_finger_position = scene.right_joint->get_translation(plant_context);

  const ContactResults<double>& contacts =
      scene.plant->get_contact_results_output_port()
          .Eval<ContactResults<double>>(plant_context);
  row.point_contacts = contacts.num_point_pair_contacts();
  row.hydro_contacts = contacts.num_hydroelastic_contacts();
  if (row.point_contacts != 0) {
    throw std::runtime_error(
        "Strict hydroelastic scene unexpectedly produced point contact");
  }
  bool left_seen = false;
  bool right_seen = false;
  for (int i = 0; i < row.hydro_contacts; ++i) {
    const auto& info = contacts.hydroelastic_contact_info(i);
    const auto& surface = info.contact_surface();
    ValidateSurface(surface, info, scene.expect_triangles);
    const bool spatula_is_M = surface.id_M() == scene.geometry_ids.spatula;
    const bool spatula_is_N = surface.id_N() == scene.geometry_ids.spatula;
    if (!spatula_is_M && !spatula_is_N) {
      const bool is_finger_pair =
          (surface.id_M() == scene.geometry_ids.left_finger &&
           surface.id_N() == scene.geometry_ids.right_finger) ||
          (surface.id_M() == scene.geometry_ids.right_finger &&
           surface.id_N() == scene.geometry_ids.left_finger);
      if (is_finger_pair) continue;
      throw std::runtime_error("Unexpected hydroelastic contact pair");
    }
    if (spatula_is_M && spatula_is_N) {
      throw std::runtime_error("Contact surface contains the spatula twice");
    }
    ++row.spatula_contacts;
    const GeometryId finger_id = spatula_is_M ? surface.id_N() : surface.id_M();
    bool* seen{};
    Vector3d* force{};
    double* torque{};
    double* area{};
    int* faces{};
    if (finger_id == scene.geometry_ids.left_finger) {
      seen = &left_seen;
      force = &row.left_contact_force_W;
      torque = &row.left_handle_axis_torque;
      area = &row.left_contact_area;
      faces = &row.left_surface_faces;
    } else if (finger_id == scene.geometry_ids.right_finger) {
      seen = &right_seen;
      force = &row.right_contact_force_W;
      torque = &row.right_handle_axis_torque;
      area = &row.right_contact_area;
      faces = &row.right_surface_faces;
    } else {
      throw std::runtime_error("Spatula contacted an unexpected geometry");
    }
    if (*seen) {
      throw std::runtime_error("Duplicate contact surface for one finger");
    }
    *seen = true;
    const SpatialForce<double> F_Sc_W =
        spatula_is_M ? info.F_Ac_W() : -info.F_Ac_W();
    const SpatialForce<double> F_Sh_W =
        F_Sc_W.Shift(X_WH.translation() - surface.centroid());
    *force = F_Sh_W.translational();
    *torque = F_Sh_W.rotational().dot(row.handle_axis_W);
    *area = surface.total_area();
    *faces = surface.num_faces();
  }

  if (!row.position_WB.allFinite() || !row.quaternion_WB.coeffs().allFinite() ||
      !row.handle_axis_W.allFinite() ||
      !std::isfinite(row.left_finger_position) ||
      !std::isfinite(row.right_finger_position) ||
      !row.left_contact_force_W.allFinite() ||
      !std::isfinite(row.left_handle_axis_torque) ||
      !std::isfinite(row.left_contact_area) ||
      !row.right_contact_force_W.allFinite() ||
      !std::isfinite(row.right_handle_axis_torque) ||
      !std::isfinite(row.right_contact_area)) {
    throw std::runtime_error("Trajectory sample contains a non-finite value");
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

void WriteCsv(const SpatulaSlipConfig& config,
              const SpatulaSlipResult& result) {
  if (config.output.empty()) return;
  if (config.output.has_parent_path()) {
    std::filesystem::create_directories(config.output.parent_path());
  }
  std::ofstream output(config.output);
  if (!output) {
    throw std::runtime_error("Unable to create trajectory CSV '" +
                             config.output.string() + "'");
  }
  const GitProvenance provenance = ReadGitProvenanceOrUnknown();
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << kCsvHeader << '\n';
  for (const SpatulaSlipRow& row : result.rows) {
    const Eigen::Quaterniond& q = row.quaternion_WB;
    output << "1," << provenance.commit << ','
           << (provenance.dirty ? "true" : "false") << ','
           << to_string(config.representation) << ',' << config.resolution_scale
           << ',' << result.ellipsoid_resolution << ','
           << result.cylinder_resolution << ',' << config.time_step << ','
           << config.sample_period << ',' << row.time << ','
           << row.position_WB.x() << ',' << row.position_WB.y() << ','
           << row.position_WB.z() << ',' << q.w() << ',' << q.x() << ','
           << q.y() << ',' << q.z() << ',' << row.handle_axis_W.x() << ','
           << row.handle_axis_W.y() << ',' << row.handle_axis_W.z() << ','
           << row.left_finger_position << ',' << row.right_finger_position
           << ',' << row.point_contacts << ',' << row.hydro_contacts << ','
           << row.spatula_contacts << ',' << row.left_contact_force_W.x() << ','
           << row.left_contact_force_W.y() << ','
           << row.left_contact_force_W.z() << ',' << row.left_handle_axis_torque
           << ',' << row.left_contact_area << ',' << row.left_surface_faces
           << ',' << row.right_contact_force_W.x() << ','
           << row.right_contact_force_W.y() << ','
           << row.right_contact_force_W.z() << ','
           << row.right_handle_axis_torque << ',' << row.right_contact_area
           << ',' << row.right_surface_faces << '\n';
  }
  if (!output) {
    throw std::runtime_error("Failed while writing trajectory CSV '" +
                             config.output.string() + "'");
  }
}

}  // namespace

SpatulaSlipResult RunSpatulaSlip(const SpatulaSlipConfig& config) {
  ValidateConfig(config);
  BuiltScene scene = BuildScene(config);
  SpatulaSlipResult result;
  result.ellipsoid_base_resolution = scene.resolutions.ellipsoid_base;
  result.cylinder_base_resolution = scene.resolutions.cylinder_base;
  result.ellipsoid_resolution =
      config.resolution_scale * result.ellipsoid_base_resolution;
  result.cylinder_resolution =
      config.resolution_scale * result.cylinder_base_resolution;

  Simulator<double> simulator(*scene.diagram);
  simulator.set_target_realtime_rate(0.0);
  Context<double>& root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      scene.plant->GetMyMutableContextFromRoot(&root_context);
  scene.plant->SetFreeBodyPose(&plant_context, *scene.spatula_body,
                               RigidTransformd(RollPitchYawd(-0.4, 0.0, 1.57),
                                               Vector3d(0.35, 0.0, 0.25)));
  scene.left_joint->set_translation(&plant_context, -0.01);
  scene.right_joint->set_translation(&plant_context, 0.01);
  simulator.Initialize();

  const int frames =
      static_cast<int>(std::llround(config.duration / config.sample_period));
  for (int frame = 0; frame <= frames; ++frame) {
    if (frame > 0) simulator.AdvanceTo(frame * config.sample_period);
    SpatulaSlipRow row = SampleRow(scene, simulator.get_context());
    if (row.spatula_contacts > 0) ++result.contact_samples;
    if (row.spatula_contacts == 2) ++result.two_finger_contact_samples;
    if (scene.expect_triangles) {
      result.triangle_surface_samples += row.spatula_contacts;
    }
    result.max_contact_force =
        std::max({result.max_contact_force, row.left_contact_force_W.norm(),
                  row.right_contact_force_W.norm()});
    result.max_abs_handle_axis_torque =
        std::max({result.max_abs_handle_axis_torque,
                  std::abs(row.left_handle_axis_torque),
                  std::abs(row.right_handle_axis_torque)});
    result.max_contact_area =
        std::max({result.max_contact_area, row.left_contact_area,
                  row.right_contact_area});
    result.rows.push_back(std::move(row));
  }

  const SpatulaSlipRow& initial = result.rows.front();
  for (const SpatulaSlipRow& row : result.rows) {
    result.max_position_displacement =
        std::max(result.max_position_displacement,
                 (row.position_WB - initial.position_WB).norm());
    const double cosine =
        std::clamp(initial.handle_axis_W.dot(row.handle_axis_W), -1.0, 1.0);
    result.max_handle_axis_swing =
        std::max(result.max_handle_axis_swing, std::acos(cosine));
  }
  WriteCsv(config, result);
  return result;
}

std::string_view SpatulaSlipCsvHeader() {
  return kCsvHeader;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
