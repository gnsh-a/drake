#include "drake/tools/voxel_sdf_experiments/frozen_spatula/frozen_spatula.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/geometry_properties.h"
#include "drake/geometry/kinematics_vector.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_object.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/tools/voxel_sdf_experiments/common/components.h"
#include "drake/tools/voxel_sdf_experiments/common/emit.h"
#include "drake/tools/voxel_sdf_experiments/common/fine_tet_reference.h"
#include "drake/tools/voxel_sdf_experiments/common/mesh_export.h"
#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Vector3d;
using geometry::ContactSurface;
using geometry::FramePoseVector;
using geometry::GeometryFrame;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::ProximityProperties;
using geometry::QueryObject;
using geometry::SceneGraph;
using geometry::Shape;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::PrismaticJoint;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;

constexpr double kNegativePressureTolerance = 1.0e-7;
constexpr double kFirstTouchIntervalTolerance = 1.0e-12;
constexpr double kInitialTouchSearchStep = 0.001;
constexpr double kMaxTouchSearchOffset = 1.0;
constexpr std::string_view kCsvHeader =
    "schema_version,git_commit,git_dirty,pose,requested_penetration_m,"
    "demo_directional_penetration_m,realized_directional_penetration_m,"
    "rigid_signed_distance_m,resolution_scale,ellipsoid_resolution_m,"
    "cylinder_resolution_m,fine_tet_resolution_hint_m,representation,"
    "reference_available,in_contact,is_triangle,num_faces,num_vertices,"
    "total_surface_area_m2,projected_area_m2,reference_projected_area_m2,"
    "area_relative_error,pressure_integral_n,force_norm_n,reference_force_n,"
    "force_relative_error,normal_force_n,transverse_force_n,min_pressure_pa,"
    "max_pressure_pa,surface_distance_rms_m,surface_distance_max_m,"
    "pressure_error_rms_pa,pressure_error_max_pa,centroid_x_m,centroid_y_m,"
    "centroid_z_m,reference_centroid_x_m,reference_centroid_y_m,"
    "reference_centroid_z_m,centroid_position_error_m,connected_components,"
    "largest_component_area_fraction,has_nonfinite,has_negative_pressure,"
    "reference_construction_wall_s";

struct LoadedSpatulaGeometry {
  std::unique_ptr<Shape> cylinder;
  std::unique_ptr<Shape> ellipsoid;
  RigidTransformd X_CE_demo;
  double cylinder_resolution{};
  double ellipsoid_resolution{};
  double cylinder_modulus{};
  double ellipsoid_modulus{};
};

void ThrowUnlessFiniteNonnegative(double value, std::string_view name) {
  if (!(std::isfinite(value) && value >= 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and nonnegative");
  }
}

void ThrowUnlessFinitePositive(double value, std::string_view name) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and strictly positive");
  }
}

void ValidateConfig(const FrozenSpatulaConfig& config) {
  ThrowUnlessFiniteNonnegative(config.penetration, "penetration");
  ThrowUnlessFinitePositive(config.resolution_scale, "resolution_scale");
  ThrowUnlessFinitePositive(config.fine_tet_resolution_hint,
                            "fine_tet_resolution_hint");
}

GeometryId GetOnlyCollisionGeometry(const MultibodyPlant<double>& plant,
                                    std::string_view body_name) {
  const auto& ids =
      plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name));
  if (ids.size() != 1) {
    throw std::runtime_error(
        fmt::format("Expected one collision geometry on '{}'; got {}",
                    body_name, ids.size()));
  }
  return ids[0];
}

std::pair<double, double> ReadHydroelasticParameters(
    const geometry::SceneGraphInspector<double>& inspector,
    GeometryId geometry_id) {
  const ProximityProperties* const properties =
      inspector.GetProximityProperties(geometry_id);
  if (properties == nullptr) {
    throw std::runtime_error("Parsed collision geometry has no proximity role");
  }
  return {
      properties->GetProperty<double>(geometry::internal::kHydroGroup,
                                      geometry::internal::kRezHint),
      properties->GetProperty<double>(geometry::internal::kHydroGroup,
                                      geometry::internal::kElastic),
  };
}

LoadedSpatulaGeometry LoadSpatulaGeometry() {
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);
  Parser parser(&builder);
  parser.AddModelsFromUrl(
      "package://drake_models/wsg_50_description/sdf/"
      "schunk_wsg_50_hydro_bubble.sdf");
  parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/spatula_slip_control/"
      "spatula.sdf");

  const RigidTransformd X_WG(RollPitchYawd(0.0, -1.57, 0.0),
                             Vector3d(0.0, 0.0, 0.25));
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("gripper"), X_WG);
  plant.Finalize();

  const GeometryId cylinder_id = GetOnlyCollisionGeometry(plant, "spatula");
  const GeometryId ellipsoid_id =
      GetOnlyCollisionGeometry(plant, "left_finger_bubble");
  std::unique_ptr<Diagram<double>> diagram = builder.Build();
  std::unique_ptr<Context<double>> root_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(root_context.get());
  const RigidTransformd X_WB(RollPitchYawd(-0.4, 0.0, 1.57),
                             Vector3d(0.35, 0.0, 0.25));
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("spatula"), X_WB);
  plant.GetJointByName<PrismaticJoint>("left_finger_sliding_joint")
      .set_translation(&plant_context, -0.01);
  plant.GetJointByName<PrismaticJoint>("right_finger_sliding_joint")
      .set_translation(&plant_context, 0.01);

  const Context<double>& scene_graph_context =
      scene_graph.GetMyContextFromRoot(*root_context);
  const QueryObject<double>& query =
      scene_graph.get_query_output_port().Eval<QueryObject<double>>(
          scene_graph_context);
  const auto& inspector = query.inspector();
  const auto [cylinder_resolution, cylinder_modulus] =
      ReadHydroelasticParameters(inspector, cylinder_id);
  const auto [ellipsoid_resolution, ellipsoid_modulus] =
      ReadHydroelasticParameters(inspector, ellipsoid_id);

  LoadedSpatulaGeometry result;
  result.cylinder = inspector.GetShape(cylinder_id).Clone();
  result.ellipsoid = inspector.GetShape(ellipsoid_id).Clone();
  result.X_CE_demo = query.GetPoseInWorld(cylinder_id).inverse() *
                     query.GetPoseInWorld(ellipsoid_id);
  result.cylinder_resolution = cylinder_resolution;
  result.ellipsoid_resolution = ellipsoid_resolution;
  result.cylinder_modulus = cylinder_modulus;
  result.ellipsoid_modulus = ellipsoid_modulus;
  return result;
}

Vector3d CalcApproachDirectionC(const RigidTransformd& X_CE_demo) {
  Vector3d radial = X_CE_demo.translation();
  radial.z() = 0.0;
  if (!(radial.allFinite() && radial.norm() > 0.0)) {
    throw std::runtime_error(
        "The demo ellipsoid center does not define an approach direction");
  }
  return radial.normalized();
}

class RigidPairDistanceQuery {
 public:
  explicit RigidPairDistanceQuery(const LoadedSpatulaGeometry& geometry) {
    source_id_ = scene_graph_.RegisterSource("frozen_spatula_distance");
    cylinder_frame_ =
        scene_graph_.RegisterFrame(source_id_, GeometryFrame("cylinder"));
    ellipsoid_frame_ =
        scene_graph_.RegisterFrame(source_id_, GeometryFrame("ellipsoid"));
    cylinder_id_ = scene_graph_.RegisterGeometry(
        source_id_, cylinder_frame_,
        std::make_unique<GeometryInstance>(RigidTransformd(),
                                           *geometry.cylinder, "cylinder"));
    ellipsoid_id_ = scene_graph_.RegisterGeometry(
        source_id_, ellipsoid_frame_,
        std::make_unique<GeometryInstance>(RigidTransformd(),
                                           *geometry.ellipsoid, "ellipsoid"));
    scene_graph_.AssignRole(source_id_, cylinder_id_, ProximityProperties());
    scene_graph_.AssignRole(source_id_, ellipsoid_id_, ProximityProperties());
    context_ = scene_graph_.CreateDefaultContext();
  }

  double CalcSignedDistance(const RigidTransformd& X_CE) {
    const FramePoseVector<double> poses{{cylinder_frame_, RigidTransformd()},
                                        {ellipsoid_frame_, X_CE}};
    scene_graph_.get_source_pose_port(source_id_)
        .FixValue(context_.get(), poses);
    const QueryObject<double>& query =
        scene_graph_.get_query_output_port().Eval<QueryObject<double>>(
            *context_);
    const double distance =
        query
            .ComputeSignedDistancePairClosestPoints(cylinder_id_, ellipsoid_id_)
            .distance;
    if (!std::isfinite(distance)) {
      throw std::runtime_error("Rigid signed-distance query was not finite");
    }
    return distance;
  }

 private:
  SceneGraph<double> scene_graph_;
  geometry::SourceId source_id_;
  geometry::FrameId cylinder_frame_;
  geometry::FrameId ellipsoid_frame_;
  GeometryId cylinder_id_;
  GeometryId ellipsoid_id_;
  std::unique_ptr<Context<double>> context_;
};

RigidTransformd FindFirstTouchPose(const RigidTransformd& X_CE_demo,
                                   const Vector3d& approach_direction_C,
                                   RigidPairDistanceQuery* distance_query) {
  if (distance_query->CalcSignedDistance(X_CE_demo) > 0.0) {
    throw std::runtime_error(
        "The demo pair is separated; cannot search outward for first touch");
  }
  auto pose_at_offset = [&](double offset) {
    return RigidTransformd(
        X_CE_demo.rotation(),
        X_CE_demo.translation() + offset * approach_direction_C);
  };
  double penetrating_offset = 0.0;
  double separated_offset = kInitialTouchSearchStep;
  double separated_distance =
      distance_query->CalcSignedDistance(pose_at_offset(separated_offset));
  while (separated_distance < 0.0 && separated_offset < kMaxTouchSearchOffset) {
    penetrating_offset = separated_offset;
    separated_offset *= 2.0;
    separated_distance =
        distance_query->CalcSignedDistance(pose_at_offset(separated_offset));
  }
  if (separated_distance < 0.0) {
    throw std::runtime_error("Failed to bracket first touch within one meter");
  }
  while (separated_offset - penetrating_offset > kFirstTouchIntervalTolerance) {
    const double midpoint = 0.5 * (penetrating_offset + separated_offset);
    if (distance_query->CalcSignedDistance(pose_at_offset(midpoint)) < 0.0) {
      penetrating_offset = midpoint;
    } else {
      separated_offset = midpoint;
    }
  }
  return pose_at_offset(separated_offset);
}

struct PoseData {
  RigidTransformd X_CE_first_touch;
  RigidTransformd X_CE;
  Vector3d approach_direction_C{Vector3d::Zero()};
  double demo_directional_penetration{};
  double realized_directional_penetration{};
  double rigid_signed_distance{};
};

PoseData MakePoseData(const LoadedSpatulaGeometry& geometry,
                      const FrozenSpatulaConfig& config) {
  PoseData result;
  result.approach_direction_C = CalcApproachDirectionC(geometry.X_CE_demo);
  RigidPairDistanceQuery distance_query(geometry);
  result.X_CE_first_touch = FindFirstTouchPose(
      geometry.X_CE_demo, result.approach_direction_C, &distance_query);
  result.demo_directional_penetration =
      (result.X_CE_first_touch.translation() - geometry.X_CE_demo.translation())
          .dot(result.approach_direction_C);
  const RigidTransformd& reference_pose = config.pose == SpatulaPose::kDemo
                                              ? geometry.X_CE_demo
                                              : result.X_CE_first_touch;
  result.X_CE =
      RigidTransformd(reference_pose.rotation(),
                      reference_pose.translation() -
                          config.penetration * result.approach_direction_C);
  result.realized_directional_penetration =
      config.penetration + (config.pose == SpatulaPose::kDemo
                                ? result.demo_directional_penetration
                                : 0.0);
  result.rigid_signed_distance = distance_query.CalcSignedDistance(result.X_CE);
  return result;
}

struct ComputedSurface {
  std::unique_ptr<ContactSurface<double>> surface;
  GeometryId cylinder_id;
};

ComputedSurface ComputeSurface(const LoadedSpatulaGeometry& geometry,
                               const PoseData& pose,
                               Representation representation,
                               double resolution_scale) {
  SceneGraph<double> scene_graph;
  const auto source_id = scene_graph.RegisterSource("frozen_spatula");
  const auto cylinder_frame =
      scene_graph.RegisterFrame(source_id, GeometryFrame("cylinder"));
  const auto ellipsoid_frame =
      scene_graph.RegisterFrame(source_id, GeometryFrame("ellipsoid"));
  const GeometryId cylinder_id = scene_graph.RegisterGeometry(
      source_id, cylinder_frame,
      std::make_unique<GeometryInstance>(RigidTransformd(), *geometry.cylinder,
                                         "spatula_handle_cylinder"));
  const GeometryId ellipsoid_id = scene_graph.RegisterGeometry(
      source_id, ellipsoid_frame,
      std::make_unique<GeometryInstance>(RigidTransformd(), *geometry.ellipsoid,
                                         "left_finger_ellipsoid"));
  scene_graph.AssignRole(
      source_id, cylinder_id,
      MakeProperties(representation,
                     resolution_scale * geometry.cylinder_resolution,
                     geometry.cylinder_modulus));
  scene_graph.AssignRole(
      source_id, ellipsoid_id,
      MakeProperties(representation,
                     resolution_scale * geometry.ellipsoid_resolution,
                     geometry.ellipsoid_modulus));

  std::unique_ptr<Context<double>> context = scene_graph.CreateDefaultContext();
  const FramePoseVector<double> poses{{cylinder_frame, RigidTransformd()},
                                      {ellipsoid_frame, pose.X_CE}};
  scene_graph.get_source_pose_port(source_id).FixValue(context.get(), poses);
  const QueryObject<double>& query =
      scene_graph.get_query_output_port().Eval<QueryObject<double>>(*context);
  std::vector<ContactSurface<double>> surfaces =
      query.ComputeContactSurfaces(SurfaceTypeFor(representation));
  if (surfaces.size() > 1) {
    throw std::runtime_error(
        fmt::format("Expected at most one cylinder-ellipsoid surface; got {}",
                    surfaces.size()));
  }
  ComputedSurface result{.cylinder_id = cylinder_id};
  if (surfaces.empty()) return result;
  const bool expect_triangle = IsMarchingCubes(representation);
  if (surfaces[0].is_triangle() != expect_triangle) {
    throw std::runtime_error(
        "Contact surface type did not match the representation");
  }
  result.surface =
      std::make_unique<ContactSurface<double>>(std::move(surfaces[0]));
  return result;
}

double RelativeError(double value, double reference) {
  if (!(std::isfinite(reference) && reference > 0.0)) {
    throw std::logic_error("Reference value must be finite and positive");
  }
  return std::abs(value - reference) / reference;
}

std::pair<int, double> CalcTopology(const SurfaceView& surface,
                                    double total_area) {
  const std::vector<int> component_ids =
      CalcFaceComponentIds(surface, DefaultComponentTolerance(surface));
  std::set<int> distinct(component_ids.begin(), component_ids.end());
  std::map<int, double> areas;
  for (int face = 0; face < ssize(surface.faces); ++face) {
    areas[component_ids[face]] += surface.faces[face].area;
  }
  double largest = 0.0;
  for (const auto& [component, area] : areas) {
    static_cast<void>(component);
    largest = std::max(largest, area);
  }
  return {distinct.size(), largest / total_area};
}

FrozenSpatulaMetrics CalcMetrics(const ComputedSurface& computed,
                                 const Vector3d& approach_direction_C,
                                 const FineTetReference* reference) {
  FrozenSpatulaMetrics result;
  if (computed.surface == nullptr) return result;
  result.in_contact = true;
  result.is_triangle = computed.surface->is_triangle();
  const SurfaceView surface = MakeSurfaceView(*computed.surface);
  result.num_vertices = surface.num_vertices;
  result.num_faces = surface.faces.size();
  result.min_pressure = std::numeric_limits<double>::infinity();
  result.max_pressure = -std::numeric_limits<double>::infinity();
  if (reference != nullptr) {
    result.surface_distance_max = 0.0;
    result.pressure_error_max = 0.0;
  }

  const bool cylinder_is_M = computed.surface->id_M() == computed.cylinder_id;
  if (!cylinder_is_M && computed.surface->id_N() != computed.cylinder_id) {
    throw std::runtime_error("Contact surface does not contain the cylinder");
  }
  Vector3d area_vector_C = Vector3d::Zero();
  Vector3d centroid_integral_C = Vector3d::Zero();
  double squared_surface_error_integral = 0.0;
  double squared_pressure_error_integral = 0.0;
  for (const Face& face : surface.faces) {
    const Vector3d normal_into_cylinder_C =
        cylinder_is_M ? face.normal_W : -face.normal_W;
    if (!(std::isfinite(face.area) && face.area > 0.0 &&
          std::isfinite(face.pressure) && face.centroid_W.allFinite() &&
          normal_into_cylinder_C.allFinite())) {
      result.has_nonfinite = true;
      continue;
    }
    result.has_negative_pressure |= face.pressure < -kNegativePressureTolerance;
    result.min_pressure = std::min(result.min_pressure, face.pressure);
    result.max_pressure = std::max(result.max_pressure, face.pressure);
    result.total_surface_area += face.area;
    area_vector_C += face.area * normal_into_cylinder_C;
    centroid_integral_C += face.area * face.centroid_W;
    const double pressure_integral = face.area * face.pressure;
    result.pressure_integral += pressure_integral;
    result.force_on_cylinder_C += pressure_integral * normal_into_cylinder_C;
    if (reference != nullptr) {
      const double surface_error =
          reference->distance_to_surface(face.centroid_W);
      squared_surface_error_integral +=
          face.area * surface_error * surface_error;
      result.surface_distance_max =
          std::max(result.surface_distance_max, surface_error);
      const double pressure_error =
          face.pressure - reference->pressure_at(face.centroid_W);
      squared_pressure_error_integral +=
          face.area * pressure_error * pressure_error;
      result.pressure_error_max =
          std::max(result.pressure_error_max, std::abs(pressure_error));
    }
  }
  if (!(result.total_surface_area > 0.0)) {
    throw std::runtime_error(
        "Contact surface has no valid positive-area faces");
  }
  result.projected_area = area_vector_C.norm();
  result.force_norm = result.force_on_cylinder_C.norm();
  result.normal_force = result.force_on_cylinder_C.dot(-approach_direction_C);
  result.transverse_force_norm =
      (result.force_on_cylinder_C + result.normal_force * approach_direction_C)
          .norm();
  result.centroid_C = centroid_integral_C / result.total_surface_area;
  std::tie(result.connected_components,
           result.largest_component_area_fraction) =
      CalcTopology(surface, result.total_surface_area);
  if (reference != nullptr) {
    result.reference_projected_area = reference->area();
    result.area_relative_error =
        RelativeError(result.projected_area, result.reference_projected_area);
    result.reference_force = reference->force();
    result.force_relative_error =
        RelativeError(result.force_norm, result.reference_force);
    result.surface_distance_rms =
        std::sqrt(squared_surface_error_integral / result.total_surface_area);
    result.pressure_error_rms =
        std::sqrt(squared_pressure_error_integral / result.total_surface_area);
    result.centroid_position_error =
        (result.centroid_C - reference->centroid()).norm();
  }
  return result;
}

std::unique_ptr<FineTetReference> ConstructFineReference(
    const LoadedSpatulaGeometry& geometry, const PoseData& pose,
    double resolution_hint, FineSpatulaReferenceValues* values) {
  values->resolution_hint = resolution_hint;
  // First touch is deliberately the separated side of the rigid-distance
  // bracket, where no positive-area fine reference exists.
  if (pose.rigid_signed_distance >= 0.0) return nullptr;
  const auto start = std::chrono::steady_clock::now();
  auto reference = std::make_unique<FineTetReference>(
      *geometry.cylinder, RigidTransformd(), geometry.cylinder_modulus,
      *geometry.ellipsoid, pose.X_CE, geometry.ellipsoid_modulus,
      resolution_hint);
  values->construction_wall_time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  values->available = true;
  values->force = reference->force();
  values->projected_area = reference->area();
  values->centroid_C = reference->centroid();
  return reference;
}

}  // namespace

SpatulaPose ParseSpatulaPose(std::string_view value) {
  if (value == "demo") return SpatulaPose::kDemo;
  if (value == "first_touch") return SpatulaPose::kFirstTouch;
  throw std::logic_error("Unknown pose '" + std::string(value) +
                         "'; expected demo or first_touch");
}

std::string_view to_string(SpatulaPose pose) {
  switch (pose) {
    case SpatulaPose::kDemo:
      return "demo";
    case SpatulaPose::kFirstTouch:
      return "first_touch";
  }
  throw std::logic_error("Invalid SpatulaPose value");
}

FrozenSpatulaResult RunFrozenSpatula(const FrozenSpatulaConfig& config) {
  ValidateConfig(config);
  const LoadedSpatulaGeometry geometry = LoadSpatulaGeometry();
  const PoseData pose = MakePoseData(geometry, config);

  FrozenSpatulaResult result;
  result.config = config;
  result.X_CE_demo = geometry.X_CE_demo;
  result.X_CE_first_touch = pose.X_CE_first_touch;
  result.X_CE = pose.X_CE;
  result.approach_direction_C = pose.approach_direction_C;
  result.demo_directional_penetration = pose.demo_directional_penetration;
  result.realized_directional_penetration =
      pose.realized_directional_penetration;
  result.rigid_signed_distance = pose.rigid_signed_distance;
  result.ellipsoid_base_resolution = geometry.ellipsoid_resolution;
  result.cylinder_base_resolution = geometry.cylinder_resolution;
  result.ellipsoid_resolution =
      config.resolution_scale * geometry.ellipsoid_resolution;
  result.cylinder_resolution =
      config.resolution_scale * geometry.cylinder_resolution;
  result.ellipsoid_modulus = geometry.ellipsoid_modulus;
  result.cylinder_modulus = geometry.cylinder_modulus;
  std::unique_ptr<FineTetReference> reference = ConstructFineReference(
      geometry, pose, config.fine_tet_resolution_hint, &result.reference);

  const std::array<Representation, 3> representations{
      Representation::kTet, Representation::kPlaneClip,
      Representation::kMarchingCubes};
  for (int i = 0; i < ssize(representations); ++i) {
    const Representation representation = representations[i];
    const ComputedSurface computed =
        ComputeSurface(geometry, pose, representation, config.resolution_scale);
    result.representations[i].representation = representation;
    result.representations[i].metrics =
        CalcMetrics(computed, pose.approach_direction_C, reference.get());
    if (!config.mesh_output_dir.empty() && computed.surface != nullptr) {
      WriteSurfaceVtk(
          config.mesh_output_dir /
              (std::string(to_string(representation)) + ".vtk"),
          MakeSurfaceView(*computed.surface),
          fmt::format("frozen spatula {} scale={}", to_string(representation),
                      config.resolution_scale));
    }
  }
  return result;
}

FineSpatulaReferenceValues MeasureFrozenSpatulaReference(
    const FrozenSpatulaConfig& config, double fine_tet_resolution_hint) {
  ValidateConfig(config);
  ThrowUnlessFinitePositive(fine_tet_resolution_hint,
                            "fine_tet_resolution_hint");
  const LoadedSpatulaGeometry geometry = LoadSpatulaGeometry();
  const PoseData pose = MakePoseData(geometry, config);
  FineSpatulaReferenceValues values;
  const std::unique_ptr<FineTetReference> reference =
      ConstructFineReference(geometry, pose, fine_tet_resolution_hint, &values);
  static_cast<void>(reference);
  return values;
}

std::string_view FrozenSpatulaCsvHeader() {
  return kCsvHeader;
}

void EmitFrozenSpatulaCsv(const std::filesystem::path& path,
                          const FrozenSpatulaResult& result) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to create CSV file '" + path.string() +
                             "'");
  }
  const GitProvenance provenance = ReadGitProvenance();
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << kCsvHeader << '\n';
  for (const FrozenSpatulaRepresentationResult& representation :
       result.representations) {
    const FrozenSpatulaMetrics& metrics = representation.metrics;
    output << "1," << provenance.commit << ','
           << (provenance.dirty ? "true" : "false") << ','
           << to_string(result.config.pose) << ',' << result.config.penetration
           << ',' << result.demo_directional_penetration << ','
           << result.realized_directional_penetration << ','
           << result.rigid_signed_distance << ','
           << result.config.resolution_scale << ','
           << result.ellipsoid_resolution << ',' << result.cylinder_resolution
           << ',' << result.reference.resolution_hint << ','
           << to_string(representation.representation) << ','
           << (result.reference.available ? "true" : "false") << ','
           << (metrics.in_contact ? "true" : "false") << ','
           << (metrics.is_triangle ? "true" : "false") << ','
           << metrics.num_faces << ',' << metrics.num_vertices << ','
           << metrics.total_surface_area << ',' << metrics.projected_area << ','
           << metrics.reference_projected_area << ','
           << metrics.area_relative_error << ',' << metrics.pressure_integral
           << ',' << metrics.force_norm << ',' << metrics.reference_force << ','
           << metrics.force_relative_error << ',' << metrics.normal_force << ','
           << metrics.transverse_force_norm << ',' << metrics.min_pressure
           << ',' << metrics.max_pressure << ',' << metrics.surface_distance_rms
           << ',' << metrics.surface_distance_max << ','
           << metrics.pressure_error_rms << ',' << metrics.pressure_error_max
           << ',' << metrics.centroid_C.x() << ',' << metrics.centroid_C.y()
           << ',' << metrics.centroid_C.z() << ','
           << result.reference.centroid_C.x() << ','
           << result.reference.centroid_C.y() << ','
           << result.reference.centroid_C.z() << ','
           << metrics.centroid_position_error << ','
           << metrics.connected_components << ','
           << metrics.largest_component_area_fraction << ','
           << (metrics.has_nonfinite ? "true" : "false") << ','
           << (metrics.has_negative_pressure ? "true" : "false") << ','
           << result.reference.construction_wall_time << '\n';
  }
  if (!output) {
    throw std::runtime_error("Failed while writing CSV file '" + path.string() +
                             "'");
  }
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
