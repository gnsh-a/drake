#include "drake/tools/voxel_sdf_experiments/frozen_surface/frozen_surface.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/kinematics_vector.h"
#include "drake/geometry/query_object.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/systems/framework/context.h"
#include "drake/tools/voxel_sdf_experiments/common/mesh_export.h"
#include "drake/tools/voxel_sdf_experiments/common/reference.h"
#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using geometry::Box;
using geometry::ContactSurface;
using geometry::FramePoseVector;
using geometry::GeometryFrame;
using geometry::GeometryInstance;
using geometry::QueryObject;
using geometry::SceneGraph;
using geometry::Sphere;
using math::RigidTransformd;
using systems::Context;

constexpr double kDegreesToRadians =
    3.141592653589793238462643383279502884 / 180.0;

void ThrowUnlessFinitePositive(double value, std::string_view name) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and strictly positive");
  }
}

void ValidateConfig(const FrozenSurfaceConfig& config) {
  ThrowUnlessFinitePositive(config.penetration, "penetration");
  ThrowUnlessFinitePositive(config.voxel_width, "voxel_width");
  ThrowUnlessFinitePositive(config.tet_resolution_hint, "tet_resolution_hint");
  ThrowUnlessFinitePositive(config.radius, "radius");
  ThrowUnlessFinitePositive(config.modulus_a, "modulus_a");
  ThrowUnlessFinitePositive(config.modulus_b, "modulus_b");
  if (!(config.penetration < 2.0 * config.radius)) {
    throw std::logic_error("penetration must be smaller than the diameter");
  }
  if (!config.grid_rpy_deg.allFinite()) {
    throw std::logic_error("grid RPY angles must be finite");
  }
  if (config.scene == Scene::kSphereBox) {
    for (int axis = 0; axis < 3; ++axis) {
      ThrowUnlessFinitePositive(config.box_half_widths[axis], "box half-width");
    }
    constexpr double kRelativeTolerance = 1e-14;
    if (std::abs(config.box_half_widths.z() - config.radius) >
            kRelativeTolerance * config.radius ||
        config.box_half_widths.x() < config.radius ||
        config.box_half_widths.y() < config.radius) {
      throw std::logic_error(
          "AnalyticParaboloid requires the box's smallest half-width to equal "
          "the sphere radius, with that half-width on z");
    }
  }
}

ContactSurface<double> ComputeSurface(const FrozenSurfaceConfig& config) {
  SceneGraph<double> scene_graph;
  const auto source_id = scene_graph.RegisterSource("frozen_surface");
  const auto frame_a =
      scene_graph.RegisterFrame(source_id, GeometryFrame("geometry_a"));
  const auto frame_b =
      scene_graph.RegisterFrame(source_id, GeometryFrame("geometry_b"));

  std::unique_ptr<geometry::Shape> shape_a;
  std::unique_ptr<geometry::Shape> shape_b;
  if (config.scene == Scene::kSphereSphere) {
    shape_a = std::make_unique<Sphere>(config.radius);
    shape_b = std::make_unique<Sphere>(config.radius);
  } else {
    shape_a = std::make_unique<Sphere>(config.radius);
    shape_b = std::make_unique<Box>(2.0 * config.box_half_widths.x(),
                                    2.0 * config.box_half_widths.y(),
                                    2.0 * config.box_half_widths.z());
  }
  const auto geometry_a = scene_graph.RegisterGeometry(
      source_id, frame_a,
      std::make_unique<GeometryInstance>(RigidTransformd(), std::move(shape_a),
                                         "geometry_a"));
  const auto geometry_b = scene_graph.RegisterGeometry(
      source_id, frame_b,
      std::make_unique<GeometryInstance>(RigidTransformd(), std::move(shape_b),
                                         "geometry_b"));
  const double resolution = config.representation == Representation::kTet
                                ? config.tet_resolution_hint
                                : config.voxel_width;
  scene_graph.AssignRole(
      source_id, geometry_a,
      MakeProperties(config.representation, resolution, config.modulus_a));
  scene_graph.AssignRole(
      source_id, geometry_b,
      MakeProperties(config.representation, resolution, config.modulus_b));

  std::unique_ptr<Context<double>> context = scene_graph.CreateDefaultContext();
  const math::RollPitchYawd grid_rpy(
      kDegreesToRadians * config.grid_rpy_deg.x(),
      kDegreesToRadians * config.grid_rpy_deg.y(),
      kDegreesToRadians * config.grid_rpy_deg.z());
  const double center_height = 2.0 * config.radius - config.penetration;
  RigidTransformd X_WA;
  RigidTransformd X_WB;
  if (config.scene == Scene::kSphereSphere) {
    X_WA = RigidTransformd(grid_rpy, Eigen::Vector3d::Zero());
    X_WB = RigidTransformd(Eigen::Vector3d(0.0, 0.0, center_height));
  } else {
    X_WA = RigidTransformd(grid_rpy, Eigen::Vector3d(0.0, 0.0, center_height));
    X_WB = RigidTransformd();
  }
  const FramePoseVector<double> poses{{frame_a, X_WA}, {frame_b, X_WB}};
  scene_graph.get_source_pose_port(source_id).FixValue(context.get(), poses);

  const QueryObject<double>& query =
      scene_graph.get_query_output_port().Eval<QueryObject<double>>(*context);
  std::vector<ContactSurface<double>> surfaces =
      query.ComputeContactSurfaces(SurfaceTypeFor(config.representation));
  if (surfaces.size() != 1) {
    throw std::runtime_error("Expected exactly one contact surface; got " +
                             std::to_string(surfaces.size()));
  }
  const bool expect_triangle =
      config.representation == Representation::kMarchingCubes;
  if (surfaces[0].is_triangle() != expect_triangle) {
    throw std::runtime_error("Contact surface type did not match extraction");
  }
  return std::move(surfaces[0]);
}

std::unique_ptr<Reference> MakeReference(const FrozenSurfaceConfig& config) {
  if (config.scene == Scene::kSphereSphere) {
    return std::make_unique<AnalyticPlane>(config.radius, config.penetration,
                                           config.modulus_a, config.modulus_b);
  }
  return std::make_unique<AnalyticParaboloid>(
      config.radius, config.penetration, config.modulus_a, config.modulus_b);
}

}  // namespace

Scene ParseScene(std::string_view value) {
  if (value == "sphere_sphere") return Scene::kSphereSphere;
  if (value == "sphere_box") return Scene::kSphereBox;
  throw std::logic_error("Unknown scene '" + std::string(value) +
                         "'; expected sphere_sphere or sphere_box");
}

std::string_view to_string(Scene scene) {
  switch (scene) {
    case Scene::kSphereSphere:
      return "sphere_sphere";
    case Scene::kSphereBox:
      return "sphere_box";
  }
  throw std::logic_error("Invalid Scene value");
}

Metrics RunOne(const FrozenSurfaceConfig& config) {
  ValidateConfig(config);
  const ContactSurface<double> surface = ComputeSurface(config);
  const std::unique_ptr<Reference> reference = MakeReference(config);
  const SurfaceView view = MakeSurfaceView(surface);
  if (!config.mesh_output.empty()) {
    WriteSurfaceVtk(
        config.mesh_output, view,
        fmt::format("{} {} h={}", to_string(config.scene),
                    to_string(config.representation), config.voxel_width));
  }
  return CalcMetrics(view, *reference);
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
