#include "drake/tools/voxel_sdf_experiments/common/fine_tet_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/kinematics_vector.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/query_object.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/systems/framework/context.h"
#include "drake/tools/voxel_sdf_experiments/common/representation.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using geometry::Aabb;
using geometry::ContactSurface;
using geometry::FramePoseVector;
using geometry::GeometryFrame;
using geometry::GeometryInstance;
using geometry::QueryObject;
using geometry::SceneGraph;
using geometry::SurfaceTriangle;
using geometry::TriangleSurfaceMesh;
using geometry::internal::Bvh;
using systems::Context;

void ThrowUnlessFinitePositive(double value, std::string_view name) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and strictly positive");
  }
}

void ThrowUnlessFinite(const Eigen::Vector3d& value, std::string_view name) {
  if (!value.allFinite()) {
    throw std::logic_error(std::string(name) + " must be finite");
  }
}

ContactSurface<double> ComputeFineTetSurface(const geometry::Shape& shape_a,
                                             const math::RigidTransformd& X_RA,
                                             double modulus_a,
                                             const geometry::Shape& shape_b,
                                             const math::RigidTransformd& X_RB,
                                             double modulus_b,
                                             double tet_resolution_hint) {
  ThrowUnlessFinitePositive(modulus_a, "modulus_a");
  ThrowUnlessFinitePositive(modulus_b, "modulus_b");
  ThrowUnlessFinitePositive(tet_resolution_hint, "tet_resolution_hint");
  if (!X_RA.GetAsMatrix4().allFinite() || !X_RB.GetAsMatrix4().allFinite()) {
    throw std::logic_error("Fine-tet geometry poses must be finite");
  }

  SceneGraph<double> scene_graph;
  const auto source_id = scene_graph.RegisterSource("fine_tet_reference");
  const auto frame_a =
      scene_graph.RegisterFrame(source_id, GeometryFrame("fine_tet_frame_a"));
  const auto frame_b =
      scene_graph.RegisterFrame(source_id, GeometryFrame("fine_tet_frame_b"));
  const auto geometry_a = scene_graph.RegisterGeometry(
      source_id, frame_a,
      std::make_unique<GeometryInstance>(math::RigidTransformd(), shape_a,
                                         "fine_tet_geometry_a"));
  const auto geometry_b = scene_graph.RegisterGeometry(
      source_id, frame_b,
      std::make_unique<GeometryInstance>(math::RigidTransformd(), shape_b,
                                         "fine_tet_geometry_b"));
  scene_graph.AssignRole(
      source_id, geometry_a,
      MakeProperties(Representation::kTet, tet_resolution_hint, modulus_a));
  scene_graph.AssignRole(
      source_id, geometry_b,
      MakeProperties(Representation::kTet, tet_resolution_hint, modulus_b));

  std::unique_ptr<Context<double>> context = scene_graph.CreateDefaultContext();
  const FramePoseVector<double> poses{{frame_a, X_RA}, {frame_b, X_RB}};
  scene_graph.get_source_pose_port(source_id).FixValue(context.get(), poses);
  const QueryObject<double>& query =
      scene_graph.get_query_output_port().Eval<QueryObject<double>>(*context);
  std::vector<ContactSurface<double>> surfaces = query.ComputeContactSurfaces(
      geometry::HydroelasticContactRepresentation::kPolygon);
  if (surfaces.size() != 1) {
    throw std::runtime_error(
        "Expected exactly one fine-tet contact surface; got " +
        std::to_string(surfaces.size()));
  }
  if (surfaces[0].is_triangle()) {
    throw std::runtime_error("Fine-tet contact surface was not polygonal");
  }
  return std::move(surfaces[0]);
}

Eigen::Vector3d ClosestPointOnSegment(const Eigen::Vector3d& p_RQ,
                                      const Eigen::Vector3d& p_RA,
                                      const Eigen::Vector3d& p_RB) {
  const Eigen::Vector3d p_AB = p_RB - p_RA;
  const double length_squared = p_AB.squaredNorm();
  if (length_squared == 0.0) return p_RA;
  const double fraction =
      std::clamp((p_RQ - p_RA).dot(p_AB) / length_squared, 0.0, 1.0);
  return p_RA + fraction * p_AB;
}

Eigen::Vector3d ClosestPointOnTriangle(
    const Eigen::Vector3d& p_RQ, const TriangleSurfaceMesh<double>& mesh_R,
    int triangle_index) {
  const auto barycentric = mesh_R.CalcBarycentric(p_RQ, triangle_index);
  if ((barycentric.array() >= 0.0).all()) {
    return mesh_R.CalcCartesianFromBarycentric(triangle_index, barycentric);
  }

  const auto& triangle = mesh_R.element(triangle_index);
  const Eigen::Vector3d& p_RA = mesh_R.vertex(triangle.vertex(0));
  const Eigen::Vector3d& p_RB = mesh_R.vertex(triangle.vertex(1));
  const Eigen::Vector3d& p_RC = mesh_R.vertex(triangle.vertex(2));
  const Eigen::Vector3d on_ab = ClosestPointOnSegment(p_RQ, p_RA, p_RB);
  const Eigen::Vector3d on_bc = ClosestPointOnSegment(p_RQ, p_RB, p_RC);
  const Eigen::Vector3d on_ca = ClosestPointOnSegment(p_RQ, p_RC, p_RA);
  const double ab_squared = (p_RQ - on_ab).squaredNorm();
  const double bc_squared = (p_RQ - on_bc).squaredNorm();
  const double ca_squared = (p_RQ - on_ca).squaredNorm();
  if (ab_squared <= bc_squared && ab_squared <= ca_squared) return on_ab;
  return bc_squared <= ca_squared ? on_bc : on_ca;
}

}  // namespace

class FineTetReference::Impl {
 public:
  explicit Impl(ContactSurface<double> surface) : surface_(std::move(surface)) {
    const auto& polygon_mesh_R = surface_.poly_mesh_W();
    std::vector<Eigen::Vector3d> vertices_R;
    vertices_R.reserve(polygon_mesh_R.num_vertices());
    for (int vertex = 0; vertex < polygon_mesh_R.num_vertices(); ++vertex) {
      vertices_R.push_back(polygon_mesh_R.vertex(vertex));
    }

    std::vector<SurfaceTriangle> triangles;
    triangles.reserve(2 * polygon_mesh_R.num_vertices());
    for (int face_index = 0; face_index < polygon_mesh_R.num_faces();
         ++face_index) {
      const auto polygon = polygon_mesh_R.element(face_index);
      const int first = polygon.vertex(0);
      int previous = polygon.vertex(1);
      for (int vertex = 2; vertex < polygon.num_vertices(); ++vertex) {
        const int current = polygon.vertex(vertex);
        const Eigen::Vector3d edge_a =
            polygon_mesh_R.vertex(previous) - polygon_mesh_R.vertex(first);
        const Eigen::Vector3d edge_b =
            polygon_mesh_R.vertex(current) - polygon_mesh_R.vertex(first);
        if (edge_a.cross(edge_b).squaredNorm() > 0.0) {
          triangles.emplace_back(first, previous, current);
          triangle_to_polygon_.push_back(face_index);
        }
        previous = current;
      }
    }
    if (triangles.empty()) {
      throw std::runtime_error("Fine-tet contact surface has no triangles");
    }
    triangle_mesh_R_ = std::make_unique<TriangleSurfaceMesh<double>>(
        std::move(triangles), std::move(vertices_R));
    bvh_R_ = std::make_unique<Bvh<Aabb, TriangleSurfaceMesh<double>>>(
        *triangle_mesh_R_);

    Eigen::Vector3d force_R = Eigen::Vector3d::Zero();
    Eigen::Vector3d area_vector_R = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid_integral_R = Eigen::Vector3d::Zero();
    double total_surface_area = 0.0;
    const auto& pressure = surface_.poly_e_MN();
    for (int face_index = 0; face_index < polygon_mesh_R.num_faces();
         ++face_index) {
      const double face_area = polygon_mesh_R.area(face_index);
      if (!(std::isfinite(face_area) && face_area > 0.0)) continue;
      const Eigen::Vector3d& face_normal_R =
          polygon_mesh_R.face_normal(face_index);
      const Eigen::Vector3d& face_centroid_R =
          polygon_mesh_R.element_centroid(face_index);
      const double face_pressure =
          pressure.EvaluateCartesian(face_index, face_centroid_R);
      if (!(std::isfinite(face_pressure) && face_pressure >= 0.0) ||
          !face_normal_R.allFinite() || !face_centroid_R.allFinite()) {
        throw std::runtime_error("Fine-tet contact surface data is invalid");
      }
      force_R += face_area * face_pressure * face_normal_R;
      area_vector_R += face_area * face_normal_R;
      centroid_integral_R += face_area * face_centroid_R;
      total_surface_area += face_area;
    }
    if (!(total_surface_area > 0.0)) {
      throw std::runtime_error("Fine-tet contact surface has zero area");
    }
    force_ = force_R.norm();
    // Reference::area() is projected area. The norm of the integrated area
    // vector preserves that meaning without inventing one representative
    // normal for a curved or multi-lobed contact patch.
    area_ = area_vector_R.norm();
    centroid_R_ = centroid_integral_R / total_surface_area;
    if (!(std::isfinite(force_) && force_ > 0.0 && std::isfinite(area_) &&
          area_ > 0.0 && centroid_R_.allFinite())) {
      throw std::runtime_error("Fine-tet integrated reference is invalid");
    }
  }

  struct NearestPoint {
    double squared_distance{std::numeric_limits<double>::infinity()};
    Eigen::Vector3d p_RN{Eigen::Vector3d::Zero()};
    int triangle_index{-1};
  };

  NearestPoint FindNearestPoint(const Eigen::Vector3d& p_RQ) const {
    ThrowUnlessFinite(p_RQ, "p_RQ");
    NearestPoint result;
    VisitNode(p_RQ, bvh_R_->root_node(), &result);
    if (result.triangle_index < 0) {
      throw std::runtime_error("Fine-tet BVH found no nearest face");
    }
    return result;
  }

  double force() const { return force_; }
  double area() const { return area_; }
  const Eigen::Vector3d& centroid() const { return centroid_R_; }

  double pressure_at(const NearestPoint& nearest) const {
    const int polygon_index = triangle_to_polygon_.at(nearest.triangle_index);
    const double pressure =
        surface_.poly_e_MN().EvaluateCartesian(polygon_index, nearest.p_RN);
    if (!std::isfinite(pressure)) {
      throw std::runtime_error("Interpolated fine-tet pressure is not finite");
    }
    return std::max(0.0, pressure);
  }

 private:
  using Node = Bvh<Aabb, TriangleSurfaceMesh<double>>::NodeType;

  static double SquaredDistanceLowerBound(const Eigen::Vector3d& p_RQ,
                                          const Aabb& box_R) {
    const Eigen::Array3d excess =
        ((p_RQ - box_R.center()).array().abs() - box_R.half_width().array())
            .max(0.0);
    return excess.matrix().squaredNorm();
  }

  void VisitNode(const Eigen::Vector3d& p_RQ, const Node& node_R,
                 NearestPoint* result) const {
    if (SquaredDistanceLowerBound(p_RQ, node_R.bv()) >
        result->squared_distance) {
      return;
    }
    if (node_R.is_leaf()) {
      for (int i = 0; i < node_R.num_element_indices(); ++i) {
        const int triangle_index = node_R.element_index(i);
        const Eigen::Vector3d p_RN =
            ClosestPointOnTriangle(p_RQ, *triangle_mesh_R_, triangle_index);
        const double squared_distance = (p_RQ - p_RN).squaredNorm();
        if (squared_distance < result->squared_distance) {
          result->squared_distance = squared_distance;
          result->p_RN = p_RN;
          result->triangle_index = triangle_index;
        }
      }
      return;
    }

    const Node* first = &node_R.left();
    const Node* second = &node_R.right();
    if (SquaredDistanceLowerBound(p_RQ, first->bv()) >
        SquaredDistanceLowerBound(p_RQ, second->bv())) {
      std::swap(first, second);
    }
    VisitNode(p_RQ, *first, result);
    VisitNode(p_RQ, *second, result);
  }

  ContactSurface<double> surface_;
  std::unique_ptr<TriangleSurfaceMesh<double>> triangle_mesh_R_;
  std::vector<int> triangle_to_polygon_;
  std::unique_ptr<Bvh<Aabb, TriangleSurfaceMesh<double>>> bvh_R_;
  double force_{};
  double area_{};
  Eigen::Vector3d centroid_R_{Eigen::Vector3d::Zero()};
};

FineTetReference::FineTetReference(const geometry::Shape& shape_a,
                                   const math::RigidTransformd& X_RA,
                                   double modulus_a,
                                   const geometry::Shape& shape_b,
                                   const math::RigidTransformd& X_RB,
                                   double modulus_b, double tet_resolution_hint)
    : impl_(std::make_unique<Impl>(
          ComputeFineTetSurface(shape_a, X_RA, modulus_a, shape_b, X_RB,
                                modulus_b, tet_resolution_hint))) {}

FineTetReference::FineTetReference(ContactSurface<double> surface)
    : impl_(std::make_unique<Impl>(std::move(surface))) {}

FineTetReference::~FineTetReference() = default;

double FineTetReference::force() const {
  return impl_->force();
}

double FineTetReference::area() const {
  return impl_->area();
}

double FineTetReference::distance_to_surface(
    const Eigen::Vector3d& p_RQ) const {
  return std::sqrt(impl_->FindNearestPoint(p_RQ).squared_distance);
}

double FineTetReference::pressure_at(const Eigen::Vector3d& p_RQ) const {
  return impl_->pressure_at(impl_->FindNearestPoint(p_RQ));
}

double FineTetReference::patch_radius() const {
  // A general fine-tet contact patch is not a disc, so it has no radius.
  return std::numeric_limits<double>::quiet_NaN();
}

double FineTetReference::peak_pressure() const {
  // A general fine-tet patch can have multiple incomparable pressure peaks.
  return std::numeric_limits<double>::quiet_NaN();
}

Eigen::Vector3d FineTetReference::centroid() const {
  return impl_->centroid();
}

Eigen::Vector3d FineTetReference::normal() const {
  // A curved or multi-lobed fine-tet patch has no single representative normal.
  return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
