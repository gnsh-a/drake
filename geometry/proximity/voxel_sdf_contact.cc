#include "drake/geometry/proximity/voxel_sdf_contact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"
#include "drake/geometry/proximity/mesh_intersection.h"
#include "drake/geometry/proximity/posed_half_space.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

constexpr double kToleranceScale = 64.0;
constexpr double kMinimumPolygonArea = 1e-14;

bool IsFinite(const AffineSdfField& field) {
  return std::isfinite(field.value) && field.gradient.allFinite() &&
         std::isfinite(field.pressure_scale) &&
         std::isfinite(field.characteristic_length);
}

void RemoveNearDuplicates(double tolerance, std::vector<Vector3d>* vertices) {
  DRAKE_DEMAND(vertices != nullptr);
  const double tolerance_squared = tolerance * tolerance;
  std::vector<Vector3d> unique;
  unique.reserve(vertices->size());
  for (const Vector3d& candidate : *vertices) {
    const auto is_near = [&candidate, tolerance_squared](const Vector3d& p) {
      return (candidate - p).squaredNorm() <= tolerance_squared;
    };
    if (std::none_of(unique.begin(), unique.end(), is_near)) {
      unique.push_back(candidate);
    }
  }
  *vertices = std::move(unique);
}

Vector3d CalcCentroid(const std::vector<Vector3d>& vertices) {
  DRAKE_DEMAND(!vertices.empty());
  Vector3d centroid = Vector3d::Zero();
  for (const Vector3d& vertex : vertices) {
    centroid += vertex;
  }
  return centroid / static_cast<double>(vertices.size());
}

double CalcSignedArea(const std::vector<Vector3d>& vertices,
                      const Vector3d& normal) {
  if (vertices.size() < 3) return 0.0;
  const Vector3d centroid = CalcCentroid(vertices);
  double twice_area = 0.0;
  for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
    const Vector3d a = vertices[i] - centroid;
    const Vector3d b = vertices[(i + 1) % vertices.size()] - centroid;
    twice_area += normal.dot(a.cross(b));
  }
  return 0.5 * twice_area;
}

void SortCounterClockwise(const Vector3d& normal,
                          std::vector<Vector3d>* vertices) {
  DRAKE_DEMAND(vertices != nullptr);
  DRAKE_DEMAND(vertices->size() >= 3);
  const Vector3d centroid = CalcCentroid(*vertices);

  Eigen::Index least_aligned_axis{};
  normal.cwiseAbs().minCoeff(&least_aligned_axis);
  Vector3d axis = Vector3d::Zero();
  axis[least_aligned_axis] = 1.0;
  const Vector3d u = normal.cross(axis).normalized();
  const Vector3d v = normal.cross(u);

  const auto angle = [&centroid, &u, &v](const Vector3d& p) {
    const Vector3d offset = p - centroid;
    return std::atan2(offset.dot(v), offset.dot(u));
  };
  std::sort(vertices->begin(), vertices->end(),
            [&angle, &centroid](const Vector3d& a, const Vector3d& b) {
              const double angle_a = angle(a);
              const double angle_b = angle(b);
              if (angle_a != angle_b) return angle_a < angle_b;
              return (a - centroid).squaredNorm() <
                     (b - centroid).squaredNorm();
            });
  if (CalcSignedArea(*vertices, normal) < 0.0) {
    std::reverse(vertices->begin(), vertices->end());
  }
}

}  // namespace

std::optional<VoxelSdfContactPolygon> CalcVoxelSdfContactPolygon(
    const Vector3d& center_A, double voxel_width, const AffineSdfField& sdf_A,
    const AffineSdfField& sdf_B_A) {
  DRAKE_DEMAND(center_A.allFinite());
  DRAKE_DEMAND(std::isfinite(voxel_width) && voxel_width > 0.0);
  DRAKE_DEMAND(IsFinite(sdf_A));
  DRAKE_DEMAND(IsFinite(sdf_B_A));
  DRAKE_DEMAND(sdf_A.pressure_scale > 0.0);
  DRAKE_DEMAND(sdf_B_A.pressure_scale > 0.0);
  DRAKE_DEMAND(sdf_A.characteristic_length > 0.0);
  DRAKE_DEMAND(sdf_B_A.characteristic_length > 0.0);

  const double p_A0 = -sdf_A.pressure_scale * sdf_A.value;
  const double p_B0 = -sdf_B_A.pressure_scale * sdf_B_A.value;
  const Vector3d grad_p_A = -sdf_A.pressure_scale * sdf_A.gradient;
  const Vector3d grad_p_B_A = -sdf_B_A.pressure_scale * sdf_B_A.gradient;
  DRAKE_DEMAND(std::isfinite(p_A0) && std::isfinite(p_B0));
  DRAKE_DEMAND(grad_p_A.allFinite() && grad_p_B_A.allFinite());

  const Vector3d grad_F = grad_p_A - grad_p_B_A;
  const double grad_F_norm = grad_F.norm();
  const double gradient_tolerance =
      kToleranceScale * std::numeric_limits<double>::epsilon() *
      std::max(grad_p_A.norm(), grad_p_B_A.norm());
  // Skip degenerate planes before normalization. Equal gradients imply that
  // the affine fields are either equal everywhere or nowhere equal.
  if (grad_F_norm <= gradient_tolerance) return std::nullopt;

  const double F_center = p_A0 - p_B0;
  const Vector3d plane_point =
      center_A - F_center * grad_F / grad_F.squaredNorm();
  const Vector3d nhat_BA_A = grad_F / grad_F_norm;

  const double radius = 0.5 * voxel_width;
  const std::array<Vector3d, 8> corners_A{{
      center_A + Vector3d(-radius, -radius, -radius),
      center_A + Vector3d(radius, -radius, -radius),
      center_A + Vector3d(-radius, radius, -radius),
      center_A + Vector3d(radius, radius, -radius),
      center_A + Vector3d(-radius, -radius, radius),
      center_A + Vector3d(radius, -radius, radius),
      center_A + Vector3d(-radius, radius, radius),
      center_A + Vector3d(radius, radius, radius),
  }};
  constexpr std::array<std::array<int, 2>, 12> edges{{
      {{0, 1}},
      {{2, 3}},
      {{4, 5}},
      {{6, 7}},
      {{0, 2}},
      {{1, 3}},
      {{4, 6}},
      {{5, 7}},
      {{0, 4}},
      {{1, 5}},
      {{2, 6}},
      {{3, 7}},
  }};

  const double spatial_tolerance =
      kToleranceScale * std::numeric_limits<double>::epsilon() *
      std::max({voxel_width, sdf_A.characteristic_length,
                sdf_B_A.characteristic_length});
  std::array<double, 8> distances{};
  for (int i = 0; i < static_cast<int>(corners_A.size()); ++i) {
    distances[i] = nhat_BA_A.dot(corners_A[i] - plane_point);
  }

  // Edge enumeration does not produce cyclic polygon order. It can also emit
  // a shared cube vertex from three edges or both endpoints of a face edge.
  std::vector<Vector3d> vertices_A;
  vertices_A.reserve(12);
  for (const auto& edge : edges) {
    const int a = edge[0];
    const int b = edge[1];
    const double d_a = distances[a];
    const double d_b = distances[b];
    const bool a_on_plane = std::abs(d_a) <= spatial_tolerance;
    const bool b_on_plane = std::abs(d_b) <= spatial_tolerance;
    if (a_on_plane) vertices_A.push_back(corners_A[a]);
    if (b_on_plane) vertices_A.push_back(corners_A[b]);
    if (!a_on_plane && !b_on_plane && std::signbit(d_a) != std::signbit(d_b)) {
      const double t = d_a / (d_a - d_b);
      vertices_A.push_back(corners_A[a] + t * (corners_A[b] - corners_A[a]));
    }
  }
  RemoveNearDuplicates(spatial_tolerance, &vertices_A);
  if (vertices_A.size() < 3) return std::nullopt;
  SortCounterClockwise(nhat_BA_A, &vertices_A);

  std::vector<Vector3d> clipped_vertices_A;
  const double grad_p_A_norm = grad_p_A.norm();
  if (grad_p_A_norm == 0.0) {
    if (p_A0 < 0.0) return std::nullopt;
    clipped_vertices_A = std::move(vertices_A);
  } else {
    const Vector3d zero_pressure_point =
        center_A - p_A0 * grad_p_A / grad_p_A.squaredNorm();
    // PosedHalfSpace keeps nonpositive signed distance. An outward normal
    // opposite grad_p_A therefore retains precisely p_A >= 0.
    const PosedHalfSpace<double> positive_pressure_half_space(
        -grad_p_A / grad_p_A_norm, zero_pressure_point,
        /*already_normalized=*/true);
    ClipPolygonByHalfSpace(vertices_A, positive_pressure_half_space,
                           &clipped_vertices_A);
  }

  RemoveNearDuplicates(spatial_tolerance, &clipped_vertices_A);
  if (clipped_vertices_A.size() < 3) return std::nullopt;
  double signed_area = CalcSignedArea(clipped_vertices_A, nhat_BA_A);
  if (signed_area < 0.0) {
    std::reverse(clipped_vertices_A.begin(), clipped_vertices_A.end());
    signed_area = -signed_area;
  }
  if (signed_area <= kMinimumPolygonArea) return std::nullopt;

  std::vector<double> pressures;
  pressures.reserve(clipped_vertices_A.size());
  for (const Vector3d& vertex_A : clipped_vertices_A) {
    const double pressure = p_A0 + grad_p_A.dot(vertex_A - center_A);
    DRAKE_DEMAND(std::isfinite(pressure));
    pressures.push_back(std::max(0.0, pressure));
  }

  return VoxelSdfContactPolygon{std::move(clipped_vertices_A),
                                std::move(pressures), nhat_BA_A, grad_p_A,
                                grad_p_B_A};
}

std::unique_ptr<ContactSurface<double>> CalcVoxelSdfCompliantContact(
    const VoxelSdfGeometry& A, const math::RigidTransformd& X_WA,
    GeometryId id_A, const VoxelSdfGeometry& B,
    const math::RigidTransformd& X_WB, GeometryId id_B) {
  DRAKE_DEMAND(id_A < id_B);

  // Traverse and build in A. B's queried point and gradient are converted into
  // A for each cell; no registered representation stores posed data.
  const math::RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const math::RigidTransformd X_BA = X_AB.inverse();
  PolyMeshBuilder<double> builder_A;
  std::vector<Vector3d> grad_p_A_A_per_face;
  std::vector<Vector3d> grad_p_B_A_per_face;

  for (int k = 0; k < A.cell_counts()[2]; ++k) {
    for (int j = 0; j < A.cell_counts()[1]; ++j) {
      for (int i = 0; i < A.cell_counts()[0]; ++i) {
        const Vector3d center_A = A.cell_center(i, j, k);
        const VoxelSdfGeometry::SdfSample& sample_A = A.sample(i, j, k);

        const Vector3d center_B = X_BA * center_A;
        const auto distance_B =
            point_distance::DistanceToPoint<double>::ComputeDistanceToBox<3>(
                B.half_widths(), center_B);
        const Vector3d& nearest_B = std::get<0>(distance_B);
        const Vector3d& gradient_B = std::get<1>(distance_B);
        const double phi_B = gradient_B.dot(center_B - nearest_B);

        const AffineSdfField sdf_A{sample_A.value, sample_A.gradient,
                                   A.pressure_scale(),
                                   A.characteristic_length()};
        const AffineSdfField sdf_B_A{phi_B, X_AB.rotation() * gradient_B,
                                     B.pressure_scale(),
                                     B.characteristic_length()};
        std::optional<VoxelSdfContactPolygon> polygon =
            CalcVoxelSdfContactPolygon(center_A, A.voxel_width(), sdf_A,
                                       sdf_B_A);
        if (!polygon.has_value()) continue;

        DRAKE_DEMAND(polygon->vertices_A.size() == polygon->pressures.size());
        std::vector<int> vertex_indices;
        vertex_indices.reserve(polygon->vertices_A.size());
        for (int v = 0; v < static_cast<int>(polygon->vertices_A.size()); ++v) {
          vertex_indices.push_back(builder_A.AddVertex(polygon->vertices_A[v],
                                                       polygon->pressures[v]));
        }
        const int faces_added = builder_A.AddPolygon(
            vertex_indices, polygon->nhat_BA_A, polygon->grad_p_A);
        DRAKE_DEMAND(faces_added == 1);
        grad_p_A_A_per_face.push_back(polygon->grad_p_A);
        grad_p_B_A_per_face.push_back(polygon->grad_p_B_A);
      }
    }
  }

  if (builder_A.num_faces() == 0) return nullptr;
  DRAKE_DEMAND(static_cast<int>(grad_p_A_A_per_face.size()) ==
               builder_A.num_faces());
  DRAKE_DEMAND(static_cast<int>(grad_p_B_A_per_face.size()) ==
               builder_A.num_faces());

  auto [mesh_A, field_A] = builder_A.MakeMeshAndField();
  // MeshFieldLinear requires that its underlying mesh be transformed first.
  mesh_A->TransformVertices(X_WA);
  field_A->Transform(X_WA);

  auto grad_p_A_W_per_face =
      std::make_unique<std::vector<Vector3d>>(std::move(grad_p_A_A_per_face));
  auto grad_p_B_W_per_face =
      std::make_unique<std::vector<Vector3d>>(std::move(grad_p_B_A_per_face));
  for (Vector3d& gradient_W : *grad_p_A_W_per_face) {
    gradient_W = X_WA.rotation() * gradient_W;
  }
  for (Vector3d& gradient_W : *grad_p_B_W_per_face) {
    gradient_W = X_WA.rotation() * gradient_W;
  }

  return std::make_unique<ContactSurface<double>>(
      id_A, id_B, std::move(mesh_A), std::move(field_A),
      std::move(grad_p_A_W_per_face), std::move(grad_p_B_W_per_face));
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
