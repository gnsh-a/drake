#include "drake/geometry/proximity/voxel_sdf_contact_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/contact_surface_utility.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

constexpr double kToleranceScale = 64.0;

struct TraversalGridData {
  Vector3<int> counts;
  Vector3d first_center;
};

TraversalGridData GetTraversalGridData(const VoxelSdfGeometry& A,
                                       VoxelSdfTraversalGrid grid) {
  switch (grid) {
    case VoxelSdfTraversalGrid::kPlaneClipCells:
      return TraversalGridData{A.cell_counts(), A.cell_center(0, 0, 0)};
    case VoxelSdfTraversalGrid::kMarchingCubes:
      return TraversalGridData{A.mc_cube_counts(),
                               A.mc_node_position(0, 0, 0) +
                                   Vector3d::Constant(0.5 * A.voxel_width())};
  }
  DRAKE_UNREACHABLE();
}

}  // namespace

VoxelSdfIndexRange MakeFullVoxelSdfIndexRange(const VoxelSdfGeometry& A,
                                              VoxelSdfTraversalGrid grid) {
  return VoxelSdfIndexRange{Vector3<int>::Zero(),
                            GetTraversalGridData(A, grid).counts};
}

VoxelSdfIndexRange CalcVoxelSdfCandidateRange(const VoxelSdfGeometry& A,
                                              const VoxelSdfGeometry& B,
                                              const math::RigidTransformd& X_AB,
                                              VoxelSdfTraversalGrid grid) {
  const TraversalGridData data = GetTraversalGridData(A, grid);
  const Vector3d center_A = X_AB.translation();
  const double spatial_tolerance = CalcSpatialTolerance(
      A.voxel_width(), A.characteristic_length(), B.characteristic_length());
  const double halo =
      0.5 * std::sqrt(3.0) * A.voxel_width() + spatial_tolerance;
  // Expand in B before transforming the box. A host element is contained by a
  // radius-`halo` ball about its center, so every B coordinate can move by at
  // most `halo`. Transforming this expanded B-frame box is conservative under
  // arbitrary relative rotation, including for Cylinder's cap/radial branch
  // partition and the stored-grid interpolation-domain test.
  const Vector3d expanded_half_width_B =
      -B.lower_cell_boundary() + Vector3d::Constant(halo);
  const Vector3d extent_A =
      X_AB.rotation().matrix().cwiseAbs() * expanded_half_width_B;
  Vector3d lower_A = center_A - extent_A;
  Vector3d upper_A = center_A + extent_A;
  for (int axis = 0; axis < 3; ++axis) {
    lower_A[axis] =
        std::nextafter(lower_A[axis], -std::numeric_limits<double>::infinity());
    upper_A[axis] =
        std::nextafter(upper_A[axis], std::numeric_limits<double>::infinity());
  }

  VoxelSdfIndexRange result;
  for (int axis = 0; axis < 3; ++axis) {
    const int count = data.counts[axis];
    const double first = data.first_center[axis];
    const double last =
        std::fma(A.voxel_width(), static_cast<double>(count - 1), first);
    if (upper_A[axis] < first || lower_A[axis] > last) return {};

    const double clipped_lower = std::max(lower_A[axis], first);
    const double clipped_upper = std::min(upper_A[axis], last);
    const double first_index =
        std::ceil((clipped_lower - first) / A.voxel_width());
    const double last_index =
        std::floor((clipped_upper - first) / A.voxel_width());
    result.begin[axis] = std::clamp(static_cast<int>(first_index), 0, count);
    result.end[axis] = std::clamp(static_cast<int>(last_index) + 1, 0, count);
  }
  return result;
}

PressureFieldSample MakePressureField(const VoxelSdfGeometry& geometry,
                                      const VoxelSdfShape::Sample& sample) {
  return PressureFieldSample{sample.value, sample.gradient,
                             geometry.pressure_scale(),
                             geometry.characteristic_length()};
}

double CalcSpatialTolerance(double voxel_width, double characteristic_length_A,
                            double characteristic_length_B) {
  return kToleranceScale * std::numeric_limits<double>::epsilon() *
         std::max(
             {voxel_width, characteristic_length_A, characteristic_length_B});
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

template <typename Builder>
std::unique_ptr<ContactSurface<double>> FinalizeContactSurface(
    Builder builder_A, std::vector<Vector3d> grad_p_A_A_per_face,
    std::vector<Vector3d> grad_p_B_A_per_face,
    const math::RigidTransformd& X_WA, GeometryId id_A, GeometryId id_B) {
  if (builder_A.num_faces() == 0) return nullptr;
  DRAKE_DEMAND(static_cast<int>(grad_p_A_A_per_face.size()) ==
               builder_A.num_faces());
  DRAKE_DEMAND(static_cast<int>(grad_p_B_A_per_face.size()) ==
               builder_A.num_faces());

  // MakeMeshAndField() consumes the builder's vectors. MeshFieldLinear
  // requires that its underlying mesh be transformed first.
  auto [mesh_A, field_A] = builder_A.MakeMeshAndField();
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

template std::unique_ptr<ContactSurface<double>>
FinalizeContactSurface<PolyMeshBuilder<double>>(PolyMeshBuilder<double>,
                                                std::vector<Vector3d>,
                                                std::vector<Vector3d>,
                                                const math::RigidTransformd&,
                                                GeometryId, GeometryId);

template std::unique_ptr<ContactSurface<double>>
FinalizeContactSurface<TriMeshBuilder<double>>(TriMeshBuilder<double>,
                                               std::vector<Vector3d>,
                                               std::vector<Vector3d>,
                                               const math::RigidTransformd&,
                                               GeometryId, GeometryId);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
