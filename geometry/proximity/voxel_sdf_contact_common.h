#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/voxel_sdf_geometry.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

enum class VoxelSdfTraversalGrid {
  kPlaneClipCells,
  kMarchingCubes,
};

/* A rectangular, half-open range of indices in one of a voxel SDF's contact
 traversal grids. */
struct VoxelSdfIndexRange {
  Vector3<int> begin{Vector3<int>::Zero()};
  Vector3<int> end{Vector3<int>::Zero()};

  bool empty() const { return (begin.array() >= end.array()).any(); }

  int64_t num_elements() const {
    if (empty()) return 0;
    const Vector3<int> extent = end - begin;
    return static_cast<int64_t>(extent[0]) * extent[1] * extent[2];
  }
};

/* Returns the complete range for `grid` in A. */
VoxelSdfIndexRange MakeFullVoxelSdfIndexRange(const VoxelSdfGeometry& A,
                                              VoxelSdfTraversalGrid grid);

/* Returns a conservative range of A elements that can contribute contact with
 B. B's core-grid box is expanded in B by an A-element circumscribed radius
 plus spatial tolerance, transformed into A, and enclosed by an AABB before its
 bounds are mapped outward to integer indices. */
VoxelSdfIndexRange CalcVoxelSdfCandidateRange(const VoxelSdfGeometry& A,
                                              const VoxelSdfGeometry& B,
                                              const math::RigidTransformd& X_AB,
                                              VoxelSdfTraversalGrid grid);

/* One signed-distance-field sample together with the scales needed to convert
 it to a pressure field. The sample's value and gradient must be expressed in
 the same frame as the point at which it will be used. */
struct PressureFieldSample {
  double value{};
  Vector3<double> gradient{};
  double pressure_scale{};
  double characteristic_length{};
};

/* Combines `sample` with the immutable scalar data owned by `geometry`. This
 copies all values; the result retains no reference to registered geometry. */
PressureFieldSample MakePressureField(const VoxelSdfGeometry& geometry,
                                      const VoxelSdfShape::Sample& sample);

double CalcSpatialTolerance(double voxel_width, double characteristic_length_A,
                            double characteristic_length_B);

Vector3<double> CalcCentroid(const std::vector<Vector3<double>>& vertices);

double CalcSignedArea(const std::vector<Vector3<double>>& vertices,
                      const Vector3<double>& normal);

void SortCounterClockwise(const Vector3<double>& normal,
                          std::vector<Vector3<double>>* vertices);

void RemoveNearDuplicates(double tolerance,
                          std::vector<Vector3<double>>* vertices);

/* Consumes contact data assembled in frame A and returns an owning
 ContactSurface in World. The builder and constituent-gradient vectors are
 accepted by value so callers must transfer their query-local data with
 std::move and must not inspect it afterward. */
template <typename Builder>
std::unique_ptr<ContactSurface<double>> FinalizeContactSurface(
    Builder builder_A, std::vector<Vector3<double>> grad_p_A_A_per_face,
    std::vector<Vector3<double>> grad_p_B_A_per_face,
    const math::RigidTransformd& X_WA, GeometryId id_A, GeometryId id_B);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
