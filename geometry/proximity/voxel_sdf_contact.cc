#include "drake/geometry/proximity/voxel_sdf_contact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
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

using AffineHalfSpace = VoxelSdfShape::AffineHalfSpace;
using SdfBranch = VoxelSdfGeometry::SdfBranch;

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

double CalcSpatialTolerance(double voxel_width, double characteristic_length_A,
                            double characteristic_length_B) {
  return kToleranceScale * std::numeric_limits<double>::epsilon() *
         std::max(
             {voxel_width, characteristic_length_A, characteristic_length_B});
}

bool ActiveRegionMayIntersectVoxel(
    std::span<const AffineHalfSpace> active_region_A, const Vector3d& center_A,
    double voxel_radius, double spatial_tolerance) {
  for (const AffineHalfSpace& half_space_A : active_region_A) {
    DRAKE_DEMAND(half_space_A.normal.allFinite());
    DRAKE_DEMAND(std::isfinite(half_space_A.offset));
    const double normal_norm = half_space_A.normal.norm();
    DRAKE_DEMAND(normal_norm > 0.0 && std::isfinite(normal_norm));
    // The exact minimum of n⋅p + d over an axis-aligned cube is its value at
    // the center minus r‖n‖₁. If even that minimum is positive, no point in the
    // voxel can belong to this branch. The tolerance makes this rejection
    // conservative at a shared branch boundary.
    const double minimum = half_space_A.Evaluate(center_A) -
                           voxel_radius * half_space_A.normal.cwiseAbs().sum();
    if (minimum > spatial_tolerance * normal_norm) return false;
  }
  return true;
}

AffineHalfSpace ReExpressHalfSpaceInA(const AffineHalfSpace& half_space_B,
                                      const math::RigidTransformd& X_AB,
                                      const math::RigidTransformd& X_BA) {
  // Substituting p_BQ = R_BA p_AQ + p_BAo into
  // n_B⋅p_BQ + d <= 0 gives (R_AB n_B)⋅p_AQ + n_B⋅p_BAo + d <= 0.
  return AffineHalfSpace{
      X_AB.rotation() * half_space_B.normal,
      half_space_B.normal.dot(X_BA.translation()) + half_space_B.offset};
}

SdfBranch ReExpressBranchInA(const SdfBranch& branch_B,
                             const math::RigidTransformd& X_AB,
                             const math::RigidTransformd& X_BA) {
  SdfBranch branch_A = branch_B;
  branch_A.sample.gradient = X_AB.rotation() * branch_B.sample.gradient;
  for (int i = 0; i < static_cast<int>(branch_B.active_region.size()); ++i) {
    branch_A.active_region[i] =
        ReExpressHalfSpaceInA(branch_B.active_region[i], X_AB, X_BA);
  }
  return branch_A;
}

std::vector<SdfBranch> PruneBranchesForVoxel(std::vector<SdfBranch> branches_A,
                                             const Vector3d& center_A,
                                             double voxel_radius,
                                             double spatial_tolerance) {
  std::vector<SdfBranch> result;
  result.reserve(branches_A.size());
  for (SdfBranch& branch_A : branches_A) {
    if (ActiveRegionMayIntersectVoxel(branch_A.active_region, center_A,
                                      voxel_radius, spatial_tolerance)) {
      result.push_back(std::move(branch_A));
    }
  }
  return result;
}

void ClipPolygonToActiveRegion(std::span<const AffineHalfSpace> active_region_A,
                               double spatial_tolerance,
                               std::vector<Vector3d>* vertices_A) {
  DRAKE_DEMAND(vertices_A != nullptr);
  std::vector<Vector3d> scratch_A;
  for (const AffineHalfSpace& half_space_A : active_region_A) {
    if (vertices_A->size() < 3) return;
    const double squared_norm = half_space_A.normal.squaredNorm();
    DRAKE_DEMAND(squared_norm > 0.0 && std::isfinite(squared_norm));
    // Clip to n⋅p + d <= tolerance. The small outward shift keeps adjacent
    // closed branch regions from opening a floating-point crack.
    const Vector3d boundary_point_A =
        (spatial_tolerance * std::sqrt(squared_norm) - half_space_A.offset) *
        half_space_A.normal / squared_norm;
    const PosedHalfSpace<double> posed_half_space_A(
        half_space_A.normal / std::sqrt(squared_norm), boundary_point_A,
        /*already_normalized=*/true);
    ClipPolygonByHalfSpace(*vertices_A, posed_half_space_A, &scratch_A);
    vertices_A->swap(scratch_A);
    scratch_A.clear();
    RemoveNearDuplicates(spatial_tolerance, vertices_A);
  }
}

bool IsCanonicalBranchAtPoint(std::span<const SdfBranch> branches_A,
                              const SdfBranch& selected_A,
                              const Vector3d& center_A, const Vector3d& p_AQ,
                              double spatial_tolerance) {
  const double selected_value =
      selected_A.sample.value + selected_A.sample.gradient.dot(p_AQ - center_A);
  for (const SdfBranch& candidate_A : branches_A) {
    if (candidate_A.index >= selected_A.index) continue;
    const double candidate_value =
        candidate_A.sample.value +
        candidate_A.sample.gradient.dot(p_AQ - center_A);
    if (std::abs(candidate_value - selected_value) <= spatial_tolerance) {
      return false;
    }
  }
  return true;
}

bool CellOwnsBoundaryPolygon(const VoxelSdfGeometry& A, int i, int j, int k,
                             const Vector3d& center_A,
                             std::span<const Vector3d> vertices_A,
                             double spatial_tolerance) {
  const std::array<int, 3> cell_index{i, j, k};
  const double voxel_radius = 0.5 * A.voxel_width();
  for (int axis = 0; axis < 3; ++axis) {
    if (cell_index[axis] == 0) continue;
    const double lower_boundary = center_A[axis] - voxel_radius;
    const bool lies_on_lower_boundary = std::all_of(
        vertices_A.begin(), vertices_A.end(),
        [axis, lower_boundary, spatial_tolerance](const Vector3d& vertex_A) {
          return std::abs(vertex_A[axis] - lower_boundary) <= spatial_tolerance;
        });
    // A polygon on a shared cell boundary is constructed by both adjacent
    // voxels. Assign it to the lower-index voxel, whose corresponding boundary
    // is its upper face. Repeating this rule per axis also gives unique
    // ownership to polygons on grid edges and vertices.
    if (lies_on_lower_boundary) return false;
  }
  return true;
}

bool LiesOnCellBoundary(const Vector3d& center_A, double voxel_width,
                        std::span<const Vector3d> vertices_A,
                        double spatial_tolerance) {
  // Crossing a cell boundary is ordinary and needs no special ownership. Only
  // flag a polygon when every vertex lies on the same face of the host cell;
  // such a polygon can be emitted in full by the cell across that face.
  const double voxel_radius = 0.5 * voxel_width;
  for (int axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      const double boundary = center_A[axis] + sign * voxel_radius;
      const bool lies_on_boundary = std::all_of(
          vertices_A.begin(), vertices_A.end(),
          [axis, boundary, spatial_tolerance](const Vector3d& vertex_A) {
            return std::abs(vertex_A[axis] - boundary) <= spatial_tolerance;
          });
      if (lies_on_boundary) return true;
    }
  }
  return false;
}

bool NearlyEqual(double a, double b) {
  // Pressure and gradient data can differ by a few ulps when the same field is
  // evaluated from opposite sides of an interpolation-cell boundary. Scale the
  // comparison by the data rather than reusing the length-valued tolerance.
  const double tolerance =
      kToleranceScale * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= tolerance;
}

bool NearlyEqual(const Vector3d& a, const Vector3d& b) {
  const double tolerance =
      kToleranceScale * std::numeric_limits<double>::epsilon() *
      std::max({1.0, a.norm(), b.norm()});
  return (a - b).norm() <= tolerance;
}

bool AreEquivalentBoundaryPolygons(const VoxelSdfContactPolygon& a,
                                   const VoxelSdfContactPolygon& b,
                                   double spatial_tolerance) {
  // Geometric coincidence alone is insufficient: two local sampled fields can
  // place polygons on the same host-cell face while assigning them different
  // pressure data. Treat polygons as duplicate copies only when their geometry
  // and all data stored on ContactSurface agree. Vertex order is deliberately
  // ignored because independently clipped copies need not choose the same
  // starting vertex.
  if (a.vertices_A.size() != b.vertices_A.size() ||
      a.pressures.size() != b.pressures.size() ||
      a.vertices_A.size() != a.pressures.size()) {
    return false;
  }
  if ((CalcCentroid(a.vertices_A) - CalcCentroid(b.vertices_A)).norm() >
          spatial_tolerance ||
      !NearlyEqual(a.nhat_BA_A, b.nhat_BA_A) ||
      !NearlyEqual(a.grad_p_A, b.grad_p_A) ||
      !NearlyEqual(a.grad_p_B_A, b.grad_p_B_A)) {
    return false;
  }

  std::vector<bool> matched_b(b.vertices_A.size(), false);
  for (int va = 0; va < static_cast<int>(a.vertices_A.size()); ++va) {
    bool matched = false;
    for (int vb = 0; vb < static_cast<int>(b.vertices_A.size()); ++vb) {
      if (matched_b[vb]) continue;
      if ((a.vertices_A[va] - b.vertices_A[vb]).norm() <= spatial_tolerance &&
          NearlyEqual(a.pressures[va], b.pressures[vb])) {
        matched_b[vb] = true;
        matched = true;
        break;
      }
    }
    if (!matched) return false;
  }
  return true;
}

using CellIndex = std::tuple<int, int, int>;
// Only boundary-contained polygons are retained here. The map is indexed by
// the host cell that supplied the accepted copy, not by a quantized world-space
// position, so duplicate detection does not introduce another geometric
// tolerance or depend on the pose X_WA.
using BoundaryPolygonMap =
    std::map<CellIndex, std::vector<VoxelSdfContactPolygon>>;

bool HasEquivalentBoundaryPolygon(const VoxelSdfContactPolygon& polygon,
                                  int i, int j, int k,
                                  double spatial_tolerance,
                                  const BoundaryPolygonMap& accepted) {
  // Two nonzero-area polygons clipped to different host voxels can be
  // coincident only if those closed cubes touch: they must come from the same
  // cell or one of its 26 face-, edge-, or vertex-neighbors. The k-j-i
  // traversal makes one of those copies arrive first; retaining that accepted
  // copy gives deterministic ownership without assuming that a non-invariant
  // field also produced a copy in any particular lower-index neighbor.
  for (int dk = -1; dk <= 1; ++dk) {
    for (int dj = -1; dj <= 1; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        const auto iter = accepted.find(CellIndex{i + di, j + dj, k + dk});
        if (iter == accepted.end()) continue;
        for (const VoxelSdfContactPolygon& candidate : iter->second) {
          if (AreEquivalentBoundaryPolygons(polygon, candidate,
                                            spatial_tolerance)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

std::optional<VoxelSdfContactPolygon> DoCalcVoxelSdfContactPolygon(
    const Vector3d& center_A, double voxel_width, const AffineSdfField& sdf_A,
    const AffineSdfField& sdf_B_A,
    std::span<const AffineHalfSpace> active_region_A,
    std::span<const AffineHalfSpace> active_region_B_A) {
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
  const double radius = 0.5 * voxel_width;
  const double pressure_tolerance =
      kToleranceScale * std::numeric_limits<double>::epsilon() *
      std::max({std::abs(p_A0), std::abs(p_B0), radius * grad_p_A.norm(),
                radius * grad_p_B_A.norm()});
  // This is the exact range of the affine equal-pressure field over the cube.
  // Rejecting a branch pair here is equivalent to the later plane-cube test;
  // it does not skip evaluation of the voxel or any other branch pair.
  const double maximum_variation = radius * grad_F.cwiseAbs().sum();
  if (std::abs(F_center) > maximum_variation + pressure_tolerance) {
    return std::nullopt;
  }
  const Vector3d plane_point =
      center_A - F_center * grad_F / grad_F.squaredNorm();
  const Vector3d nhat_BA_A = grad_F / grad_F_norm;

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

  const double spatial_tolerance = CalcSpatialTolerance(
      voxel_width, sdf_A.characteristic_length, sdf_B_A.characteristic_length);
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

  ClipPolygonToActiveRegion(active_region_A, spatial_tolerance,
                            &clipped_vertices_A);
  ClipPolygonToActiveRegion(active_region_B_A, spatial_tolerance,
                            &clipped_vertices_A);
  // A Box's face-affine maximum equals its Euclidean SDF only inside the Box.
  // This construction remains exact there: active-region clipping makes the
  // selected face value the maximum, and equal nonnegative pressures make that
  // maximum nonpositive for both geometries. Therefore every retained Box
  // point is inside or on its boundary; no exterior edge or corner formula is
  // being approximated by these branches.
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

}  // namespace

std::optional<VoxelSdfContactPolygon> CalcVoxelSdfContactPolygon(
    const Vector3d& center_A, double voxel_width, const AffineSdfField& sdf_A,
    const AffineSdfField& sdf_B_A) {
  return DoCalcVoxelSdfContactPolygon(center_A, voxel_width, sdf_A, sdf_B_A, {},
                                      {});
}

std::unique_ptr<ContactSurface<double>> CalcVoxelSdfCompliantContact(
    const VoxelSdfGeometry& A, const math::RigidTransformd& X_WA,
    GeometryId id_A, const VoxelSdfGeometry& B,
    const math::RigidTransformd& X_WB, GeometryId id_B) {
  if (B.evaluation_mode() == VoxelSdfEvaluationMode::kSampledTrilinear) {
    DRAKE_DEMAND(A.voxel_width() <= B.voxel_width());
  }
  // Traverse and build in A. B's queried point and gradient are converted into
  // A for each cell; no registered representation stores posed data. A may
  // have either id; ContactSurface orders M and N by GeometryId.
  const math::RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const math::RigidTransformd X_BA = X_AB.inverse();
  PolyMeshBuilder<double> builder_A;
  std::vector<Vector3d> grad_p_A_A_per_face;
  std::vector<Vector3d> grad_p_B_A_per_face;
  // Cell-invariant primitive branches use CellOwnsBoundaryPolygon() below,
  // which can assign ownership without inspecting the neighboring cell.
  // Sampled affine branches lack that guarantee, so remember accepted boundary
  // polygons and suppress only a subsequently confirmed equivalent copy.
  BoundaryPolygonMap accepted_boundary_polygons;

  for (int k = 0; k < A.cell_counts()[2]; ++k) {
    for (int j = 0; j < A.cell_counts()[1]; ++j) {
      for (int i = 0; i < A.cell_counts()[0]; ++i) {
        const Vector3d center_A = A.cell_center(i, j, k);
        const Vector3d center_B = X_BA * center_A;
        const double voxel_radius = 0.5 * A.voxel_width();
        const double spatial_tolerance =
            CalcSpatialTolerance(A.voxel_width(), A.characteristic_length(),
                                 B.characteristic_length());

        if (B.evaluation_mode() == VoxelSdfEvaluationMode::kSampledTrilinear) {
          // Reject only when the B-frame AABB of the rotated host cube cannot
          // overlap B's core grid. The dispatch-selected h_A <= h_B bounds the
          // largest projected half-width by sqrt(3) h_A / 2 <= 0.866 h_B,
          // while the stored sample-center lattice extends 1.5 h_B beyond the
          // core boundary. Therefore every potentially overlapping center is
          // inside B's interpolation domain.
          const Vector3d extent_B = X_BA.rotation().matrix().cwiseAbs() *
                                    Vector3d::Constant(voxel_radius);
          const Vector3d core_lower_B = B.lower_cell_boundary();
          const Vector3d core_upper_B = -core_lower_B;
          const bool separated = ((center_B + extent_B).array() <
                                  (core_lower_B.array() - spatial_tolerance))
                                     .any() ||
                                 ((center_B - extent_B).array() >
                                  (core_upper_B.array() + spatial_tolerance))
                                     .any();
          if (separated) continue;
        }

        // Every A voxel is visited. Pruning below only removes an affine shape
        // branch whose active region provably cannot intersect this voxel.
        const std::vector<SdfBranch> branches_A =
            PruneBranchesForVoxel(A.CalcCellSdfBranches(i, j, k), center_A,
                                  voxel_radius, spatial_tolerance);
        std::vector<SdfBranch> branches_B_A;
        if (B.evaluation_mode() == VoxelSdfEvaluationMode::kPrimitiveAffine) {
          for (const SdfBranch& branch_B : B.EvaluateSdfBranches(center_B)) {
            branches_B_A.push_back(ReExpressBranchInA(branch_B, X_AB, X_BA));
          }
        } else {
          const std::optional<VoxelSdfGeometry::SdfSample> sample_B =
              B.InterpolateSdf(center_B);
          // A center whose host cube can overlap B's core must be bracketed by
          // the fixed stored padding. Falling outside is an internal invariant
          // failure, never permission to evaluate the primitive.
          DRAKE_DEMAND(sample_B.has_value());
          // Trilinear interpolation contains one scalar field in each lattice
          // cell, not six recoverable Box face branches. The affine kernel
          // linearizes that field at the host center, so its polygon is not the
          // exact intersection of a curved trilinear equal-pressure surface.
          // Persistent feature errors are a representation/kernel limitation;
          // the branch-aware primitive mode remains the reference instead of
          // serving as a hidden fallback. A zero interpolated gradient is
          // valid; only the combined equal-pressure-plane degeneracy is
          // rejected by the polygon kernel.
          const SdfBranch branch_B{*sample_B, {}, 0, false};
          branches_B_A.push_back(ReExpressBranchInA(branch_B, X_AB, X_BA));
        }
        branches_B_A = PruneBranchesForVoxel(std::move(branches_B_A), center_A,
                                             voxel_radius, spatial_tolerance);

        for (const SdfBranch& branch_A : branches_A) {
          const AffineSdfField sdf_A{
              branch_A.sample.value, branch_A.sample.gradient,
              A.pressure_scale(), A.characteristic_length()};
          for (const SdfBranch& branch_B_A : branches_B_A) {
            const AffineSdfField sdf_B_A{
                branch_B_A.sample.value, branch_B_A.sample.gradient,
                B.pressure_scale(), B.characteristic_length()};
            std::optional<VoxelSdfContactPolygon> polygon =
                DoCalcVoxelSdfContactPolygon(center_A, A.voxel_width(), sdf_A,
                                             sdf_B_A, branch_A.active_region,
                                             branch_B_A.active_region);
            if (!polygon.has_value()) continue;

            // Closed active regions overlap on their boundaries. A contact
            // polygon normally crosses such a boundary and has its centroid
            // strictly within one region. If the entire polygon lies on a tie,
            // only the lowest stable branch index owns it.
            const Vector3d centroid_A = CalcCentroid(polygon->vertices_A);
            if (!IsCanonicalBranchAtPoint(branches_A, branch_A, center_A,
                                          centroid_A, spatial_tolerance) ||
                !IsCanonicalBranchAtPoint(branches_B_A, branch_B_A, center_A,
                                          centroid_A, spatial_tolerance)) {
              continue;
            }
            const bool has_cell_invariant_fields =
                branch_A.is_cell_invariant && branch_B_A.is_cell_invariant;
            if (has_cell_invariant_fields &&
                !CellOwnsBoundaryPolygon(A, i, j, k, center_A,
                                         polygon->vertices_A,
                                         spatial_tolerance)) {
              continue;
            }

            DRAKE_DEMAND(polygon->vertices_A.size() ==
                         polygon->pressures.size());
            const bool lies_on_cell_boundary = LiesOnCellBoundary(
                center_A, A.voxel_width(), polygon->vertices_A,
                spatial_tolerance);
            // Sampled affine fields are local linearizations and can change
            // across cells. Therefore, unlike the cell-invariant path above, a
            // polygon on this cell's lower boundary cannot simply be discarded
            // on the assumption that its neighbor emitted the same polygon.
            // Suppress only a copy confirmed against an already accepted cell.
            if (lies_on_cell_boundary &&
                HasEquivalentBoundaryPolygon(*polygon, i, j, k,
                                             spatial_tolerance,
                                             accepted_boundary_polygons)) {
              continue;
            }
            std::vector<int> vertex_indices;
            vertex_indices.reserve(polygon->vertices_A.size());
            for (int v = 0; v < static_cast<int>(polygon->vertices_A.size());
                 ++v) {
              vertex_indices.push_back(builder_A.AddVertex(
                  polygon->vertices_A[v], polygon->pressures[v]));
            }
            // The contact field convention stores geometry M's pressure
            // gradient. M is the lower-id geometry even when the higher-id A
            // supplies the finer traversal grid.
            const Vector3d& grad_p_M_A =
                id_A < id_B ? polygon->grad_p_A : polygon->grad_p_B_A;
            const int faces_added = builder_A.AddPolygon(
                vertex_indices, polygon->nhat_BA_A, grad_p_M_A);
            DRAKE_DEMAND(faces_added == 1);
            grad_p_A_A_per_face.push_back(polygon->grad_p_A);
            grad_p_B_A_per_face.push_back(polygon->grad_p_B_A);
            if (lies_on_cell_boundary) {
              accepted_boundary_polygons[CellIndex{i, j, k}].push_back(
                  *polygon);
            }
          }
        }
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
