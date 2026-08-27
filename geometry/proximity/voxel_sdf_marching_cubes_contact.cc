#include "drake/geometry/proximity/voxel_sdf_marching_cubes_contact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/marching_cubes_table.h"
#include "drake/geometry/proximity/voxel_sdf_contact_common.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

constexpr double kToleranceScale = 64.0;

// A raw marching-cubes vertex is identified by the dual-grid edge containing
// its g = p_A - p_B zero crossing.
using EdgeKey = std::array<int, 4>;
// A clipped rim vertex is identified by the raw mesh edge containing its
// zero-contact-pressure crossing.
using BoundaryVertexKey = std::array<EdgeKey, 2>;

struct EdgeVertex {
  Vector3d p_AV_A;
  double contact_pressure{};
  EdgeKey key{};
};

struct ClippedVertex {
  Vector3d p_AV_A;
  double contact_pressure{};
  EdgeKey raw_edge_key{};
  std::optional<BoundaryVertexKey> boundary_vertex_key;
};

struct ClippedPolygon {
  std::array<ClippedVertex, 4> vertices;
  int size{};
};

EdgeKey MakeEdgeKey(const Vector3<int>& cube_index, int edge_index) {
  const auto& endpoints = kMcEdgeEndpoints[edge_index];
  const auto& offset0 = kMcCornerOffsets[endpoints[0]];
  const auto& offset1 = kMcCornerOffsets[endpoints[1]];
  int axis = -1;
  std::array<int, 3> lowest{};
  for (int d = 0; d < 3; ++d) {
    DRAKE_DEMAND(cube_index[d] >= 0);
    DRAKE_DEMAND(cube_index[d] < std::numeric_limits<int>::max());
    const int node0 = cube_index[d] + offset0[d];
    const int node1 = cube_index[d] + offset1[d];
    if (node0 != node1) {
      DRAKE_DEMAND(axis == -1);
      axis = d;
    }
    lowest[d] = std::min(node0, node1);
  }
  DRAKE_DEMAND(axis >= 0);
  return {axis, lowest[0], lowest[1], lowest[2]};
}

BoundaryVertexKey MakeBoundaryVertexKey(const EdgeKey& a, const EdgeKey& b) {
  // Sorting makes incident triangles agree on the clipped vertex even when
  // they traverse their shared raw edge in opposite directions.
  return a < b ? BoundaryVertexKey{a, b} : BoundaryVertexKey{b, a};
}

/* Clips a raw marching-cubes triangle against linearly interpolated contact
 pressure >= 0. The input winding is preserved, and the result has at most four
 vertices. Strict sign changes create exact-zero rim vertices; existing
 exact-zero vertices are retained. */
ClippedPolygon ClipTriangleToNonnegativePressure(
    const std::array<const EdgeVertex*, 3>& triangle) {
  ClippedPolygon result;
  const auto add_raw_vertex = [&result](const EdgeVertex& vertex) {
    DRAKE_DEMAND(result.size < static_cast<int>(result.vertices.size()));
    result.vertices[result.size++] = ClippedVertex{
        vertex.p_AV_A, vertex.contact_pressure, vertex.key, std::nullopt};
  };
  const auto add_boundary_vertex = [&result](const EdgeVertex& a,
                                             const EdgeVertex& b) {
    DRAKE_DEMAND(result.size < static_cast<int>(result.vertices.size()));
    DRAKE_DEMAND((a.contact_pressure < 0.0) != (b.contact_pressure < 0.0));
    DRAKE_DEMAND(a.contact_pressure != 0.0);
    DRAKE_DEMAND(b.contact_pressure != 0.0);
    const double denominator = a.contact_pressure - b.contact_pressure;
    DRAKE_DEMAND(denominator != 0.0 && std::isfinite(denominator));
    const double t = a.contact_pressure / denominator;
    DRAKE_DEMAND(t > 0.0 && t < 1.0 && std::isfinite(t));
    result.vertices[result.size++] =
        ClippedVertex{(1.0 - t) * a.p_AV_A + t * b.p_AV_A,
                      0.0,
                      {},
                      MakeBoundaryVertexKey(a.key, b.key)};
  };

  for (int i = 0; i < 3; ++i) {
    const EdgeVertex& current = *triangle[i];
    const EdgeVertex& next = *triangle[(i + 1) % 3];
    if (current.contact_pressure >= 0.0) {
      add_raw_vertex(current);
    }
    if ((current.contact_pressure < 0.0 && next.contact_pressure > 0.0) ||
        (current.contact_pressure > 0.0 && next.contact_pressure < 0.0)) {
      add_boundary_vertex(current, next);
    }
  }
  return result;
}

Vector3d CalcTrilinearGradient(const std::array<double, 8>& values,
                               const Vector3d& uvw, double voxel_width) {
  DRAKE_DEMAND(uvw.allFinite());
  Vector3d gradient = Vector3d::Zero();
  for (int corner = 0; corner < 8; ++corner) {
    const auto& offset = kMcCornerOffsets[corner];
    const double wx = offset[0] == 0 ? 1.0 - uvw[0] : uvw[0];
    const double wy = offset[1] == 0 ? 1.0 - uvw[1] : uvw[1];
    const double wz = offset[2] == 0 ? 1.0 - uvw[2] : uvw[2];
    const double dx = offset[0] == 0 ? -1.0 : 1.0;
    const double dy = offset[1] == 0 ? -1.0 : 1.0;
    const double dz = offset[2] == 0 ? -1.0 : 1.0;
    gradient +=
        values[corner] * Vector3d(dx * wy * wz, wx * dy * wz, wx * wy * dz);
  }
  return gradient / voxel_width;
}

}  // namespace

MarchingCubesContactBuilder::MarchingCubesContactBuilder(double voxel_width)
    : voxel_width_(voxel_width) {
  DRAKE_DEMAND(voxel_width_ > 0.0 && std::isfinite(voxel_width_));
}

void MarchingCubesContactBuilder::AddCube(
    const Vector3<int>& cube_index,
    const std::array<MarchingCubesNode, 8>& nodes_A) {
  DRAKE_DEMAND(!consumed_);

  std::array<double, 8> g{};
  uint8_t case_index = 0;
  for (int corner = 0; corner < 8; ++corner) {
    const MarchingCubesNode& node = nodes_A[corner];
    DRAKE_DEMAND(node.p_AN_A.allFinite());
    DRAKE_DEMAND(std::isfinite(node.pressure_A));
    DRAKE_DEMAND(std::isfinite(node.pressure_B));
    g[corner] = node.pressure_A - node.pressure_B;
    DRAKE_DEMAND(std::isfinite(g[corner]));
    if (g[corner] < 0.0) {
      case_index |= static_cast<uint8_t>(1u << corner);
    }
  }

  std::array<std::optional<EdgeVertex>, 12> edge_vertices;
  const auto get_edge_vertex = [&](int edge_index) -> const EdgeVertex& {
    DRAKE_DEMAND(edge_index >= 0 && edge_index < 12);
    std::optional<EdgeVertex>& result = edge_vertices[edge_index];
    if (result.has_value()) return *result;

    const auto& endpoints = kMcEdgeEndpoints[edge_index];
    const int a = endpoints[0];
    const int b = endpoints[1];
    DRAKE_DEMAND((g[a] < 0.0) != (g[b] < 0.0));
    const double denominator = g[a] - g[b];
    DRAKE_DEMAND(denominator != 0.0 && std::isfinite(denominator));
    const double t = g[a] / denominator;
    DRAKE_DEMAND(t >= 0.0 && t <= 1.0 && std::isfinite(t));
    const double one_t = 1.0 - t;
    const double pressure_A =
        one_t * nodes_A[a].pressure_A + t * nodes_A[b].pressure_A;
    const double pressure_B =
        one_t * nodes_A[a].pressure_B + t * nodes_A[b].pressure_B;
    double contact_pressure = 0.5 * (pressure_A + pressure_B);
    const double pressure_scale = std::max(
        {1.0, std::abs(nodes_A[a].pressure_A), std::abs(nodes_A[a].pressure_B),
         std::abs(nodes_A[b].pressure_A), std::abs(nodes_A[b].pressure_B)});
    const double pressure_tolerance = kToleranceScale *
                                      std::numeric_limits<double>::epsilon() *
                                      pressure_scale;
    // Canonical zero prevents roundoff from giving incident triangles
    // inconsistent rim classifications.
    if (std::abs(contact_pressure) <= pressure_tolerance) {
      contact_pressure = 0.0;
    }
    result = EdgeVertex{one_t * nodes_A[a].p_AN_A + t * nodes_A[b].p_AN_A,
                        contact_pressure, MakeEdgeKey(cube_index, edge_index)};
    DRAKE_DEMAND(result->p_AV_A.allFinite());
    DRAKE_DEMAND(std::isfinite(result->contact_pressure));
    return *result;
  };

  const std::span<const int> triangles = McTriangles(case_index);
  for (int offset = 0; triangles[offset] != -1; offset += 3) {
    DRAKE_DEMAND(offset + 2 < static_cast<int>(triangles.size()));
    const std::array<int, 3> triangle_edges{{
        triangles[offset],
        triangles[offset + 1],
        triangles[offset + 2],
    }};
    std::array<const EdgeVertex*, 3> vertices{{
        &get_edge_vertex(triangle_edges[0]),
        &get_edge_vertex(triangle_edges[1]),
        &get_edge_vertex(triangle_edges[2]),
    }};

    // Corner g values locate the crossings, but the actual mesh normal comes
    // from the vertex cross product. The derivative of the trilinear g field
    // below is used only to choose that geometric normal's sign.
    const Vector3d e01 = vertices[1]->p_AV_A - vertices[0]->p_AV_A;
    const Vector3d e02 = vertices[2]->p_AV_A - vertices[0]->p_AV_A;
    const Vector3d e12 = vertices[2]->p_AV_A - vertices[1]->p_AV_A;
    const Vector3d cross = e01.cross(e02);
    const double cross_norm = cross.norm();
    const double edge_scale_squared =
        std::max({e01.squaredNorm(), e02.squaredNorm(), e12.squaredNorm()});
    DRAKE_DEMAND(std::isfinite(cross_norm));
    DRAKE_DEMAND(std::isfinite(edge_scale_squared));
    const double area_tolerance = kToleranceScale *
                                  std::numeric_limits<double>::epsilon() *
                                  edge_scale_squared;
    if (cross_norm <= area_tolerance) continue;

    const Vector3d centroid_A =
        (vertices[0]->p_AV_A + vertices[1]->p_AV_A + vertices[2]->p_AV_A) / 3.0;
    const Vector3d uvw = (centroid_A - nodes_A[0].p_AN_A) / voxel_width_;
    const Vector3d grad_g_h_A = CalcTrilinearGradient(g, uvw, voxel_width_);
    const double grad_norm = grad_g_h_A.norm();
    DRAKE_DEMAND(std::isfinite(grad_norm));
    const double orientation = cross.dot(grad_g_h_A);
    const double orientation_tolerance =
        kToleranceScale * std::numeric_limits<double>::epsilon() * cross_norm *
        grad_norm;
    DRAKE_DEMAND(std::isfinite(orientation));
    DRAKE_DEMAND(std::isfinite(orientation_tolerance));
    if (grad_norm == 0.0 || std::abs(orientation) <= orientation_tolerance) {
      continue;
    }
    if (orientation < 0.0) {
      std::swap(vertices[1], vertices[2]);
    }

    // On the raw p_A - p_B = 0 triangle, clipping their mean pressure once
    // clips both constituent fields without creating two roundoff-offset rims.
    const ClippedPolygon polygon = ClipTriangleToNonnegativePressure(vertices);
    // A clipped triangle is either unchanged, a smaller triangle, or a quad.
    // The preserved winding makes this fan consistent with the raw triangle.
    for (int i = 1; i + 1 < polygon.size; ++i) {
      const std::array<const ClippedVertex*, 3> clipped_triangle{{
          &polygon.vertices[0],
          &polygon.vertices[i],
          &polygon.vertices[i + 1],
      }};
      const Vector3d clipped_e01 =
          clipped_triangle[1]->p_AV_A - clipped_triangle[0]->p_AV_A;
      const Vector3d clipped_e02 =
          clipped_triangle[2]->p_AV_A - clipped_triangle[0]->p_AV_A;
      const Vector3d clipped_e12 =
          clipped_triangle[2]->p_AV_A - clipped_triangle[1]->p_AV_A;
      const double clipped_cross_norm = clipped_e01.cross(clipped_e02).norm();
      const double clipped_edge_scale_squared =
          std::max({clipped_e01.squaredNorm(), clipped_e02.squaredNorm(),
                    clipped_e12.squaredNorm()});
      DRAKE_DEMAND(std::isfinite(clipped_cross_norm));
      DRAKE_DEMAND(std::isfinite(clipped_edge_scale_squared));
      const double clipped_area_tolerance =
          kToleranceScale * std::numeric_limits<double>::epsilon() *
          clipped_edge_scale_squared;
      if (clipped_cross_norm <= clipped_area_tolerance) continue;

      std::array<int, 3> triangle_vertices{};
      for (int v = 0; v < 3; ++v) {
        const ClippedVertex& vertex = *clipped_triangle[v];
        int vertex_index{};
        // Rim vertices are shared by raw mesh-edge identity; untouched
        // marching-cubes vertices are shared by dual-grid-edge identity.
        if (vertex.boundary_vertex_key.has_value()) {
          const auto [iter, inserted] = boundary_vertex_indices_.try_emplace(
              *vertex.boundary_vertex_key, mesh_data_.builder_A.num_vertices());
          if (inserted) {
            vertex_index = mesh_data_.builder_A.AddVertex(
                vertex.p_AV_A, vertex.contact_pressure);
            DRAKE_DEMAND(vertex_index == iter->second);
          } else {
            vertex_index = iter->second;
          }
        } else {
          const auto [iter, inserted] = edge_vertex_indices_.try_emplace(
              vertex.raw_edge_key, mesh_data_.builder_A.num_vertices());
          if (inserted) {
            vertex_index = mesh_data_.builder_A.AddVertex(
                vertex.p_AV_A, vertex.contact_pressure);
            DRAKE_DEMAND(vertex_index == iter->second);
          } else {
            vertex_index = iter->second;
          }
        }
        triangle_vertices[v] = vertex_index;
      }
      DRAKE_DEMAND(mesh_data_.builder_A.AddTriangle(triangle_vertices) == 1);
      // One raw triangle can become two faces, so gradients need a centroid
      // recorded after clipping for each emitted face.
      mesh_data_.face_centroids_A.push_back((clipped_triangle[0]->p_AV_A +
                                             clipped_triangle[1]->p_AV_A +
                                             clipped_triangle[2]->p_AV_A) /
                                            3.0);
    }
  }
}

MarchingCubesMeshData MarchingCubesContactBuilder::TakeMeshData() && {
  DRAKE_DEMAND(!consumed_);
  DRAKE_DEMAND(static_cast<int>(mesh_data_.face_centroids_A.size()) ==
               mesh_data_.builder_A.num_faces());
  consumed_ = true;
  // The vertex caches are intentionally not moved out. They have no role after
  // the builder is transferred toward the ContactSurface ownership sink.
  return std::move(mesh_data_);
}

std::unique_ptr<ContactSurface<double>>
CalcVoxelSdfMarchingCubesContactOverRange(const VoxelSdfGeometry& A,
                                          const math::RigidTransformd& X_WA,
                                          GeometryId id_A,
                                          const VoxelSdfGeometry& B,
                                          const math::RigidTransformd& X_WB,
                                          GeometryId id_B,
                                          const VoxelSdfIndexRange& range) {
  DRAKE_DEMAND(A.evaluation_mode() == VoxelSdfEvaluationMode::kPrimitiveSdf);
  DRAKE_DEMAND(B.evaluation_mode() == VoxelSdfEvaluationMode::kPrimitiveSdf);
  DRAKE_DEMAND(A.extraction_method() ==
               VoxelSdfExtractionMethod::kMarchingCubes);
  DRAKE_DEMAND(B.extraction_method() ==
               VoxelSdfExtractionMethod::kMarchingCubes);
  for (int axis = 0; axis < 3; ++axis) {
    DRAKE_DEMAND(0 <= range.begin[axis]);
    DRAKE_DEMAND(range.begin[axis] <= range.end[axis]);
    DRAKE_DEMAND(range.end[axis] <= A.mc_cube_counts()[axis]);
  }

  // Traverse and build in A. Every query of B is expressed in B, and its
  // pressure gradient is re-expressed in A. Registered geometry stores no
  // posed data, while the edge cache remains local to this invocation.
  const math::RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const math::RigidTransformd X_BA = X_AB.inverse();
  MarchingCubesContactBuilder builder(A.voxel_width());

  for (int k = range.begin[2]; k < range.end[2]; ++k) {
    for (int j = range.begin[1]; j < range.end[1]; ++j) {
      for (int i = range.begin[0]; i < range.end[0]; ++i) {
        const Vector3<int> cube_index(i, j, k);
        std::array<MarchingCubesNode, 8> nodes_A;
        for (int corner = 0; corner < 8; ++corner) {
          const auto& offset = kMcCornerOffsets[corner];
          const int node_i = i + offset[0];
          const int node_j = j + offset[1];
          const int node_k = k + offset[2];
          const Vector3d p_AN_A = A.mc_node_position(node_i, node_j, node_k);
          const double phi_A = A.mc_node_value(node_i, node_j, node_k);
          const Vector3d p_BN_B = X_BA * p_AN_A;
          const double phi_B = B.EvaluateSdf(p_BN_B).value;
          nodes_A[corner] = MarchingCubesNode{
              p_AN_A, -A.pressure_scale() * phi_A, -B.pressure_scale() * phi_B};
        }
        builder.AddCube(cube_index, nodes_A);
      }
    }
  }

  MarchingCubesMeshData mesh_data = std::move(builder).TakeMeshData();
  std::vector<Vector3d> grad_p_A_A_per_face;
  std::vector<Vector3d> grad_p_B_A_per_face;
  grad_p_A_A_per_face.reserve(mesh_data.face_centroids_A.size());
  grad_p_B_A_per_face.reserve(mesh_data.face_centroids_A.size());
  for (const Vector3d& centroid_A : mesh_data.face_centroids_A) {
    const VoxelSdfGeometry::SdfSample sdf_A = A.EvaluateSdf(centroid_A);
    grad_p_A_A_per_face.push_back(-A.pressure_scale() * sdf_A.gradient);

    const Vector3d centroid_B = X_BA * centroid_A;
    const VoxelSdfGeometry::SdfSample sdf_B = B.EvaluateSdf(centroid_B);
    grad_p_B_A_per_face.push_back(X_AB.rotation() *
                                  (-B.pressure_scale() * sdf_B.gradient));
  }

  return FinalizeContactSurface<TriMeshBuilder<double>>(
      std::move(mesh_data.builder_A), std::move(grad_p_A_A_per_face),
      std::move(grad_p_B_A_per_face), X_WA, id_A, id_B);
}

std::unique_ptr<ContactSurface<double>> CalcVoxelSdfMarchingCubesContact(
    const VoxelSdfGeometry& A, const math::RigidTransformd& X_WA,
    GeometryId id_A, const VoxelSdfGeometry& B,
    const math::RigidTransformd& X_WB, GeometryId id_B) {
  const math::RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const VoxelSdfIndexRange range = CalcVoxelSdfCandidateRange(
      A, B, X_AB, VoxelSdfTraversalGrid::kMarchingCubes);
  return CalcVoxelSdfMarchingCubesContactOverRange(A, X_WA, id_A, B, X_WB, id_B,
                                                   range);
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
