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

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

constexpr double kToleranceScale = 64.0;

struct EdgeVertex {
  Vector3d p_AV_A;
  double pressure_A{};
  double pressure_B{};
  double contact_pressure{};
  std::array<int, 4> key{};
};

std::array<int, 4> MakeEdgeKey(const Vector3<int>& cube_index, int edge_index) {
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
    result = EdgeVertex{one_t * nodes_A[a].p_AN_A + t * nodes_A[b].p_AN_A,
                        pressure_A, pressure_B, 0.5 * (pressure_A + pressure_B),
                        MakeEdgeKey(cube_index, edge_index)};
    DRAKE_DEMAND(result->p_AV_A.allFinite());
    DRAKE_DEMAND(std::isfinite(result->pressure_A));
    DRAKE_DEMAND(std::isfinite(result->pressure_B));
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

    // V1 deliberately drops the whole triangle when any constituent pressure
    // is negative. It does not clamp or geometrically clip boundary crossings.
    const bool has_negative_pressure =
        std::any_of(vertices.begin(), vertices.end(), [](const EdgeVertex* v) {
          return v->pressure_A < 0.0 || v->pressure_B < 0.0;
        });
    if (has_negative_pressure) continue;

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

    // Resolve cache entries only after all triangle-level rejection tests.
    // The cache retains integer indices, never references into builder vectors.
    std::array<int, 3> triangle_vertices{};
    for (int v = 0; v < 3; ++v) {
      const EdgeVertex& edge_vertex = *vertices[v];
      const auto [iter, inserted] = edge_vertex_indices_.try_emplace(
          edge_vertex.key, mesh_data_.builder_A.num_vertices());
      if (inserted) {
        const int vertex_index = mesh_data_.builder_A.AddVertex(
            edge_vertex.p_AV_A, edge_vertex.contact_pressure);
        DRAKE_DEMAND(vertex_index == iter->second);
      }
      triangle_vertices[v] = iter->second;
    }
    DRAKE_DEMAND(mesh_data_.builder_A.AddTriangle(triangle_vertices) == 1);
    mesh_data_.face_centroids_A.push_back(centroid_A);
  }
}

MarchingCubesMeshData MarchingCubesContactBuilder::TakeMeshData() && {
  DRAKE_DEMAND(!consumed_);
  DRAKE_DEMAND(static_cast<int>(mesh_data_.face_centroids_A.size()) ==
               mesh_data_.builder_A.num_faces());
  consumed_ = true;
  // The edge cache is intentionally not moved out. It has no role after the
  // builder is transferred toward the ContactSurface ownership sink.
  return std::move(mesh_data_);
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
