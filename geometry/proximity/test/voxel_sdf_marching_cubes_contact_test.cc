#include "drake/geometry/proximity/voxel_sdf_marching_cubes_contact.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/marching_cubes_table.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;
using std::set;

template <typename CalcG, typename CalcMeanPressure>
std::array<MarchingCubesNode, 8> MakeNodes(
    const Vector3<int>& cube_index, const CalcG& calc_g,
    const CalcMeanPressure& calc_mean_pressure) {
  std::array<MarchingCubesNode, 8> result;
  for (int corner = 0; corner < 8; ++corner) {
    const auto& offset = kMcCornerOffsets[corner];
    const Vector3d p_AN_A =
        cube_index.cast<double>() + Vector3d(offset[0], offset[1], offset[2]);
    const double g = calc_g(p_AN_A);
    const double mean_pressure = calc_mean_pressure(p_AN_A);
    result[corner] = MarchingCubesNode{p_AN_A, mean_pressure + 0.5 * g,
                                       mean_pressure - 0.5 * g};
  }
  return result;
}

std::array<MarchingCubesNode, 8> MakeExtremeCaseOneNodes(double positive_g) {
  std::array<MarchingCubesNode, 8> result;
  const double denorm = std::numeric_limits<double>::denorm_min();
  for (int corner = 0; corner < 8; ++corner) {
    const auto& offset = kMcCornerOffsets[corner];
    result[corner].p_AN_A = Vector3d(offset[0], offset[1], offset[2]);
    if (corner == 0) {
      result[corner].pressure_A = 0.0;
      result[corner].pressure_B = denorm;
    } else {
      result[corner].pressure_A = positive_g;
      result[corner].pressure_B = 0.0;
    }
  }
  return result;
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest, PlanarCrossingAndWinding) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.x() - 0.25;
      },
      [](const Vector3d&) {
        return 2.0;
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();

  EXPECT_EQ(data.builder_A.num_vertices(), 4);
  EXPECT_EQ(data.builder_A.num_faces(), 2);
  ASSERT_EQ(data.face_centroids_A.size(), 2);
  auto [mesh_A, field_A] = data.builder_A.MakeMeshAndField();
  ASSERT_EQ(mesh_A->num_vertices(), 4);
  ASSERT_EQ(mesh_A->num_elements(), 2);
  set<std::array<double, 2>> yz_crossings;
  for (int v = 0; v < mesh_A->num_vertices(); ++v) {
    EXPECT_EQ(mesh_A->vertex(v).x(), 0.25);
    yz_crossings.insert({mesh_A->vertex(v).y(), mesh_A->vertex(v).z()});
    EXPECT_EQ(field_A->EvaluateAtVertex(v), 2.0);
  }
  const set<std::array<double, 2>> expected_yz_crossings{
      {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}};
  EXPECT_EQ(yz_crossings, expected_yz_crossings);
  for (int f = 0; f < mesh_A->num_elements(); ++f) {
    EXPECT_TRUE(
        CompareMatrices(mesh_A->face_normal(f), Vector3d::UnitX(), 1e-14));
    EXPECT_TRUE(CompareMatrices(mesh_A->element_centroid(f),
                                data.face_centroids_A[f], 1e-14));
  }
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           ConservativePressureFilterAddsNoVertices) {
  MarchingCubesContactBuilder builder_A(1.0);
  auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.squaredNorm() == 0.0 ? -1.0 : 1.0;
      },
      [](const Vector3d& p_AN_A) {
        return p_AN_A.x() == 0.0 && p_AN_A.y() == 0.0 ? -1.0 : 1.0;
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();
  EXPECT_EQ(data.builder_A.num_vertices(), 0);
  EXPECT_EQ(data.builder_A.num_faces(), 0);
  EXPECT_TRUE(data.face_centroids_A.empty());
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest, SharedGridEdgesReuseVertices) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto calc_g = [](const Vector3d& p_AN_A) {
    return p_AN_A.y() - 0.25;
  };
  const auto calc_mean_pressure = [](const Vector3d&) {
    return 2.0;
  };
  builder_A.AddCube(
      Vector3<int>(0, 0, 0),
      MakeNodes(Vector3<int>(0, 0, 0), calc_g, calc_mean_pressure));
  builder_A.AddCube(
      Vector3<int>(1, 0, 0),
      MakeNodes(Vector3<int>(1, 0, 0), calc_g, calc_mean_pressure));
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();

  EXPECT_EQ(data.builder_A.num_vertices(), 6);
  EXPECT_EQ(data.builder_A.num_faces(), 4);
  ASSERT_EQ(data.face_centroids_A.size(), 4);
  auto [mesh_A, field_A] = data.builder_A.MakeMeshAndField();
  ASSERT_EQ(mesh_A->num_elements(), 4);
  set<int> first_cube_vertices;
  set<int> second_cube_vertices;
  for (int f = 0; f < 4; ++f) {
    EXPECT_TRUE(
        CompareMatrices(mesh_A->face_normal(f), Vector3d::UnitY(), 1e-14));
    const SurfaceTriangle& triangle = mesh_A->element(f);
    set<int>& cube_vertices =
        f < 2 ? first_cube_vertices : second_cube_vertices;
    for (int v = 0; v < 3; ++v) {
      cube_vertices.insert(triangle.vertex(v));
    }
  }
  std::vector<int> shared_vertices;
  std::set_intersection(first_cube_vertices.begin(), first_cube_vertices.end(),
                        second_cube_vertices.begin(),
                        second_cube_vertices.end(),
                        std::back_inserter(shared_vertices));
  EXPECT_EQ(shared_vertices.size(), 2);
  for (int v = 0; v < mesh_A->num_vertices(); ++v) {
    EXPECT_EQ(field_A->EvaluateAtVertex(v), 2.0);
  }
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest, RejectsZeroAreaTriangle) {
  MarchingCubesContactBuilder builder_A(1.0);
  // The three case-1 crossings are a denormal distance from corner zero, so
  // their geometric cross product underflows to zero.
  builder_A.AddCube(Vector3<int>::Zero(), MakeExtremeCaseOneNodes(1.0));
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();
  EXPECT_EQ(data.builder_A.num_vertices(), 0);
  EXPECT_EQ(data.builder_A.num_faces(), 0);
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest, RejectsUnorientableTriangle) {
  MarchingCubesContactBuilder builder_A(1.0);
  const double denorm = std::numeric_limits<double>::denorm_min();
  // Symmetric denormal g values put the crossings at edge midpoints, while
  // the trilinear gradient norm underflows to zero.
  builder_A.AddCube(Vector3<int>::Zero(), MakeExtremeCaseOneNodes(denorm));
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();
  EXPECT_EQ(data.builder_A.num_vertices(), 0);
  EXPECT_EQ(data.builder_A.num_faces(), 0);
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
