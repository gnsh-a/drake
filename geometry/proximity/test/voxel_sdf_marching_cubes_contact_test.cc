#include "drake/geometry/proximity/voxel_sdf_marching_cubes_contact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/marching_cubes_table.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"

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

void ExpectValidMarchingCubesSurface(const ContactSurface<double>& surface) {
  ASSERT_TRUE(surface.is_triangle());
  ASSERT_GT(surface.num_vertices(), 0);
  ASSERT_GT(surface.num_faces(), 0);
  ASSERT_TRUE(surface.HasGradE_M());
  ASSERT_TRUE(surface.HasGradE_N());
  const TriangleSurfaceMesh<double>& mesh = surface.tri_mesh_W();
  const TriangleSurfaceMeshFieldLinear<double, double>& field =
      surface.tri_e_MN();
  EXPECT_EQ(&field.mesh(), &mesh);
  EXPECT_GT(mesh.total_area(), 0.0);
  for (int v = 0; v < surface.num_vertices(); ++v) {
    EXPECT_TRUE(mesh.vertex(v).allFinite());
    const double pressure = field.EvaluateAtVertex(v);
    EXPECT_TRUE(std::isfinite(pressure));
    EXPECT_GE(pressure, 0.0);
  }
  for (int f = 0; f < surface.num_faces(); ++f) {
    const Vector3d& grad_M_W = surface.EvaluateGradE_M_W(f);
    const Vector3d& grad_N_W = surface.EvaluateGradE_N_W(f);
    EXPECT_TRUE(grad_M_W.allFinite());
    EXPECT_TRUE(grad_N_W.allFinite());
    EXPECT_GT(mesh.face_normal(f).dot(grad_M_W - grad_N_W), 0.0);
  }
}

struct SphereSphereMeasurements {
  double max_surface_distance{};
  double mean_surface_distance{};
  double area{};
};

SphereSphereMeasurements CalcSphereSphereMeasurements(double voxel_width,
                                                      bool A_has_lower_id) {
  constexpr double kRadius = 1.0;
  constexpr double kSeparation = 1.85;
  constexpr double kInterfaceX = 0.5 * kSeparation;
  const GeometryId lower_id = GeometryId::get_new_id();
  const GeometryId higher_id = GeometryId::get_new_id();
  const GeometryId id_A = A_has_lower_id ? lower_id : higher_id;
  const GeometryId id_B = A_has_lower_id ? higher_id : lower_id;

  std::unique_ptr<ContactSurface<double>> retained;
  {
    const VoxelSdfGeometry A(Sphere(kRadius), voxel_width, 100.0,
                             VoxelSdfEvaluationMode::kPrimitiveSdf,
                             VoxelSdfExtractionMethod::kMarchingCubes);
    const VoxelSdfGeometry B(Sphere(kRadius), voxel_width, 100.0,
                             VoxelSdfEvaluationMode::kPrimitiveSdf,
                             VoxelSdfExtractionMethod::kMarchingCubes);
    if (voxel_width == 0.25) {
      const int last_core_i = A.cell_counts().x() - 1;
      EXPECT_LT(A.cell_center(last_core_i, 0, 0).x(), kInterfaceX);
      EXPECT_GT(-A.lower_cell_boundary().x(), kInterfaceX);
    }
    const math::RigidTransformd X_WA;
    const math::RigidTransformd X_WB(Vector3d(kSeparation, 0.0, 0.0));
    retained = CalcVoxelSdfMarchingCubesContact(A, X_WA, id_A, B, X_WB, id_B);
    if (retained == nullptr) {
      ADD_FAILURE() << "Expected a sphere-sphere marching-cubes surface";
      return {};
    }
    ExpectValidMarchingCubesSurface(*retained);

    const std::unique_ptr<ContactSurface<double>> repeated =
        CalcVoxelSdfMarchingCubesContact(A, X_WA, id_A, B, X_WB, id_B);
    if (repeated == nullptr) {
      ADD_FAILURE() << "Expected a repeated sphere-sphere surface";
      return {};
    }
    ExpectValidMarchingCubesSurface(*repeated);
    EXPECT_EQ(retained->id_M(), repeated->id_M());
    EXPECT_EQ(retained->id_N(), repeated->id_N());
    EXPECT_TRUE(retained->Equal(*repeated));
    EXPECT_NE(&retained->tri_mesh_W(), &repeated->tri_mesh_W());
    EXPECT_NE(&retained->tri_e_MN(), &repeated->tri_e_MN());
    EXPECT_EQ(&retained->tri_e_MN().mesh(), &retained->tri_mesh_W());
    EXPECT_EQ(&repeated->tri_e_MN().mesh(), &repeated->tri_mesh_W());
    EXPECT_NE(&retained->EvaluateGradE_M_W(0), &repeated->EvaluateGradE_M_W(0));
    EXPECT_NE(&retained->EvaluateGradE_N_W(0), &repeated->EvaluateGradE_N_W(0));
    for (int f = 0; f < retained->num_faces(); ++f) {
      EXPECT_EQ(retained->EvaluateGradE_M_W(f), repeated->EvaluateGradE_M_W(f));
      EXPECT_EQ(retained->EvaluateGradE_N_W(f), repeated->EvaluateGradE_N_W(f));
    }
  }

  // The registered geometries, poses, builder, and edge cache are gone. The
  // returned ContactSurface must still own every datum used below.
  ExpectValidMarchingCubesSurface(*retained);
  double max_distance = 0.0;
  double distance_sum = 0.0;
  for (int v = 0; v < retained->num_vertices(); ++v) {
    const double distance =
        std::abs(retained->tri_mesh_W().vertex(v).x() - kInterfaceX);
    max_distance = std::max(max_distance, distance);
    distance_sum += distance;
  }
  return SphereSphereMeasurements{max_distance,
                                  distance_sum / retained->num_vertices(),
                                  retained->tri_mesh_W().total_area()};
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

GTEST_TEST(VoxelSdfMarchingCubesContactTest, ClipsOnePositiveVertexToTriangle) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.squaredNorm() == 0.0 ? -1.0 : 1.0;
      },
      [](const Vector3d& p_AN_A) {
        return p_AN_A.x() - p_AN_A.y() - p_AN_A.z();
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();

  EXPECT_EQ(data.builder_A.num_vertices(), 3);
  EXPECT_EQ(data.builder_A.num_faces(), 1);
  ASSERT_EQ(data.face_centroids_A.size(), 1);
  auto [mesh_A, field_A] = data.builder_A.MakeMeshAndField();
  int zero_pressure_vertices = 0;
  int positive_pressure_vertices = 0;
  for (int v = 0; v < mesh_A->num_vertices(); ++v) {
    const double pressure = field_A->EvaluateAtVertex(v);
    if (pressure == 0.0) {
      ++zero_pressure_vertices;
    } else {
      EXPECT_EQ(pressure, 0.5);
      ++positive_pressure_vertices;
    }
  }
  EXPECT_EQ(zero_pressure_vertices, 2);
  EXPECT_EQ(positive_pressure_vertices, 1);
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           ClipsTwoPositiveVerticesToTwoTriangles) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.squaredNorm() == 0.0 ? -1.0 : 1.0;
      },
      [](const Vector3d& p_AN_A) {
        return p_AN_A.x() + p_AN_A.y() - p_AN_A.z();
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();

  EXPECT_EQ(data.builder_A.num_vertices(), 4);
  EXPECT_EQ(data.builder_A.num_faces(), 2);
  ASSERT_EQ(data.face_centroids_A.size(), 2);
  auto [mesh_A, field_A] = data.builder_A.MakeMeshAndField();
  int zero_pressure_vertices = 0;
  int positive_pressure_vertices = 0;
  for (int v = 0; v < mesh_A->num_vertices(); ++v) {
    const double pressure = field_A->EvaluateAtVertex(v);
    if (pressure == 0.0) {
      ++zero_pressure_vertices;
    } else {
      EXPECT_EQ(pressure, 0.5);
      ++positive_pressure_vertices;
    }
  }
  EXPECT_EQ(zero_pressure_vertices, 2);
  EXPECT_EQ(positive_pressure_vertices, 2);
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           RejectsAllNegativePressureTriangle) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.squaredNorm() == 0.0 ? -1.0 : 1.0;
      },
      [](const Vector3d&) {
        return -1.0;
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();
  EXPECT_EQ(data.builder_A.num_vertices(), 0);
  EXPECT_EQ(data.builder_A.num_faces(), 0);
  EXPECT_TRUE(data.face_centroids_A.empty());
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           RejectsPressureClipThatCollapsesToLine) {
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

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           SharedTriangleEdgeReusesBoundaryVertex) {
  MarchingCubesContactBuilder builder_A(1.0);
  const auto nodes_A = MakeNodes(
      Vector3<int>::Zero(),
      [](const Vector3d& p_AN_A) {
        return p_AN_A.x() - 0.25;
      },
      [](const Vector3d& p_AN_A) {
        return p_AN_A.y() - 0.5;
      });
  builder_A.AddCube(Vector3<int>::Zero(), nodes_A);
  MarchingCubesMeshData data = std::move(builder_A).TakeMeshData();

  EXPECT_EQ(data.builder_A.num_vertices(), 5);
  EXPECT_EQ(data.builder_A.num_faces(), 3);
  ASSERT_EQ(data.face_centroids_A.size(), 3);
  auto [mesh_A, field_A] = data.builder_A.MakeMeshAndField();
  int zero_pressure_vertices = 0;
  int shared_diagonal_vertices = 0;
  for (int v = 0; v < mesh_A->num_vertices(); ++v) {
    const Vector3d& p_AV_A = mesh_A->vertex(v);
    EXPECT_EQ(p_AV_A.x(), 0.25);
    if (field_A->EvaluateAtVertex(v) == 0.0) {
      ++zero_pressure_vertices;
      if (p_AV_A.y() == 0.5 && p_AV_A.z() == 0.5) {
        ++shared_diagonal_vertices;
      }
    }
  }
  EXPECT_EQ(zero_pressure_vertices, 3);
  EXPECT_EQ(shared_diagonal_vertices, 1);
  for (int f = 0; f < mesh_A->num_elements(); ++f) {
    EXPECT_TRUE(
        CompareMatrices(mesh_A->face_normal(f), Vector3d::UnitX(), 1e-14));
  }
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

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           SphereSphereRefinesAndOwnsReturnedData) {
  constexpr double kRadius = 1.0;
  constexpr double kInterfaceX = 0.925;
  const double exact_area =
      std::numbers::pi * (kRadius * kRadius - kInterfaceX * kInterfaceX);
  for (bool A_has_lower_id : {false, true}) {
    SCOPED_TRACE(A_has_lower_id ? "A has lower id" : "A has higher id");
    std::vector<SphereSphereMeasurements> measurements;
    for (double voxel_width : {0.25, 0.125, 0.0625}) {
      measurements.push_back(
          CalcSphereSphereMeasurements(voxel_width, A_has_lower_id));
    }
    ASSERT_EQ(measurements.size(), 3u);
    EXPECT_LT(measurements.back().max_surface_distance,
              measurements.front().max_surface_distance);
    EXPECT_LT(measurements.back().mean_surface_distance,
              measurements.front().mean_surface_distance);
    EXPECT_LT(std::abs(measurements.back().area - exact_area),
              std::abs(measurements.front().area - exact_area));
  }
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest,
           SphereSphereHasZeroPressureRimAndAccurateArea) {
  constexpr double kRadius = 1.0;
  constexpr double kVoxelWidth = 0.05;
  const Vector3d p_AB_A(1.5, 0.15, 0.1);
  const double separation = p_AB_A.norm();
  const double contact_radius_squared =
      kRadius * kRadius - 0.25 * separation * separation;
  const double exact_area = std::numbers::pi * contact_radius_squared;

  const VoxelSdfGeometry A(Sphere(kRadius), kVoxelWidth, 100.0,
                           VoxelSdfEvaluationMode::kPrimitiveSdf,
                           VoxelSdfExtractionMethod::kMarchingCubes);
  const VoxelSdfGeometry B(Sphere(kRadius), kVoxelWidth, 100.0,
                           VoxelSdfEvaluationMode::kPrimitiveSdf,
                           VoxelSdfExtractionMethod::kMarchingCubes);
  const std::unique_ptr<ContactSurface<double>> surface =
      CalcVoxelSdfMarchingCubesContact(
          A, math::RigidTransformd(), GeometryId::get_new_id(), B,
          math::RigidTransformd(p_AB_A), GeometryId::get_new_id());
  ASSERT_NE(surface, nullptr);
  ExpectValidMarchingCubesSurface(*surface);

  bool has_zero_pressure = false;
  for (int v = 0; v < surface->num_vertices(); ++v) {
    has_zero_pressure =
        has_zero_pressure || surface->tri_e_MN().EvaluateAtVertex(v) == 0.0;
  }
  EXPECT_TRUE(has_zero_pressure);
  EXPECT_NEAR(surface->tri_mesh_W().total_area(), exact_area,
              0.02 * exact_area);
}

/* A cylinder resting on its flat face is the case the exact-rim kernel exists
 for. The patch boundary is the circle where the cylinder's rim edge meets the
 box's top face, and the equal-pressure surface turns out of the contact plane
 there within a small fraction of one cell -- the penetration is about a
 thousandth of the coarser voxel here. Marching cubes has no vertex on that
 circle, so its rim is pinned to the last grid column inside the patch and the
 missing annulus stays roughly one cell wide however fine the grid gets. This
 is the regime the disk benchmark runs in; a test at a penetration comparable
 to the cell size would instead find plain marching cubes at its best. */
struct FlatDiskAreas {
  double plain{};
  double exact_rim{};
};

FlatDiskAreas CalcFlatDiskAreas(double voxel_width) {
  // Masterjohn et al.'s quarter-coin puck on a ground box, at the penetration
  // where the compliant normal load carries the puck's weight.
  constexpr double kRadius = 0.01213;
  constexpr double kThickness = 0.00175;
  constexpr double kBoxHalfHeight = 0.01;
  constexpr double kPenetration = 1.32e-7;
  constexpr double kModulus = 1.0e7;

  FlatDiskAreas result;
  for (const VoxelSdfExtractionMethod method :
       {VoxelSdfExtractionMethod::kMarchingCubes,
        VoxelSdfExtractionMethod::kMarchingCubesExactRim}) {
    const VoxelSdfGeometry disk(Cylinder(kRadius, kThickness), voxel_width,
                                kModulus,
                                VoxelSdfEvaluationMode::kPrimitiveSdf, method);
    const VoxelSdfGeometry ground(Box(0.04, 0.04, 2.0 * kBoxHalfHeight),
                                  voxel_width, kModulus,
                                  VoxelSdfEvaluationMode::kPrimitiveSdf,
                                  method);
    const math::RigidTransformd X_WD(Vector3d(
        0.0, 0.0, kBoxHalfHeight + 0.5 * kThickness - kPenetration));
    const std::unique_ptr<ContactSurface<double>> surface =
        method == VoxelSdfExtractionMethod::kMarchingCubes
            ? CalcVoxelSdfMarchingCubesContact(
                  disk, X_WD, GeometryId::get_new_id(), ground,
                  math::RigidTransformd(), GeometryId::get_new_id())
            : CalcVoxelSdfMarchingCubesExactRimContact(
                  disk, X_WD, GeometryId::get_new_id(), ground,
                  math::RigidTransformd(), GeometryId::get_new_id());
    if (surface == nullptr) {
      ADD_FAILURE() << "Expected a disk-on-box surface at h = " << voxel_width;
      return {};
    }
    ExpectValidMarchingCubesSurface(*surface);
    double& area = method == VoxelSdfExtractionMethod::kMarchingCubes
                       ? result.plain
                       : result.exact_rim;
    area = surface->tri_mesh_W().total_area();
  }
  return result;
}

GTEST_TEST(VoxelSdfMarchingCubesContactTest, FlatDiskRimIsPinnedToTheGrid) {
  constexpr double kRadius = 0.01213;
  constexpr double kCoarse = 1.25e-3;
  constexpr double kFine = 0.3125e-3;
  const double exact_area = std::numbers::pi * kRadius * kRadius;

  const FlatDiskAreas coarse = CalcFlatDiskAreas(kCoarse);
  const FlatDiskAreas fine = CalcFlatDiskAreas(kFine);

  // Refining by four leaves the plain rim deficit inside the same cell-wide
  // band, so the area error falls by far less than the sixteen a second-order
  // patch would give.
  for (const auto& [h, area] : {std::pair{kCoarse, coarse.plain},
                                std::pair{kFine, fine.plain}}) {
    const double rim_deficit_in_cells =
        (1.0 - std::sqrt(area / exact_area)) * kRadius / h;
    EXPECT_GT(rim_deficit_in_cells, 0.2);
    EXPECT_LT(rim_deficit_in_cells, 1.5);
  }

  // The projected rim converges: at second order the error falls by sixteen
  // over these two rungs. It may do better, but not much worse.
  const double coarse_error = std::abs(coarse.exact_rim / exact_area - 1.0);
  const double fine_error = std::abs(fine.exact_rim / exact_area - 1.0);
  EXPECT_LT(coarse_error, 0.01);
  EXPECT_GT(coarse_error / fine_error, 10.0);

  // At both rungs the projected rim beats the pinned one by more than an order
  // of magnitude.
  EXPECT_LT(coarse_error, 0.1 * std::abs(coarse.plain / exact_area - 1.0));
  EXPECT_LT(fine_error, 0.1 * std::abs(fine.plain / exact_area - 1.0));
}

/* Two spheres have no surface edge anywhere, so nothing pins the rim to the
 grid and plain marching cubes is already accurate. Projecting the rim there
 is a Newton step onto the intersection of the two tangent planes, which is a
 curved boundary's straight local model rather than an exact one. It must not
 make an already-good patch worse. It does in fact help here as well, cutting
 the area error from 1.9e-3 to 4.1e-4, but the guarantee this test defends is
 only the weaker one. */
GTEST_TEST(VoxelSdfMarchingCubesContactTest, SmoothPatchIsNotHarmedByRimProjection) {
  constexpr double kRadius = 1.0;
  constexpr double kVoxelWidth = 0.05;
  const Vector3d p_AB_A(1.5, 0.15, 0.1);
  const double separation = p_AB_A.norm();
  const double exact_area = std::numbers::pi *
                            (kRadius * kRadius - 0.25 * separation * separation);

  double plain_error = 0.0;
  double exact_rim_error = 0.0;
  for (const VoxelSdfExtractionMethod method :
       {VoxelSdfExtractionMethod::kMarchingCubes,
        VoxelSdfExtractionMethod::kMarchingCubesExactRim}) {
    const VoxelSdfGeometry A(Sphere(kRadius), kVoxelWidth, 100.0,
                             VoxelSdfEvaluationMode::kPrimitiveSdf, method);
    const VoxelSdfGeometry B(Sphere(kRadius), kVoxelWidth, 100.0,
                             VoxelSdfEvaluationMode::kPrimitiveSdf, method);
    const std::unique_ptr<ContactSurface<double>> surface =
        method == VoxelSdfExtractionMethod::kMarchingCubes
            ? CalcVoxelSdfMarchingCubesContact(
                  A, math::RigidTransformd(), GeometryId::get_new_id(), B,
                  math::RigidTransformd(p_AB_A), GeometryId::get_new_id())
            : CalcVoxelSdfMarchingCubesExactRimContact(
                  A, math::RigidTransformd(), GeometryId::get_new_id(), B,
                  math::RigidTransformd(p_AB_A), GeometryId::get_new_id());
    ASSERT_NE(surface, nullptr);
    ExpectValidMarchingCubesSurface(*surface);
    (method == VoxelSdfExtractionMethod::kMarchingCubes ? plain_error
                                                        : exact_rim_error) =
        std::abs(surface->tri_mesh_W().total_area() / exact_area - 1.0);
  }
  EXPECT_LT(plain_error, 0.02);
  EXPECT_LE(exact_rim_error, plain_error + 1e-3);
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
