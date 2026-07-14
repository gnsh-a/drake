#include "drake/geometry/proximity/voxel_sdf_contact.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;
using math::RigidTransformd;
using math::RotationMatrixd;

constexpr double kTolerance = 1e-13;

AffineSdfField MakeSdfFromPressure(double pressure_at_center,
                                   const Vector3d& pressure_gradient,
                                   double pressure_scale = 1.0,
                                   double characteristic_length = 1.0) {
  return AffineSdfField{-pressure_at_center / pressure_scale,
                        -pressure_gradient / pressure_scale, pressure_scale,
                        characteristic_length};
}

struct FieldPair {
  AffineSdfField A;
  AffineSdfField B_A;
};

// Returns fields whose equal-pressure plane is
// normal.dot(x - center) = offset. On that plane, their common pressure is
// common_pressure + tangent_gradient.dot(x - center).
FieldPair MakeFieldsForPlane(
    const Vector3d& normal, double offset, double common_pressure,
    const Vector3d& tangent_gradient = Vector3d::Zero()) {
  const Vector3d grad_p_A = 0.5 * normal + tangent_gradient;
  const Vector3d grad_p_B = -0.5 * normal + tangent_gradient;
  return FieldPair{
      MakeSdfFromPressure(common_pressure - 0.5 * offset, grad_p_A),
      MakeSdfFromPressure(common_pressure + 0.5 * offset, grad_p_B)};
}

double EvaluatePressure(const AffineSdfField& sdf, const Vector3d& center,
                        const Vector3d& point) {
  const double p0 = -sdf.pressure_scale * sdf.value;
  const Vector3d grad_p = -sdf.pressure_scale * sdf.gradient;
  return p0 + grad_p.dot(point - center);
}

double CalcArea(const VoxelSdfContactPolygon& polygon) {
  Vector3d centroid = Vector3d::Zero();
  for (const Vector3d& vertex : polygon.vertices_A) centroid += vertex;
  centroid /= static_cast<double>(polygon.vertices_A.size());
  double twice_area = 0.0;
  for (int i = 0; i < static_cast<int>(polygon.vertices_A.size()); ++i) {
    const Vector3d a = polygon.vertices_A[i] - centroid;
    const Vector3d b =
        polygon.vertices_A[(i + 1) % polygon.vertices_A.size()] - centroid;
    twice_area += polygon.nhat_BA_A.dot(a.cross(b));
  }
  return 0.5 * twice_area;
}

void ExpectPolygonInvariants(const VoxelSdfContactPolygon& polygon,
                             const Vector3d& center, double voxel_width,
                             const Vector3d& plane_normal,
                             double plane_offset) {
  ASSERT_GE(polygon.vertices_A.size(), 3u);
  ASSERT_EQ(polygon.pressures.size(), polygon.vertices_A.size());
  EXPECT_TRUE(CompareMatrices(polygon.nhat_BA_A, plane_normal.normalized(),
                              kTolerance));
  EXPECT_GT(CalcArea(polygon), 0.0);

  const double radius = 0.5 * voxel_width;
  for (int i = 0; i < static_cast<int>(polygon.vertices_A.size()); ++i) {
    const Vector3d& vertex = polygon.vertices_A[i];
    EXPECT_NEAR(plane_normal.dot(vertex - center), plane_offset, kTolerance);
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_GE(vertex[axis], center[axis] - radius - kTolerance);
      EXPECT_LE(vertex[axis], center[axis] + radius + kTolerance);
    }
    EXPECT_TRUE(std::isfinite(polygon.pressures[i]));
    EXPECT_GE(polygon.pressures[i], 0.0);
    for (int j = i + 1; j < static_cast<int>(polygon.vertices_A.size()); ++j) {
      EXPECT_GT((vertex - polygon.vertices_A[j]).norm(), kTolerance);
    }
  }
}

std::pair<GeometryId, GeometryId> MakeOrderedGeometryIds() {
  const GeometryId id_A = GeometryId::get_new_id();
  const GeometryId id_B = GeometryId::get_new_id();
  EXPECT_LT(id_A, id_B);
  return {id_A, id_B};
}

void ExpectSurfaceInvariants(const ContactSurface<double>& surface,
                             GeometryId id_A, GeometryId id_B) {
  EXPECT_EQ(surface.id_M(), id_A);
  EXPECT_EQ(surface.id_N(), id_B);
  EXPECT_EQ(surface.representation(),
            HydroelasticContactRepresentation::kPolygon);
  EXPECT_FALSE(surface.is_triangle());
  EXPECT_GT(surface.num_faces(), 0);
  EXPECT_GT(surface.num_vertices(), 0);
  EXPECT_TRUE(surface.HasGradE_M());
  EXPECT_TRUE(surface.HasGradE_N());

  const PolygonSurfaceMesh<double>& mesh_W = surface.poly_mesh_W();
  const PolygonSurfaceMeshFieldLinear<double, double>& field_W =
      surface.poly_e_MN();
  int face_vertex_count = 0;
  for (int v = 0; v < mesh_W.num_vertices(); ++v) {
    EXPECT_TRUE(mesh_W.vertex(v).allFinite());
    const double pressure = field_W.EvaluateAtVertex(v);
    EXPECT_TRUE(std::isfinite(pressure));
    EXPECT_GE(pressure, 0.0);
  }
  for (int f = 0; f < mesh_W.num_faces(); ++f) {
    const SurfacePolygon face = mesh_W.element(f);
    face_vertex_count += face.num_vertices();
    EXPECT_GT(face.num_vertices(), 2);
    EXPECT_TRUE(std::isfinite(surface.area(f)));
    EXPECT_GT(surface.area(f), 0.0);
    EXPECT_TRUE(surface.face_normal(f).allFinite());
    EXPECT_NEAR(surface.face_normal(f).norm(), 1.0, 1e-13);
    EXPECT_TRUE(surface.centroid(f).allFinite());

    const double centroid_pressure =
        field_W.EvaluateCartesian(f, surface.centroid(f));
    EXPECT_TRUE(std::isfinite(centroid_pressure));
    EXPECT_GE(centroid_pressure, -1e-12);
    const int first_vertex = face.vertex(0);
    EXPECT_NEAR(field_W.EvaluateCartesian(f, mesh_W.vertex(first_vertex)),
                field_W.EvaluateAtVertex(first_vertex), 1e-12);

    const Vector3d& grad_p_A_W = surface.EvaluateGradE_M_W(f);
    const Vector3d& grad_p_B_W = surface.EvaluateGradE_N_W(f);
    EXPECT_TRUE(grad_p_A_W.allFinite());
    EXPECT_TRUE(grad_p_B_W.allFinite());
    EXPECT_TRUE(
        CompareMatrices(field_W.EvaluateGradient(f), grad_p_A_W, 1e-13));
    // The face normal must follow increasing A pressure and decreasing B
    // pressure, i.e., point out of B and into A.
    EXPECT_GT(surface.face_normal(f).dot(grad_p_A_W - grad_p_B_W), 0.0);
  }
  // Every voxel contributes an independent polygon; vertices are not welded
  // or shared across faces.
  EXPECT_EQ(face_vertex_count, mesh_W.num_vertices());
}

GTEST_TEST(VoxelSdfContactTest, PlaneCubeSectionTopologies) {
  const Vector3d center = Vector3d::Zero();
  constexpr double voxel_width = 2.0;
  struct TestCase {
    Vector3d normal;
    double offset{};
    int expected_vertices{};
    std::string description;
  };
  const std::vector<TestCase> cases{
      {Vector3d(1, 1, 1), 2.0, 3, "triangle"},
      {Vector3d::UnitX(), 0.0, 4, "quadrilateral"},
      {Vector3d(1, 1, 2), -1.0, 5, "pentagon"},
      {Vector3d(1, 1, 1), 0.0, 6, "hexagon"},
  };
  for (const TestCase& test : cases) {
    SCOPED_TRACE(test.description);
    const FieldPair fields = MakeFieldsForPlane(test.normal, test.offset, 10.0);
    const auto result =
        CalcVoxelSdfContactPolygon(center, voxel_width, fields.A, fields.B_A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->vertices_A.size(), test.expected_vertices);
    ExpectPolygonInvariants(*result, center, voxel_width, test.normal,
                            test.offset);
    for (double pressure : result->pressures)
      EXPECT_NEAR(pressure, 10.0, 1e-12);
  }

  const FieldPair miss = MakeFieldsForPlane(Vector3d::UnitX(), 2.0, 10.0);
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, voxel_width, miss.A, miss.B_A)
                   .has_value());
}

GTEST_TEST(VoxelSdfContactTest, CubeBoundaryCases) {
  const Vector3d center = Vector3d::Zero();
  constexpr double voxel_width = 2.0;

  const FieldPair face = MakeFieldsForPlane(Vector3d::UnitX(), 1.0, 10.0);
  const auto face_result =
      CalcVoxelSdfContactPolygon(center, voxel_width, face.A, face.B_A);
  ASSERT_TRUE(face_result.has_value());
  EXPECT_EQ(face_result->vertices_A.size(), 4u);
  ExpectPolygonInvariants(*face_result, center, voxel_width, Vector3d::UnitX(),
                          1.0);

  // This plane contains two complete, opposite cube edges and still defines a
  // nonzero-area quadrilateral.
  const Vector3d edge_normal(1, -1, 0);
  const FieldPair through_edges = MakeFieldsForPlane(edge_normal, 0.0, 10.0);
  const auto edge_result = CalcVoxelSdfContactPolygon(
      center, voxel_width, through_edges.A, through_edges.B_A);
  ASSERT_TRUE(edge_result.has_value());
  EXPECT_EQ(edge_result->vertices_A.size(), 4u);
  ExpectPolygonInvariants(*edge_result, center, voxel_width, edge_normal, 0.0);

  // The plane x + y + z = 1 passes through three cube vertices.
  const Vector3d vertex_normal(1, 1, 1);
  const FieldPair through_vertices =
      MakeFieldsForPlane(vertex_normal, 1.0, 10.0);
  const auto vertex_result = CalcVoxelSdfContactPolygon(
      center, voxel_width, through_vertices.A, through_vertices.B_A);
  ASSERT_TRUE(vertex_result.has_value());
  EXPECT_EQ(vertex_result->vertices_A.size(), 3u);
  ExpectPolygonInvariants(*vertex_result, center, voxel_width, vertex_normal,
                          1.0);

  // Supporting planes that touch only an edge or vertex have no area.
  const FieldPair edge_only = MakeFieldsForPlane(Vector3d(1, 1, 0), 2.0, 10.0);
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, voxel_width, edge_only.A,
                                          edge_only.B_A)
                   .has_value());
  const FieldPair vertex_only =
      MakeFieldsForPlane(Vector3d(1, 1, 1), 3.0, 10.0);
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, voxel_width, vertex_only.A,
                                          vertex_only.B_A)
                   .has_value());
}

GTEST_TEST(VoxelSdfContactTest, NearlyDegeneratePressureGradients) {
  const double eps = std::numeric_limits<double>::epsilon();
  const Vector3d center = Vector3d::Zero();
  const AffineSdfField A = MakeSdfFromPressure(1.0, Vector3d::UnitX());

  const AffineSdfField equal = MakeSdfFromPressure(1.0, Vector3d::UnitX());
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, 2.0, A, equal).has_value());

  const AffineSdfField below =
      MakeSdfFromPressure(1.0, (1.0 + 32.0 * eps) * Vector3d::UnitX());
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, 2.0, A, below).has_value());

  const AffineSdfField above =
      MakeSdfFromPressure(1.0, (1.0 + 256.0 * eps) * Vector3d::UnitX());
  const auto result = CalcVoxelSdfContactPolygon(center, 2.0, A, above);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->vertices_A.size(), 4u);
  EXPECT_GT(result->nhat_BA_A.dot(result->grad_p_A - result->grad_p_B_A), 0.0);
}

GTEST_TEST(VoxelSdfContactTest, PositivePressureClipping) {
  const Vector3d center = Vector3d::Zero();
  constexpr double voxel_width = 2.0;
  const Vector3d normal = Vector3d::UnitX();
  const Vector3d tangent_gradient = Vector3d::UnitY();

  const FieldPair retained =
      MakeFieldsForPlane(normal, 0.0, 2.0, tangent_gradient);
  const auto retained_result =
      CalcVoxelSdfContactPolygon(center, voxel_width, retained.A, retained.B_A);
  ASSERT_TRUE(retained_result.has_value());
  EXPECT_NEAR(CalcArea(*retained_result), 4.0, kTolerance);

  const FieldPair reduced =
      MakeFieldsForPlane(normal, 0.0, 0.0, tangent_gradient);
  const auto reduced_result =
      CalcVoxelSdfContactPolygon(center, voxel_width, reduced.A, reduced.B_A);
  ASSERT_TRUE(reduced_result.has_value());
  EXPECT_NEAR(CalcArea(*reduced_result), 2.0, kTolerance);
  bool has_zero_pressure = false;
  for (int i = 0; i < static_cast<int>(reduced_result->vertices_A.size());
       ++i) {
    const Vector3d& vertex = reduced_result->vertices_A[i];
    const double p_A = EvaluatePressure(reduced.A, center, vertex);
    const double p_B = EvaluatePressure(reduced.B_A, center, vertex);
    EXPECT_NEAR(p_A, p_B, kTolerance);
    EXPECT_NEAR(reduced_result->pressures[i], std::max(0.0, p_A), kTolerance);
    has_zero_pressure |= reduced_result->pressures[i] == 0.0;
  }
  EXPECT_TRUE(has_zero_pressure);

  const FieldPair removed =
      MakeFieldsForPlane(normal, 0.0, -2.0, tangent_gradient);
  EXPECT_FALSE(
      CalcVoxelSdfContactPolygon(center, voxel_width, removed.A, removed.B_A)
          .has_value());
}

GTEST_TEST(VoxelSdfContactTest, ConstantAPressure) {
  const Vector3d center = Vector3d::Zero();
  const AffineSdfField positive_A = MakeSdfFromPressure(1.0, Vector3d::Zero());
  const AffineSdfField positive_B =
      MakeSdfFromPressure(1.0, -Vector3d::UnitX());
  const auto positive =
      CalcVoxelSdfContactPolygon(center, 2.0, positive_A, positive_B);
  ASSERT_TRUE(positive.has_value());
  EXPECT_EQ(positive->vertices_A.size(), 4u);
  for (double pressure : positive->pressures) EXPECT_EQ(pressure, 1.0);

  const AffineSdfField negative_A = MakeSdfFromPressure(-1.0, Vector3d::Zero());
  const AffineSdfField negative_B =
      MakeSdfFromPressure(-1.0, -Vector3d::UnitX());
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, 2.0, negative_A, negative_B)
                   .has_value());
}

GTEST_TEST(VoxelSdfContactTest, RejectsNegligibleArea) {
  const Vector3d center = Vector3d::Zero();
  const FieldPair fields = MakeFieldsForPlane(Vector3d::UnitX(), 0.0, 1.0);
  // The square section has area 1e-16 m², below the 1e-14 m² threshold.
  EXPECT_FALSE(CalcVoxelSdfContactPolygon(center, 1e-8, fields.A, fields.B_A)
                   .has_value());
}

GTEST_TEST(VoxelSdfContactTest, ResultOwnsIndependentStorage) {
  std::optional<VoxelSdfContactPolygon> result;
  {
    const Vector3d temporary_center(1.0, 2.0, 3.0);
    const FieldPair temporary_fields =
        MakeFieldsForPlane(Vector3d::UnitX(), 0.0, 2.0);
    result = CalcVoxelSdfContactPolygon(
        temporary_center, 2.0, temporary_fields.A, temporary_fields.B_A);
  }
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result->vertices_A.empty());
  ASSERT_FALSE(result->pressures.empty());

  VoxelSdfContactPolygon copy = *result;
  EXPECT_NE(copy.vertices_A.data(), result->vertices_A.data());
  EXPECT_NE(copy.pressures.data(), result->pressures.data());
  const Vector3d copied_vertex = copy.vertices_A[0];
  const double copied_pressure = copy.pressures[0];
  result->vertices_A[0] += Vector3d::Ones();
  result->pressures[0] += 1.0;
  EXPECT_TRUE(CompareMatrices(copy.vertices_A[0], copied_vertex));
  EXPECT_EQ(copy.pressures[0], copied_pressure);

  const FieldPair fields = MakeFieldsForPlane(Vector3d::UnitX(), 0.0, 2.0);
  const auto repeated = CalcVoxelSdfContactPolygon(Vector3d(1.0, 2.0, 3.0), 2.0,
                                                   fields.A, fields.B_A);
  ASSERT_TRUE(repeated.has_value());
  EXPECT_NE(repeated->vertices_A.data(), result->vertices_A.data());
  EXPECT_NE(repeated->pressures.data(), result->pressures.data());
}

GTEST_TEST(VoxelSdfContactSurfaceTest, SeparatedAndTouchingBoxes) {
  const VoxelSdfGeometry A(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const VoxelSdfGeometry B(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA;

  const RigidTransformd X_WB_separated(Vector3d(3.0, 0.0, 0.0));
  EXPECT_EQ(
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB_separated, id_B),
      nullptr);

  const RigidTransformd X_WB_touching(Vector3d(2.0, 0.0, 0.0));
  const auto touching =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB_touching, id_B);
  ASSERT_NE(touching, nullptr);
  ExpectSurfaceInvariants(*touching, id_A, id_B);
  for (int v = 0; v < touching->num_vertices(); ++v) {
    EXPECT_NEAR(touching->poly_e_MN().EvaluateAtVertex(v), 0.0, 1e-12);
  }
}

GTEST_TEST(VoxelSdfContactSurfaceTest, FaceOverlapDepthsAndMultipleVoxels) {
  const VoxelSdfGeometry A(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const VoxelSdfGeometry B(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA;

  const auto shallow = CalcVoxelSdfCompliantContact(
      A, X_WA, id_A, B, RigidTransformd(Vector3d(1.8, 0.0, 0.0)), id_B);
  ASSERT_NE(shallow, nullptr);
  ExpectSurfaceInvariants(*shallow, id_A, id_B);
  EXPECT_GT(shallow->num_faces(), 1);

  const auto deeper = CalcVoxelSdfCompliantContact(
      A, X_WA, id_A, B, RigidTransformd(Vector3d(1.2, 0.0, 0.0)), id_B);
  ASSERT_NE(deeper, nullptr);
  ExpectSurfaceInvariants(*deeper, id_A, id_B);
  EXPECT_GT(deeper->num_faces(), 1);
}

GTEST_TEST(VoxelSdfContactSurfaceTest, TraversalGeometryMayHaveHigherId) {
  const VoxelSdfGeometry coarse_A(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const VoxelSdfGeometry fine_B(Box(2.0, 2.0, 2.0), 0.25, 100.0);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA;
  const RigidTransformd X_WB(Vector3d(1.4, 0.0, 0.0));

  const auto coarse_surface =
      CalcVoxelSdfCompliantContact(coarse_A, X_WA, id_A, fine_B, X_WB, id_B);
  const auto fine_surface =
      CalcVoxelSdfCompliantContact(fine_B, X_WB, id_B, coarse_A, X_WA, id_A);
  ASSERT_NE(coarse_surface, nullptr);
  ASSERT_NE(fine_surface, nullptr);
  ExpectSurfaceInvariants(*coarse_surface, id_A, id_B);
  ExpectSurfaceInvariants(*fine_surface, id_A, id_B);
  EXPECT_GT(fine_surface->num_faces(), coarse_surface->num_faces());
}

GTEST_TEST(VoxelSdfContactSurfaceTest, EdgeCornerAndRotatedOverlap) {
  const VoxelSdfGeometry A(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const VoxelSdfGeometry B(Box(2.0, 2.0, 2.0), 0.5, 100.0);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA;

  const auto edge = CalcVoxelSdfCompliantContact(
      A, X_WA, id_A, B, RigidTransformd(Vector3d(1.8, 1.8, 0.0)), id_B);
  ASSERT_NE(edge, nullptr);
  ExpectSurfaceInvariants(*edge, id_A, id_B);

  const auto corner = CalcVoxelSdfCompliantContact(
      A, X_WA, id_A, B, RigidTransformd(Vector3d(1.8, 1.8, 1.8)), id_B);
  ASSERT_NE(corner, nullptr);
  ExpectSurfaceInvariants(*corner, id_A, id_B);

  const RigidTransformd X_WB_rotated(RotationMatrixd::MakeZRotation(0.35),
                                     Vector3d(1.5, 0.1, 0.0));
  const auto rotated =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB_rotated, id_B);
  ASSERT_NE(rotated, nullptr);
  ExpectSurfaceInvariants(*rotated, id_A, id_B);
}

GTEST_TEST(VoxelSdfContactSurfaceTest, UnequalGeometryAndWorldPose) {
  const VoxelSdfGeometry A(Box(2.0, 3.0, 4.0), 0.5, 120.0);
  const VoxelSdfGeometry B(Box(1.5, 2.5, 3.5), 0.3, 275.0);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA(RotationMatrixd::MakeZRotation(0.4),
                             Vector3d(1.0, 2.0, 3.0));
  const RigidTransformd X_AB(Vector3d(1.4, 0.1, -0.1));
  const RigidTransformd X_WB = X_WA * X_AB;

  const auto surface_A =
      CalcVoxelSdfCompliantContact(A, RigidTransformd(), id_A, B, X_AB, id_B);
  const auto surface_W =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB, id_B);
  ASSERT_NE(surface_A, nullptr);
  ASSERT_NE(surface_W, nullptr);
  ExpectSurfaceInvariants(*surface_A, id_A, id_B);
  ExpectSurfaceInvariants(*surface_W, id_A, id_B);
  ASSERT_EQ(surface_W->num_faces(), surface_A->num_faces());
  ASSERT_EQ(surface_W->num_vertices(), surface_A->num_vertices());
  EXPECT_GT(surface_W->num_faces(), 1);

  // Roundoff in the recomputed relative pose can change a polygon's cyclic
  // starting vertex. Compare the transformed vertex-and-pressure multiset,
  // independent of that inconsequential ordering choice.
  std::vector<bool> matched_W(surface_W->num_vertices(), false);
  for (int v_A = 0; v_A < surface_A->num_vertices(); ++v_A) {
    const Vector3d expected_W = X_WA * surface_A->poly_mesh_W().vertex(v_A);
    const double expected_pressure =
        surface_A->poly_e_MN().EvaluateAtVertex(v_A);
    bool found = false;
    for (int v_W = 0; v_W < surface_W->num_vertices(); ++v_W) {
      if (matched_W[v_W]) continue;
      if ((surface_W->poly_mesh_W().vertex(v_W) - expected_W).norm() <= 1e-12 &&
          std::abs(surface_W->poly_e_MN().EvaluateAtVertex(v_W) -
                   expected_pressure) <= 1e-12) {
        matched_W[v_W] = true;
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
  for (int f = 0; f < surface_W->num_faces(); ++f) {
    EXPECT_TRUE(CompareMatrices(surface_W->face_normal(f),
                                X_WA.rotation() * surface_A->face_normal(f),
                                1e-13));
    EXPECT_TRUE(CompareMatrices(
        surface_W->EvaluateGradE_M_W(f),
        X_WA.rotation() * surface_A->EvaluateGradE_M_W(f), 1e-13));
    EXPECT_TRUE(CompareMatrices(
        surface_W->EvaluateGradE_N_W(f),
        X_WA.rotation() * surface_A->EvaluateGradE_N_W(f), 1e-13));
  }
}

GTEST_TEST(VoxelSdfContactSurfaceTest, SurfaceOwnershipAndEngineCopy) {
  const Box box(2.0, 2.0, 2.0);
  ProximityProperties properties;
  AddCompliantHydroelasticVoxelSdfProperties(0.5, 100.0, &properties);
  const auto [id_A, id_B] = MakeOrderedGeometryIds();
  const RigidTransformd X_WA;
  const RigidTransformd X_WB(Vector3d(1.5, 0.0, 0.0));

  ProximityEngine<double> engine;
  engine.AddDynamicGeometry(box, X_WA, id_A, properties);
  engine.AddDynamicGeometry(box, X_WB, id_B, properties);
  ProximityEngine<double> engine_copy(engine);

  const VoxelSdfGeometry& A =
      engine.hydroelastic_geometries().compliant_geometry(id_A).voxel_sdf();
  const VoxelSdfGeometry& B =
      engine.hydroelastic_geometries().compliant_geometry(id_B).voxel_sdf();
  const VoxelSdfGeometry& copied_A = engine_copy.hydroelastic_geometries()
                                         .compliant_geometry(id_A)
                                         .voxel_sdf();
  const VoxelSdfGeometry& copied_B = engine_copy.hydroelastic_geometries()
                                         .compliant_geometry(id_B)
                                         .voxel_sdf();
  EXPECT_NE(&A.sample(0, 0, 0), &copied_A.sample(0, 0, 0));
  EXPECT_NE(&B.sample(0, 0, 0), &copied_B.sample(0, 0, 0));

  const auto original_surface =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB, id_B);
  const auto copied_surface =
      CalcVoxelSdfCompliantContact(copied_A, X_WA, id_A, copied_B, X_WB, id_B);
  ASSERT_NE(original_surface, nullptr);
  ASSERT_NE(copied_surface, nullptr);
  ExpectSurfaceInvariants(*original_surface, id_A, id_B);
  ExpectSurfaceInvariants(*copied_surface, id_A, id_B);
  EXPECT_TRUE(original_surface->Equal(*copied_surface));
  EXPECT_NE(&original_surface->poly_mesh_W(), &copied_surface->poly_mesh_W());
  EXPECT_NE(
      static_cast<const void*>(&original_surface->poly_mesh_W().vertex(0)),
      static_cast<const void*>(&A.sample(0, 0, 0)));

  const auto repeated_surface =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB, id_B);
  ASSERT_NE(repeated_surface, nullptr);
  EXPECT_TRUE(original_surface->Equal(*repeated_surface));
  EXPECT_NE(&original_surface->poly_mesh_W(), &repeated_surface->poly_mesh_W());

  const auto* sample_address = &A.sample(0, 0, 0);
  const double sample_value = sample_address->value;
  auto temporary_surface =
      CalcVoxelSdfCompliantContact(A, X_WA, id_A, B, X_WB, id_B);
  ASSERT_NE(temporary_surface, nullptr);
  temporary_surface.reset();
  EXPECT_EQ(&A.sample(0, 0, 0), sample_address);
  EXPECT_EQ(A.sample(0, 0, 0).value, sample_value);
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
