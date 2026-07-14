#include "drake/geometry/proximity/voxel_sdf_contact.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

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

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
