#include "drake/geometry/proximity/voxel_sdf_shape.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcl/fcl.h>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

static_assert(std::is_constructible_v<VoxelSdfShape, const Box&>);
static_assert(std::is_constructible_v<VoxelSdfShape, const Cylinder&>);
static_assert(std::is_constructible_v<VoxelSdfShape, const Ellipsoid&>);
static_assert(std::is_constructible_v<VoxelSdfShape, const Sphere&>);

GTEST_TEST(VoxelSdfShapeTest, BoxEvaluationAndGeometryData) {
  const Box box(2.0, 4.0, 6.0);
  const VoxelSdfShape dut(box);
  EXPECT_EQ(dut.shape_name(), "Box");
  EXPECT_TRUE(
      CompareMatrices(dut.bounding_box_half_widths(), Vector3d(1.0, 2.0, 3.0)));
  EXPECT_EQ(dut.characteristic_length(), 1.0);
  EXPECT_TRUE(dut.supports_sampled_trilinear());

  const fcl::Boxd fcl_box(box.width(), box.depth(), box.height());
  for (const Vector3d& p_GQ :
       {Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0),
        Vector3d(-1.0, 0.0, 0.0), Vector3d(1.0, 2.0, 3.0),
        Vector3d(2.0, 3.0, 4.0)}) {
    const VoxelSdfShape::Sample actual = dut.Evaluate(p_GQ);
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const SignedDistanceToPoint<double> expected = query(fcl_box);
    EXPECT_EQ(actual.value, expected.distance);
    EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W));
  }
}

GTEST_TEST(VoxelSdfShapeTest, CylinderEvaluationAndGeometryData) {
  const Cylinder cylinder(2.0, 1.0);
  const VoxelSdfShape dut(cylinder);
  EXPECT_EQ(dut.shape_name(), "Cylinder");
  EXPECT_TRUE(
      CompareMatrices(dut.bounding_box_half_widths(), Vector3d(2.0, 2.0, 0.5)));
  EXPECT_EQ(dut.characteristic_length(), 0.5);
  EXPECT_FALSE(dut.supports_sampled_trilinear());

  // VoxelSdfShape deliberately delegates the exact value and gradient to the
  // same primitive-distance implementation used by SceneGraph. Exercise the
  // centerline convention, the cap and wall interiors, the rim, and an
  // exterior point where the distance is not affine.
  const fcl::Cylinderd fcl_cylinder(cylinder.radius(), cylinder.length());
  for (const Vector3d& p_GQ :
       {Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 0.0, 0.0),
        Vector3d(0.0, 0.0, 0.25), Vector3d(2.0, 0.0, 0.5),
        Vector3d(3.0, 0.0, 1.5)}) {
    const VoxelSdfShape::Sample actual = dut.Evaluate(p_GQ);
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const SignedDistanceToPoint<double> expected = query(fcl_cylinder);
    EXPECT_EQ(actual.value, expected.distance);
    EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W));
  }
}

GTEST_TEST(VoxelSdfShapeTest, EllipsoidEvaluationAndGeometryData) {
  const Ellipsoid ellipsoid(1.5, 0.75, 1.25);
  const VoxelSdfShape dut(ellipsoid);
  EXPECT_EQ(dut.shape_name(), "Ellipsoid");
  EXPECT_TRUE(CompareMatrices(dut.bounding_box_half_widths(),
                              Vector3d(1.5, 0.75, 1.25)));
  EXPECT_EQ(dut.characteristic_length(), 0.75);
  EXPECT_FALSE(dut.supports_sampled_trilinear());

  // VoxelSdfShape deliberately shares SceneGraph's FCL-backed Ellipsoid
  // distance implementation and its medial-axis conventions.
  const fcl::Ellipsoidd fcl_ellipsoid(ellipsoid.a(), ellipsoid.b(),
                                      ellipsoid.c());
  for (const Vector3d& p_GQ :
       {Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 0.0, 0.0),
        Vector3d(0.0, 0.5, 0.0), Vector3d(0.0, 0.0, 1.25),
        Vector3d(2.0, 1.0, 0.5)}) {
    const VoxelSdfShape::Sample actual = dut.Evaluate(p_GQ);
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const SignedDistanceToPoint<double> expected = query(fcl_ellipsoid);
    EXPECT_EQ(actual.value, expected.distance);
    EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W));
  }

  // Exercise accuracy near a non-axis point on the surface. The tolerances
  // match the existing point-to-Ellipsoid characterization of FCL's GJK/EPA
  // implementation.
  constexpr double kDistTolerance = 5e-5;
  constexpr double kVectorTolerance = 5e-4;
  const double root_three = std::sqrt(3.0);
  const Vector3d p_GN(ellipsoid.a() / root_three, ellipsoid.b() / root_three,
                      ellipsoid.c() / root_three);
  const Vector3d normal = Vector3d(p_GN.x() / (ellipsoid.a() * ellipsoid.a()),
                                   p_GN.y() / (ellipsoid.b() * ellipsoid.b()),
                                   p_GN.z() / (ellipsoid.c() * ellipsoid.c()))
                              .normalized();
  for (const double distance : {-0.125, 0.0, 0.2}) {
    const VoxelSdfShape::Sample sample = dut.Evaluate(p_GN + distance * normal);
    EXPECT_NEAR(sample.value, distance, kDistTolerance);
    EXPECT_TRUE(CompareMatrices(sample.gradient, normal, kVectorTolerance));
  }

  const VoxelSdfShape::Sample center = dut.Evaluate(Vector3d::Zero());
  EXPECT_NEAR(center.value, -ellipsoid.b(), kDistTolerance);
  EXPECT_TRUE(CompareMatrices(center.gradient.cwiseAbs(), Vector3d::UnitY(),
                              kVectorTolerance));
}

GTEST_TEST(VoxelSdfShapeTest, SphereEvaluationAndGeometryData) {
  const Sphere sphere(2.5);
  const VoxelSdfShape dut(sphere);
  EXPECT_EQ(dut.shape_name(), "Sphere");
  EXPECT_TRUE(
      CompareMatrices(dut.bounding_box_half_widths(), Vector3d::Constant(2.5)));
  EXPECT_EQ(dut.characteristic_length(), 2.5);
  EXPECT_TRUE(dut.supports_sampled_trilinear());

  const fcl::Sphered fcl_sphere(sphere.radius());
  for (const Vector3d& p_GQ :
       {Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 0.0, 0.0),
        Vector3d(-1.0, 0.0, 0.0), Vector3d(0.0, 1.5, 2.0),
        Vector3d(2.0, 2.0, 2.0)}) {
    const VoxelSdfShape::Sample actual = dut.Evaluate(p_GQ);
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const SignedDistanceToPoint<double> expected = query(fcl_sphere);
    EXPECT_EQ(actual.value, expected.distance);
    EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W));
  }

  const auto center = dut.Evaluate(Vector3d::Zero());
  EXPECT_EQ(center.value, -sphere.radius());
  EXPECT_EQ(center.gradient, Vector3d::UnitX());
}

GTEST_TEST(VoxelSdfShapeTest, BoxAffineBranches) {
  const VoxelSdfShape dut(Box(2.0, 4.0, 6.0));
  const Vector3d p_GQ(0.75, 0.25, -0.5);
  const auto branches = dut.CalcAffineBranches(p_GQ);
  ASSERT_EQ(branches.size(), 6u);

  std::vector<int> active;
  double max_value = -std::numeric_limits<double>::infinity();
  for (const auto& branch : branches) {
    EXPECT_EQ(branch.active_region.size(), 5u);
    EXPECT_TRUE(branch.is_cell_invariant);
    max_value = std::max(max_value, branch.sample.value);
    const bool is_active =
        std::all_of(branch.active_region.begin(), branch.active_region.end(),
                    [&p_GQ](const auto& half_space) {
                      return half_space.Evaluate(p_GQ) <= 0.0;
                    });
    if (is_active) active.push_back(branch.index);
  }
  EXPECT_EQ(active, std::vector<int>({0}));
  EXPECT_EQ(max_value, dut.Evaluate(p_GQ).value);

  // At equal distance from +x and +y, both closed active regions contain the
  // query point. Stable branch indices let the contact kernel own this tie
  // without perturbing either region.
  const Vector3d p_GC(0.5, 1.5, 0.0);
  active.clear();
  for (const auto& branch : dut.CalcAffineBranches(p_GC)) {
    const bool is_active =
        std::all_of(branch.active_region.begin(), branch.active_region.end(),
                    [&p_GC](const auto& half_space) {
                      return half_space.Evaluate(p_GC) <= 0.0;
                    });
    if (is_active) active.push_back(branch.index);
  }
  EXPECT_EQ(active, std::vector<int>({0, 2}));
}

GTEST_TEST(VoxelSdfShapeTest, CylinderAffineBranches) {
  const VoxelSdfShape dut(Cylinder(2.0, 1.0));

  struct Case {
    Vector3d p_GQ;
    std::vector<int> active_indices;
  };
  // Indices are stable: top cap = 0, bottom cap = 1, radial wall = 2.
  // Include both cap-wall and cap-cap medial-axis ties.
  for (const Case& test :
       {Case{Vector3d(0.0, 0.0, 0.4), {0}}, Case{Vector3d(0.0, 0.0, -0.4), {1}},
        Case{Vector3d(1.9, 0.0, 0.0), {2}},
        Case{Vector3d(1.6, 0.0, 0.1), {0, 2}},
        Case{Vector3d::Zero(), {0, 1}}}) {
    const auto branches = dut.CalcAffineBranches(test.p_GQ);
    ASSERT_EQ(branches.size(), 3u);
    std::vector<int> active;
    double max_value = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(branches.size()); ++i) {
      const auto& branch = branches[i];
      EXPECT_EQ(branch.index, i);
      EXPECT_EQ(branch.active_region.size(), 2u);
      EXPECT_FALSE(branch.is_cell_invariant);
      max_value = std::max(max_value, branch.sample.value);
      const bool is_active =
          std::all_of(branch.active_region.begin(), branch.active_region.end(),
                      [&test](const auto& half_space) {
                        return half_space.Evaluate(test.p_GQ) <= 0.0;
                      });
      if (is_active) active.push_back(branch.index);
    }
    EXPECT_EQ(active, test.active_indices);
    // The max-of-features expression is the exact Cylinder SDF throughout
    // the interior (outside a rim it would be a Euclidean corner distance).
    EXPECT_EQ(max_value, dut.Evaluate(test.p_GQ).value);
  }

  // The radial wall has a deterministic finite direction on the centerline.
  const auto center_branches = dut.CalcAffineBranches(Vector3d::Zero());
  EXPECT_EQ(center_branches[2].sample.gradient, Vector3d::UnitX());
}

GTEST_TEST(VoxelSdfShapeTest, SphereHasOneAffineBranch) {
  const VoxelSdfShape dut(Sphere(2.5));
  const Vector3d p_GQ(1.0, 0.5, -0.25);
  const auto branches = dut.CalcAffineBranches(p_GQ);
  ASSERT_EQ(branches.size(), 1u);
  EXPECT_TRUE(branches[0].active_region.empty());
  EXPECT_EQ(branches[0].index, 0);
  EXPECT_FALSE(branches[0].is_cell_invariant);
  EXPECT_EQ(branches[0].sample.value, dut.Evaluate(p_GQ).value);
  EXPECT_TRUE(CompareMatrices(branches[0].sample.gradient,
                              dut.Evaluate(p_GQ).gradient));
}

GTEST_TEST(VoxelSdfShapeTest, EllipsoidHasOneAffineBranch) {
  const VoxelSdfShape dut(Ellipsoid(1.5, 0.75, 1.25));
  const Vector3d p_GQ(1.0, 0.5, -0.25);
  const auto branches = dut.CalcAffineBranches(p_GQ);
  ASSERT_EQ(branches.size(), 1u);
  EXPECT_TRUE(branches[0].active_region.empty());
  EXPECT_EQ(branches[0].index, 0);
  EXPECT_FALSE(branches[0].is_cell_invariant);
  EXPECT_EQ(branches[0].sample.value, dut.Evaluate(p_GQ).value);
  EXPECT_TRUE(CompareMatrices(branches[0].sample.gradient,
                              dut.Evaluate(p_GQ).gradient));

  const VoxelSdfShape::Sample cached{0.25, Vector3d::UnitZ()};
  const auto cached_branches = dut.CalcAffineBranches(p_GQ, cached);
  ASSERT_EQ(cached_branches.size(), 1u);
  EXPECT_EQ(cached_branches[0].sample.value, cached.value);
  EXPECT_EQ(cached_branches[0].sample.gradient, cached.gradient);
}

GTEST_TEST(VoxelSdfShapeTest, CopyAndMovePreserveValue) {
  const VoxelSdfShape original(Box(2.0, 4.0, 6.0));
  const Vector3d p_GQ(0.25, -0.5, 0.75);
  const VoxelSdfShape::Sample expected = original.Evaluate(p_GQ);

  VoxelSdfShape copied(original);
  EXPECT_EQ(copied.Evaluate(p_GQ).value, expected.value);
  EXPECT_TRUE(
      CompareMatrices(copied.Evaluate(p_GQ).gradient, expected.gradient));

  VoxelSdfShape assigned(Box::MakeCube(1.0));
  assigned = original;
  EXPECT_EQ(assigned.Evaluate(p_GQ).value, expected.value);

  VoxelSdfShape moved(std::move(copied));
  EXPECT_EQ(moved.Evaluate(p_GQ).value, expected.value);

  VoxelSdfShape move_assigned(Box::MakeCube(1.0));
  move_assigned = std::move(assigned);
  EXPECT_EQ(move_assigned.Evaluate(p_GQ).value, expected.value);
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
