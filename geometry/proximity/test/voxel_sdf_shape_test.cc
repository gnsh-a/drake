#include "drake/geometry/proximity/voxel_sdf_shape.h"

#include <algorithm>
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
static_assert(std::is_constructible_v<VoxelSdfShape, const Sphere&>);
static_assert(!std::is_constructible_v<VoxelSdfShape, const Cylinder&>);

GTEST_TEST(VoxelSdfShapeTest, BoxEvaluationAndGeometryData) {
  const Box box(2.0, 4.0, 6.0);
  const VoxelSdfShape dut(box);
  EXPECT_EQ(dut.shape_name(), "Box");
  EXPECT_TRUE(
      CompareMatrices(dut.bounding_box_half_widths(), Vector3d(1.0, 2.0, 3.0)));
  EXPECT_EQ(dut.characteristic_length(), 1.0);

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

GTEST_TEST(VoxelSdfShapeTest, SphereEvaluationAndGeometryData) {
  const Sphere sphere(2.5);
  const VoxelSdfShape dut(sphere);
  EXPECT_EQ(dut.shape_name(), "Sphere");
  EXPECT_TRUE(
      CompareMatrices(dut.bounding_box_half_widths(), Vector3d::Constant(2.5)));
  EXPECT_EQ(dut.characteristic_length(), 2.5);

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
    max_value = std::max(max_value, branch.sample.value);
    const bool is_active = std::all_of(
        branch.active_region.begin(), branch.active_region.end(),
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
    const bool is_active = std::all_of(
        branch.active_region.begin(), branch.active_region.end(),
        [&p_GC](const auto& half_space) {
          return half_space.Evaluate(p_GC) <= 0.0;
        });
    if (is_active) active.push_back(branch.index);
  }
  EXPECT_EQ(active, std::vector<int>({0, 2}));
}

GTEST_TEST(VoxelSdfShapeTest, SphereHasOneAffineBranch) {
  const VoxelSdfShape dut(Sphere(2.5));
  const Vector3d p_GQ(1.0, 0.5, -0.25);
  const auto branches = dut.CalcAffineBranches(p_GQ);
  ASSERT_EQ(branches.size(), 1u);
  EXPECT_TRUE(branches[0].active_region.empty());
  EXPECT_EQ(branches[0].index, 0);
  EXPECT_EQ(branches[0].sample.value, dut.Evaluate(p_GQ).value);
  EXPECT_TRUE(CompareMatrices(branches[0].sample.gradient,
                              dut.Evaluate(p_GQ).gradient));
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
