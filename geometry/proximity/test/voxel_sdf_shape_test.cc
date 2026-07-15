#include "drake/geometry/proximity/voxel_sdf_shape.h"

#include <utility>

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
