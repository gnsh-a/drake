#include "drake/geometry/proximity/voxel_sdf_geometry.h"

#include <cmath>
#include <limits>
#include <utility>

#include <fcl/fcl.h>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;

GTEST_TEST(VoxelSdfGeometryTest, GridConstruction) {
  const VoxelSdfGeometry exact(Box(2.0, 3.0, 4.0), 1.0, 12.0);
  EXPECT_TRUE(CompareMatrices(exact.half_widths(), Vector3d(1.0, 1.5, 2.0)));
  EXPECT_TRUE(CompareMatrices(exact.cell_counts(), Vector3<int>(2, 3, 4)));
  EXPECT_TRUE(
      CompareMatrices(exact.lower_cell_boundary(), Vector3d(-1.0, -1.5, -2.0)));
  EXPECT_EQ(exact.voxel_width(), 1.0);
  EXPECT_EQ(exact.hydroelastic_modulus(), 12.0);
  EXPECT_EQ(exact.characteristic_length(), 1.0);
  EXPECT_EQ(exact.pressure_scale(), 12.0);

  // Fractional ratios are rounded up, padding remains symmetric, and an axis
  // narrower than one voxel still receives one cell.
  const VoxelSdfGeometry fractional(Box(2.1, 0.4, 4.01), 1.0, 10.0);
  EXPECT_TRUE(CompareMatrices(fractional.cell_counts(), Vector3<int>(3, 1, 5)));
  EXPECT_TRUE(CompareMatrices(fractional.lower_cell_boundary(),
                              Vector3d(-1.5, -0.5, -2.5)));

  EXPECT_TRUE(
      CompareMatrices(exact.cell_center(0, 0, 0), Vector3d(-0.5, -1.0, -1.5)));
  EXPECT_TRUE(
      CompareMatrices(exact.cell_center(1, 2, 3), Vector3d(0.5, 1.0, 1.5)));

  // The linear layout is x-fast, then y, then z.
  const auto* first = &exact.sample(0, 0, 0);
  EXPECT_EQ(&exact.sample(1, 0, 0), first + 1);
  EXPECT_EQ(&exact.sample(0, 1, 0), first + 2);
  EXPECT_EQ(&exact.sample(0, 0, 1), first + 6);
}

GTEST_TEST(VoxelSdfGeometryTest, CachedSamplesMatchPointDistance) {
  const Box box(5.0, 5.0, 5.0);
  const VoxelSdfGeometry dut(box, 1.0, 1e7);
  const fcl::Boxd fcl_box(box.width(), box.depth(), box.height());

  for (int k = 0; k < dut.cell_counts()[2]; ++k) {
    for (int j = 0; j < dut.cell_counts()[1]; ++j) {
      for (int i = 0; i < dut.cell_counts()[0]; ++i) {
        const Vector3d p_GQ = dut.cell_center(i, j, k);
        point_distance::DistanceToPoint<double> query(
            GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
        const SignedDistanceToPoint<double> expected = query(fcl_box);
        const auto& actual = dut.sample(i, j, k);
        EXPECT_TRUE(std::isfinite(actual.value));
        EXPECT_TRUE(actual.gradient.array().isFinite().all());
        EXPECT_NEAR(actual.value, expected.distance, 1e-14);
        EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W, 1e-14));
      }
    }
  }

  // Explicit point-distance queries exercise the same Box feature rules used
  // when caching samples: center/medial ties, interior face regions, boundary
  // faces/edges/corners, and exterior face/edge/corner regions.
  for (const Vector3d& p_GQ :
       {Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 0.0, 0.0),
        Vector3d(2.5, 0.0, 0.0), Vector3d(2.5, 2.5, 0.0),
        Vector3d(2.5, 2.5, 2.5), Vector3d(3.5, 0.0, 0.0),
        Vector3d(3.5, 3.5, 0.0), Vector3d(3.5, 3.5, 3.5)}) {
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const auto result = query(fcl_box);
    EXPECT_TRUE(std::isfinite(result.distance));
    EXPECT_TRUE(result.grad_W.array().isFinite().all());
  }
}

GTEST_TEST(VoxelSdfGeometryTest, CopyAndMoveOwnership) {
  const VoxelSdfGeometry original(Box(2.0, 3.0, 4.0), 1.0, 12.0);

  VoxelSdfGeometry copy_constructed(original);
  EXPECT_NE(&copy_constructed.sample(0, 0, 0), &original.sample(0, 0, 0));
  EXPECT_EQ(copy_constructed.sample(0, 0, 0).value,
            original.sample(0, 0, 0).value);

  VoxelSdfGeometry copy_assigned(Box(1.0, 1.0, 1.0), 1.0, 1.0);
  copy_assigned = original;
  EXPECT_NE(&copy_assigned.sample(0, 0, 0), &original.sample(0, 0, 0));
  EXPECT_TRUE(
      CompareMatrices(copy_assigned.cell_counts(), original.cell_counts()));

  const auto* moved_data = &copy_constructed.sample(0, 0, 0);
  VoxelSdfGeometry move_constructed(std::move(copy_constructed));
  EXPECT_EQ(&move_constructed.sample(0, 0, 0), moved_data);
  EXPECT_EQ(move_constructed.hydroelastic_modulus(), 12.0);

  VoxelSdfGeometry move_assigned(Box(1.0, 1.0, 1.0), 1.0, 1.0);
  moved_data = &copy_assigned.sample(0, 0, 0);
  move_assigned = std::move(copy_assigned);
  EXPECT_EQ(&move_assigned.sample(0, 0, 0), moved_data);
  EXPECT_EQ(move_assigned.voxel_width(), 1.0);
}

GTEST_TEST(VoxelSdfGeometryTest, RejectsInvalidOrOverflowingGrids) {
  const Box box(1.0, 1.0, 1.0);
  for (double bad_width : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(box, bad_width, 1.0),
                                ".*width.*finite.*positive.*");
  }
  for (double bad_modulus :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(box, 1.0, bad_modulus),
                                ".*modulus.*finite.*positive.*");
  }

  const double too_many =
      static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(box, 1.0 / too_many, 1.0),
                              ".*too many cells.*");

  const double max_count = static_cast<double>(std::numeric_limits<int>::max());
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Box(max_count, max_count, max_count), 1.0, 1.0),
      ".*sample count overflows.*");
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
