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
  EXPECT_EQ(exact.EvaluateSdf(Vector3d(1.0, 0.0, 0.0)).value, 0.0);
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

GTEST_TEST(VoxelSdfGeometryTest, SampledGridStorage) {
  const Box box(2.0, 3.0, 4.0);
  const VoxelSdfGeometry primitive(box, 1.0, 12.0);
  const VoxelSdfGeometry sampled(box, 1.0, 12.0,
                                 VoxelSdfEvaluationMode::kSampledTrilinear);

  EXPECT_EQ(primitive.evaluation_mode(),
            VoxelSdfEvaluationMode::kPrimitiveAffine);
  EXPECT_EQ(sampled.evaluation_mode(),
            VoxelSdfEvaluationMode::kSampledTrilinear);
  EXPECT_TRUE(CompareMatrices(sampled.cell_counts(), primitive.cell_counts()));
  EXPECT_TRUE(CompareMatrices(sampled.lower_cell_boundary(),
                              primitive.lower_cell_boundary()));
  EXPECT_TRUE(
      CompareMatrices(sampled.storage_counts(),
                      sampled.cell_counts() + Vector3<int>::Constant(4)));
  EXPECT_TRUE(
      CompareMatrices(primitive.storage_counts(), primitive.cell_counts()));

  for (int k = 0; k < sampled.cell_counts()[2]; ++k) {
    for (int j = 0; j < sampled.cell_counts()[1]; ++j) {
      for (int i = 0; i < sampled.cell_counts()[0]; ++i) {
        EXPECT_TRUE(CompareMatrices(sampled.cell_center(i, j, k),
                                    primitive.cell_center(i, j, k)));
        EXPECT_EQ(&sampled.sample(i, j, k),
                  &sampled.stored_sample(i + 2, j + 2, k + 2));
      }
    }
  }

  EXPECT_TRUE(CompareMatrices(sampled.stored_sample_center(0, 0, 0),
                              Vector3d(-2.5, -3.0, -3.5)));
  const Vector3<int> last = sampled.storage_counts() - Vector3<int>::Ones();
  EXPECT_TRUE(
      CompareMatrices(sampled.stored_sample_center(last[0], last[1], last[2]),
                      Vector3d(2.5, 3.0, 3.5)));
  for (const Vector3<int>& index :
       {Vector3<int>(0, 0, 0), last, Vector3<int>(1, 3, 4)}) {
    const auto& padding = sampled.stored_sample(index[0], index[1], index[2]);
    EXPECT_TRUE(std::isfinite(padding.value));
    EXPECT_TRUE(padding.gradient.allFinite());
  }
}

GTEST_TEST(VoxelSdfGeometryTest, TrilinearInterpolation) {
  constexpr double radius = 2.5;
  const VoxelSdfGeometry sampled(Sphere(radius), 1.0, 10.0,
                                 VoxelSdfEvaluationMode::kSampledTrilinear);
  const auto result = sampled.InterpolateSdf(Vector3d(0.25, 0.5, 0.75));
  ASSERT_TRUE(result.has_value());

  // These are the analytical registration values at the eight corners of the
  // [0, 1]^3 interpolation cell.
  const double f000 = -radius;
  const double f100 = 1.0 - radius;
  const double f010 = f100;
  const double f001 = f100;
  const double f110 = std::sqrt(2.0) - radius;
  const double f101 = f110;
  const double f011 = f110;
  const double f111 = std::sqrt(3.0) - radius;
  constexpr double u = 0.25;
  constexpr double v = 0.5;
  constexpr double w = 0.75;
  const double expected_value =
      (1 - u) * (1 - v) * (1 - w) * f000 + u * (1 - v) * (1 - w) * f100 +
      (1 - u) * v * (1 - w) * f010 + u * v * (1 - w) * f110 +
      (1 - u) * (1 - v) * w * f001 + u * (1 - v) * w * f101 +
      (1 - u) * v * w * f011 + u * v * w * f111;
  const Vector3d expected_gradient(
      (1 - v) * (1 - w) * (f100 - f000) + v * (1 - w) * (f110 - f010) +
          (1 - v) * w * (f101 - f001) + v * w * (f111 - f011),
      (1 - u) * (1 - w) * (f010 - f000) + u * (1 - w) * (f110 - f100) +
          (1 - u) * w * (f011 - f001) + u * w * (f111 - f101),
      (1 - u) * (1 - v) * (f001 - f000) + u * (1 - v) * (f101 - f100) +
          (1 - u) * v * (f011 - f010) + u * v * (f111 - f110));
  EXPECT_NEAR(result->value, expected_value, 1e-14);
  EXPECT_TRUE(CompareMatrices(result->gradient, expected_gradient, 1e-14));

  // At an internal lattice plane, half-open ownership selects the interval
  // beginning at the stored sample.
  const Vector3<int> center_index(4, 4, 4);
  const Vector3d center = sampled.stored_sample_center(
      center_index[0], center_index[1], center_index[2]);
  const auto at_center = sampled.InterpolateSdf(center);
  ASSERT_TRUE(at_center.has_value());
  EXPECT_EQ(at_center->value, sampled.stored_sample(4, 4, 4).value);
  EXPECT_NEAR(at_center->gradient.x(),
              sampled.stored_sample(5, 4, 4).value -
                  sampled.stored_sample(4, 4, 4).value,
              1e-14);

  const Vector3<int> last = sampled.storage_counts() - Vector3<int>::Ones();
  const Vector3d final_center =
      sampled.stored_sample_center(last[0], last[1], last[2]);
  const auto at_final = sampled.InterpolateSdf(final_center);
  ASSERT_TRUE(at_final.has_value());
  EXPECT_EQ(at_final->value,
            sampled.stored_sample(last[0], last[1], last[2]).value);
  EXPECT_FALSE(sampled.InterpolateSdf(final_center + Vector3d(1e-12, 0.0, 0.0))
                   .has_value());
}

GTEST_TEST(VoxelSdfGeometryTest, SymmetricInterpolationHasZeroGradient) {
  const VoxelSdfGeometry sampled(Sphere(2.0), 1.0, 10.0,
                                 VoxelSdfEvaluationMode::kSampledTrilinear);
  const auto result = sampled.InterpolateSdf(Vector3d::Zero());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(std::isfinite(result->value));
  EXPECT_TRUE(result->gradient.allFinite());
  EXPECT_EQ(result->gradient, Vector3d::Zero());
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

  // Cell centers are strictly inside the Box. These targeted cached samples
  // cover interior face regions and Drake's selected gradients at medial-axis
  // ties; boundary and exterior behavior belongs to point-distance tests.
  struct SampleCase {
    Vector3<int> index;
    Vector3d center;
    double value;
    Vector3d gradient;
  };
  for (const SampleCase& test :
       {SampleCase{{2, 2, 2}, Vector3d::Zero(), -2.5, Vector3d::UnitX()},
        SampleCase{{4, 2, 2}, Vector3d(2, 0, 0), -0.5, Vector3d::UnitX()},
        SampleCase{{0, 2, 2}, Vector3d(-2, 0, 0), -0.5, -Vector3d::UnitX()},
        SampleCase{{4, 4, 2}, Vector3d(2, 2, 0), -0.5, Vector3d::UnitX()},
        SampleCase{{4, 4, 4}, Vector3d(2, 2, 2), -0.5, Vector3d::UnitX()}}) {
    const int i = test.index[0];
    const int j = test.index[1];
    const int k = test.index[2];
    const Vector3d p_GQ = dut.cell_center(i, j, k);
    EXPECT_TRUE(CompareMatrices(p_GQ, test.center));
    point_distance::DistanceToPoint<double> query(
        GeometryId::get_new_id(), math::RigidTransformd(), p_GQ);
    const SignedDistanceToPoint<double> expected = query(fcl_box);
    const auto& actual = dut.sample(i, j, k);
    EXPECT_EQ(actual.value, test.value);
    EXPECT_TRUE(CompareMatrices(actual.gradient, test.gradient));
    EXPECT_EQ(actual.value, expected.distance);
    EXPECT_TRUE(CompareMatrices(actual.gradient, expected.grad_W));
  }
}

GTEST_TEST(VoxelSdfGeometryTest, SphereGridAndCachedSamples) {
  const Sphere sphere(2.5);
  const VoxelSdfGeometry dut(sphere, 1.0, 10.0);
  EXPECT_TRUE(CompareMatrices(dut.cell_counts(), Vector3<int>(5, 5, 5)));
  EXPECT_TRUE(
      CompareMatrices(dut.lower_cell_boundary(), Vector3d::Constant(-2.5)));
  EXPECT_EQ(dut.characteristic_length(), 2.5);
  EXPECT_EQ(dut.pressure_scale(), 4.0);

  const auto& center = dut.sample(2, 2, 2);
  EXPECT_EQ(dut.cell_center(2, 2, 2), Vector3d::Zero());
  EXPECT_EQ(center.value, -2.5);
  EXPECT_EQ(center.gradient, Vector3d::UnitX());

  const auto& interior = dut.sample(3, 2, 2);
  EXPECT_EQ(interior.value, -1.5);
  EXPECT_EQ(interior.gradient, Vector3d::UnitX());

  const auto& corner = dut.sample(4, 4, 4);
  EXPECT_GT(corner.value, 0.0);
  EXPECT_TRUE(CompareMatrices(corner.gradient, Vector3d::Ones().normalized()));

  const VoxelSdfGeometry one_cell(Sphere(0.25), 1.0, 10.0);
  EXPECT_TRUE(CompareMatrices(one_cell.cell_counts(), Vector3<int>(1, 1, 1)));
  EXPECT_EQ(one_cell.cell_center(0, 0, 0), Vector3d::Zero());
  EXPECT_EQ(one_cell.sample(0, 0, 0).value, -0.25);
}

GTEST_TEST(VoxelSdfGeometryTest, AffineBranchAccess) {
  const VoxelSdfGeometry sphere(Sphere(2.5), 1.0, 10.0);
  const auto sphere_branches = sphere.CalcCellSdfBranches(2, 2, 2);
  ASSERT_EQ(sphere_branches.size(), 1u);
  EXPECT_EQ(sphere_branches[0].sample.value, sphere.sample(2, 2, 2).value);
  EXPECT_EQ(sphere_branches[0].sample.gradient,
            sphere.sample(2, 2, 2).gradient);

  const VoxelSdfGeometry box(Box(2.0, 4.0, 6.0), 1.0, 10.0);
  EXPECT_EQ(box.CalcCellSdfBranches(1, 2, 3).size(), 6u);
  EXPECT_EQ(box.EvaluateSdfBranches(Vector3d(0.25, 0.5, 0.75)).size(), 6u);

  const VoxelSdfGeometry sampled_box(Box(2.0, 4.0, 6.0), 1.0, 10.0,
                                     VoxelSdfEvaluationMode::kSampledTrilinear);
  const auto sampled_branches = sampled_box.CalcCellSdfBranches(1, 2, 3);
  ASSERT_EQ(sampled_branches.size(), 1u);
  EXPECT_EQ(sampled_branches[0].sample.value,
            sampled_box.sample(1, 2, 3).value);
  EXPECT_TRUE(sampled_branches[0].active_region.empty());
  EXPECT_EQ(sampled_branches[0].index, 0);
  EXPECT_FALSE(sampled_branches[0].is_cell_invariant);
}

GTEST_TEST(VoxelSdfGeometryTest, CopyAndMoveOwnership) {
  const VoxelSdfGeometry original(Box(2.0, 3.0, 4.0), 1.0, 12.0,
                                  VoxelSdfEvaluationMode::kSampledTrilinear);

  VoxelSdfGeometry copy_constructed(original);
  EXPECT_NE(&copy_constructed.sample(0, 0, 0), &original.sample(0, 0, 0));
  EXPECT_EQ(copy_constructed.sample(0, 0, 0).value,
            original.sample(0, 0, 0).value);
  EXPECT_EQ(copy_constructed.evaluation_mode(), original.evaluation_mode());

  VoxelSdfGeometry copy_assigned(Box(1.0, 1.0, 1.0), 1.0, 1.0);
  copy_assigned = original;
  EXPECT_NE(&copy_assigned.sample(0, 0, 0), &original.sample(0, 0, 0));
  EXPECT_TRUE(
      CompareMatrices(copy_assigned.cell_counts(), original.cell_counts()));

  const auto* moved_data = &copy_constructed.sample(0, 0, 0);
  VoxelSdfGeometry move_constructed(std::move(copy_constructed));
  EXPECT_EQ(&move_constructed.sample(0, 0, 0), moved_data);
  EXPECT_EQ(move_constructed.hydroelastic_modulus(), 12.0);
  EXPECT_EQ(move_constructed.evaluation_mode(),
            VoxelSdfEvaluationMode::kSampledTrilinear);

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

  const double max = std::numeric_limits<double>::max();
  const double denorm = std::numeric_limits<double>::denorm_min();
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(box, 1.0, max),
                              ".*pressure scale.*finite.*positive.*");
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(Box(max, max, max), max, denorm),
                              ".*pressure scale.*finite.*positive.*");
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(Box(denorm, 1.0, 1.0), 1.0, 1.0),
                              ".*characteristic length.*finite.*positive.*");
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(Sphere(0.5), 1.0, max),
                              ".*Sphere.*pressure scale.*finite.*positive.*");
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(Sphere(max), max, denorm),
                              ".*Sphere.*pressure scale.*finite.*positive.*");
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(Sphere(max), max, 1.0),
                              ".*Sphere.*too many cells.*");

  // Fused center arithmetic keeps representable coordinates finite even when
  // the multiplication h * (i + 0.5) would overflow before cancellation.
  const VoxelSdfGeometry extreme(Box(max, 1.0, 1.0), 0.75 * max, 1.0);
  EXPECT_TRUE(extreme.cell_center(0, 0, 0).allFinite());
  EXPECT_TRUE(extreme.cell_center(1, 0, 0).allFinite());
  EXPECT_TRUE(std::isfinite(extreme.sample(0, 0, 0).value));
  EXPECT_TRUE(std::isfinite(extreme.sample(1, 0, 0).value));

  const double too_many =
      static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
  DRAKE_EXPECT_THROWS_MESSAGE(VoxelSdfGeometry(box, 1.0 / too_many, 1.0),
                              ".*too many cells.*");
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Sphere(1.0), 1.0 / too_many, 1.0),
      ".*Sphere.*too many cells.*");

  const double max_count = static_cast<double>(std::numeric_limits<int>::max());
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Box(max_count, max_count, max_count), 1.0, 1.0),
      ".*sample count overflows.*");
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Box(max_count, 1.0, 1.0), 1.0, 1.0,
                       VoxelSdfEvaluationMode::kSampledTrilinear),
      ".*padded sample count overflows.*");
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Box(max_count - 4.0, max_count - 4.0, max_count - 4.0),
                       1.0, 1.0, VoxelSdfEvaluationMode::kSampledTrilinear),
      ".*sample count overflows.*");
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(Box(max, 1.0, 1.0), 0.75 * max, 1.0,
                       VoxelSdfEvaluationMode::kSampledTrilinear),
      ".*padded sample coordinates.*finite.*");
  DRAKE_EXPECT_THROWS_MESSAGE(
      VoxelSdfGeometry(box, 1.0, 1.0, static_cast<VoxelSdfEvaluationMode>(-1)),
      ".*evaluation mode.*invalid.*");
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
