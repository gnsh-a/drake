#include "drake/geometry/proximity/voxel_sdf_contact_common.h"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "drake/geometry/shape_specification.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using math::RigidTransformd;

GTEST_TEST(VoxelSdfContactCommonTest, FullRangesUseCompleteLattices) {
  const VoxelSdfGeometry sphere(Sphere(0.1), 0.01, 1.0e8);
  const VoxelSdfIndexRange cells = MakeFullVoxelSdfIndexRange(
      sphere, VoxelSdfTraversalGrid::kPlaneClipCells);
  EXPECT_EQ(cells.begin, Vector3<int>::Zero());
  EXPECT_EQ(cells.end, Vector3<int>::Constant(20));
  EXPECT_EQ(cells.num_elements(), 8000);

  const VoxelSdfIndexRange cubes =
      MakeFullVoxelSdfIndexRange(sphere, VoxelSdfTraversalGrid::kMarchingCubes);
  EXPECT_EQ(cubes.begin, Vector3<int>::Zero());
  EXPECT_EQ(cubes.end, Vector3<int>::Constant(21));
  EXPECT_EQ(cubes.num_elements(), 9261);
}

GTEST_TEST(VoxelSdfContactCommonTest, SphereBoxCandidateCounts) {
  struct TestCase {
    double voxel_width{};
    Vector3<int> cell_end;
    Vector3<int> cube_end;
  };
  const std::array<TestCase, 4> cases{{
      {0.01, Vector3<int>(20, 20, 3), Vector3<int>(21, 21, 3)},
      {0.008, Vector3<int>(25, 25, 3), Vector3<int>(26, 26, 4)},
      {0.005, Vector3<int>(40, 40, 5), Vector3<int>(41, 41, 5)},
      {0.0025, Vector3<int>(80, 80, 9), Vector3<int>(81, 81, 9)},
  }};
  for (const TestCase& test : cases) {
    SCOPED_TRACE(test.voxel_width);
    const VoxelSdfGeometry sphere(Sphere(0.1), test.voxel_width, 1.0e8);
    const VoxelSdfGeometry box(Box(0.4, 0.4, 0.2), test.voxel_width, 1.0e8);
    const RigidTransformd X_WS(Vector3<double>(0.0, 0.0, 0.2 - 0.0199));
    const RigidTransformd X_SB = X_WS.inverse();

    const VoxelSdfIndexRange cells = CalcVoxelSdfCandidateRange(
        sphere, box, X_SB, VoxelSdfTraversalGrid::kPlaneClipCells);
    EXPECT_EQ(cells.begin, Vector3<int>::Zero());
    EXPECT_EQ(cells.end, test.cell_end);

    const VoxelSdfIndexRange cubes = CalcVoxelSdfCandidateRange(
        sphere, box, X_SB, VoxelSdfTraversalGrid::kMarchingCubes);
    EXPECT_EQ(cubes.begin, Vector3<int>::Zero());
    EXPECT_EQ(cubes.end, test.cube_end);
  }
}

GTEST_TEST(VoxelSdfContactCommonTest, SeparatedPairHasEmptyRange) {
  const VoxelSdfGeometry sphere(Sphere(0.1), 0.005, 1.0e8);
  const VoxelSdfGeometry box(Box(0.4, 0.4, 0.2), 0.005, 1.0e8);
  const RigidTransformd X_SB(Vector3<double>(100.0, 0.0, 0.0));
  EXPECT_TRUE(CalcVoxelSdfCandidateRange(sphere, box, X_SB,
                                         VoxelSdfTraversalGrid::kPlaneClipCells)
                  .empty());
  EXPECT_TRUE(CalcVoxelSdfCandidateRange(sphere, box, X_SB,
                                         VoxelSdfTraversalGrid::kMarchingCubes)
                  .empty());
}

GTEST_TEST(VoxelSdfContactCommonTest,
           RotatedRangeContainsExpandedSupportCenters) {
  const VoxelSdfGeometry A(Sphere(1.0), 0.2, 100.0);
  const VoxelSdfGeometry B(Cylinder(1.0, 1.6), 0.4, 100.0);
  const RigidTransformd X_AB(math::RotationMatrixd::MakeYRotation(0.65),
                             Vector3<double>(1.3, 0.2, 0.1));
  const RigidTransformd X_BA = X_AB.inverse();
  const double halo =
      0.5 * std::sqrt(3.0) * A.voxel_width() +
      CalcSpatialTolerance(A.voxel_width(), A.characteristic_length(),
                           B.characteristic_length());
  const Vector3<double> expanded_half_width_B =
      -B.lower_cell_boundary() + Vector3<double>::Constant(halo);

  for (const VoxelSdfTraversalGrid grid :
       {VoxelSdfTraversalGrid::kPlaneClipCells,
        VoxelSdfTraversalGrid::kMarchingCubes}) {
    const VoxelSdfIndexRange candidate =
        CalcVoxelSdfCandidateRange(A, B, X_AB, grid);
    const VoxelSdfIndexRange full = MakeFullVoxelSdfIndexRange(A, grid);
    EXPECT_LT(candidate.num_elements(), full.num_elements());
    for (int k = 0; k < full.end.z(); ++k) {
      for (int j = 0; j < full.end.y(); ++j) {
        for (int i = 0; i < full.end.x(); ++i) {
          Vector3<double> center_A;
          if (grid == VoxelSdfTraversalGrid::kPlaneClipCells) {
            center_A = A.cell_center(i, j, k);
          } else {
            center_A = A.mc_node_position(i, j, k) +
                       Vector3<double>::Constant(0.5 * A.voxel_width());
          }
          const Vector3<double> center_B = X_BA * center_A;
          if ((center_B.cwiseAbs().array() <= expanded_half_width_B.array())
                  .all()) {
            EXPECT_GE(i, candidate.begin.x());
            EXPECT_LT(i, candidate.end.x());
            EXPECT_GE(j, candidate.begin.y());
            EXPECT_LT(j, candidate.end.y());
            EXPECT_GE(k, candidate.begin.z());
            EXPECT_LT(k, candidate.end.z());
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
