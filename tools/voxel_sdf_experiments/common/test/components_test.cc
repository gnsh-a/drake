#include "drake/tools/voxel_sdf_experiments/common/components.h"

#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

int CountComponents(const std::vector<int>& ids) {
  return std::set<int>(ids.begin(), ids.end()).size();
}

/* Two triangles sharing vertices 0 and 2 by index. */
SurfaceView MakeSharedIndexPair() {
  return SurfaceView{.num_vertices = 4,
                     .vertices_W = {{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {0.0, 1.0, 0.0}},
                     .faces = {
                         Face{.vertex_indices = {0, 1, 2}},
                         Face{.vertex_indices = {2, 3, 0}},
                     }};
}

/* The same two triangles, but with the shared edge duplicated so no vertex
 index is reused. This is how a per-cell representation emits faces. */
SurfaceView MakeDuplicatedEdgePair(double gap) {
  return SurfaceView{.num_vertices = 6,
                     .vertices_W = {{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {1.0, 1.0, gap},
                                    {0.0, 1.0, gap},
                                    {0.0, 0.0, gap}},
                     .faces = {
                         Face{.vertex_indices = {0, 1, 2}},
                         Face{.vertex_indices = {3, 4, 5}},
                     }};
}

GTEST_TEST(ComponentsTest, SharedVertexIndicesConnect) {
  const SurfaceView surface = MakeSharedIndexPair();
  const std::vector<int> ids =
      CalcFaceComponentIds(surface, DefaultComponentTolerance(surface));
  EXPECT_EQ(ids.size(), 2);
  EXPECT_EQ(CountComponents(ids), 1);
}

GTEST_TEST(ComponentsTest, CoincidentDuplicatedVerticesStillConnect) {
  // Duplicated but exactly coincident vertices must connect, because faces are
  // grouped by position rather than by index.
  const SurfaceView surface = MakeDuplicatedEdgePair(0.0);
  const std::vector<int> ids =
      CalcFaceComponentIds(surface, DefaultComponentTolerance(surface));
  EXPECT_EQ(CountComponents(ids), 1);
}

GTEST_TEST(ComponentsTest, GapWiderThanToleranceSplits) {
  const SurfaceView surface = MakeDuplicatedEdgePair(1e-6);
  const std::vector<int> ids =
      CalcFaceComponentIds(surface, DefaultComponentTolerance(surface));
  EXPECT_EQ(CountComponents(ids), 2);
}

GTEST_TEST(ComponentsTest, ToleranceIsHonored) {
  // The same 1e-6 gap connects once the tolerance exceeds it, which is what
  // makes the tolerance the knob that decides "touching".
  const SurfaceView surface = MakeDuplicatedEdgePair(1e-6);
  EXPECT_EQ(CountComponents(CalcFaceComponentIds(surface, 1e-9)), 2);
  EXPECT_EQ(CountComponents(CalcFaceComponentIds(surface, 1e-4)), 1);
}

GTEST_TEST(ComponentsTest, DefaultToleranceScalesWithCoordinates) {
  SurfaceView near_origin = MakeSharedIndexPair();
  SurfaceView far_away = MakeSharedIndexPair();
  for (Eigen::Vector3d& vertex_W : far_away.vertices_W) {
    vertex_W += Eigen::Vector3d(1000.0, 0.0, 0.0);
  }
  EXPECT_GT(DefaultComponentTolerance(far_away),
            DefaultComponentTolerance(near_origin));
}

GTEST_TEST(ComponentsTest, NonPositiveToleranceThrows) {
  const SurfaceView surface = MakeSharedIndexPair();
  EXPECT_THROW(CalcFaceComponentIds(surface, 0.0), std::logic_error);
  EXPECT_THROW(CalcFaceComponentIds(surface, -1.0), std::logic_error);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
