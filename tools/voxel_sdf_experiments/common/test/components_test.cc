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

/* CalcComponentStats reports the count and the largest component's share of
 the area from one labelling pass. Both are checked together because neither is
 conclusive alone: a count that collapsed to one would leave the fraction
 reading a perfect 1.0. The faces carry deliberately unequal areas, so a
 fraction computed by counting faces instead of summing area would fail. */
GTEST_TEST(ComponentsTest, ComponentStatsCountAndShare) {
  SurfaceView joined = MakeSharedIndexPair();
  joined.faces[0].area = 3.0;
  joined.faces[1].area = 1.0;
  const ComponentStats joined_stats = CalcComponentStats(joined, 4.0);
  EXPECT_EQ(joined_stats.num_components, 1);
  EXPECT_NEAR(joined_stats.largest_area_fraction, 1.0, 1e-14);

  /* The same two faces pulled apart past the tolerance: two components, and
   the larger holds its own 3 of the 4 units of area rather than all of it. */
  SurfaceView split = MakeDuplicatedEdgePair(1.0);
  split.faces[0].area = 3.0;
  split.faces[1].area = 1.0;
  const ComponentStats split_stats = CalcComponentStats(split, 4.0);
  EXPECT_EQ(split_stats.num_components, 2);
  EXPECT_NEAR(split_stats.largest_area_fraction, 0.75, 1e-14);
}

/* An empty surface has nothing to label. Returning zeros rather than dividing
 by a zero total is what keeps a no-contact frame from poisoning a mean. */
GTEST_TEST(ComponentsTest, ComponentStatsOfAnEmptySurface) {
  const ComponentStats stats = CalcComponentStats(SurfaceView{}, 0.0);
  EXPECT_EQ(stats.num_components, 0);
  EXPECT_EQ(stats.largest_area_fraction, 0.0);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
