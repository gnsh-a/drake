#include "drake/geometry/proximity/marching_cubes_table.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <span>

#include <gtest/gtest.h>

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

std::set<int> ReferencedEdges(uint8_t case_index) {
  const std::span<const int> row = McTriangles(case_index);
  std::set<int> result;
  for (int offset = 0; row[offset] != -1; ++offset) {
    result.insert(row[offset]);
  }
  return result;
}

GTEST_TEST(MarchingCubesTableTest, StructuralInvariants) {
  for (int case_value = 0; case_value < 256; ++case_value) {
    SCOPED_TRACE(case_value);
    const uint8_t case_index = static_cast<uint8_t>(case_value);
    const std::span<const int> row = McTriangles(case_index);
    ASSERT_EQ(row.size(), 16);
    EXPECT_EQ(row.data(), McTriangles(case_index).data());

    const auto sentinel = std::find(row.begin(), row.end(), -1);
    ASSERT_NE(sentinel, row.end());
    const int edge_count = static_cast<int>(sentinel - row.begin());
    EXPECT_EQ(edge_count % 3, 0);
    for (auto iter = sentinel; iter != row.end(); ++iter) {
      EXPECT_EQ(*iter, -1);
    }
    for (int i = 0; i < edge_count; ++i) {
      const int edge = row[i];
      ASSERT_GE(edge, 0);
      ASSERT_LT(edge, 12);
      const auto& endpoints = kMcEdgeEndpoints[edge];
      const bool endpoint0_is_set = (case_value & (1 << endpoints[0])) != 0;
      const bool endpoint1_is_set = (case_value & (1 << endpoints[1])) != 0;
      EXPECT_NE(endpoint0_is_set, endpoint1_is_set);
    }

    const uint8_t complement =
        static_cast<uint8_t>(255 - static_cast<int>(case_index));
    // Complementing a case preserves its crossing edges. Ambiguous classic-MC
    // cases need not choose the same triangulation or triangle count; v1 makes
    // no watertightness claim and corrects each retained triangle's winding
    // from grad_g_h instead of assuming complement-wise table orientation.
    EXPECT_EQ(ReferencedEdges(case_index), ReferencedEdges(complement));
  }

  EXPECT_TRUE(ReferencedEdges(0).empty());
  EXPECT_TRUE(ReferencedEdges(255).empty());
}

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
