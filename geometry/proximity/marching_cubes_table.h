#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* Classic marching-cubes corner offsets. Corner bits are ordered with x
 varying fastest, then y, then z, matching VoxelSdfGeometry's lattice layout. */
inline constexpr std::array<std::array<int, 3>, 8> kMcCornerOffsets{{
    {{0, 0, 0}},
    {{1, 0, 0}},
    {{0, 1, 0}},
    {{1, 1, 0}},
    {{0, 0, 1}},
    {{1, 0, 1}},
    {{0, 1, 1}},
    {{1, 1, 1}},
}};

/* Endpoints of the twelve cube edges, expressed as corner indices. */
inline constexpr std::array<std::array<int, 2>, 12> kMcEdgeEndpoints{{
    {{0, 1}},
    {{1, 3}},
    {{2, 3}},
    {{0, 2}},
    {{4, 5}},
    {{5, 7}},
    {{6, 7}},
    {{4, 6}},
    {{0, 4}},
    {{1, 5}},
    {{2, 6}},
    {{3, 7}},
}};

/* Returns the immutable, static-lifetime, 16-entry classic marching-cubes row
 for `case_index`. Each row is a sequence of edge-index triples followed by
 one or more -1 sentinels. */
std::span<const int> McTriangles(uint8_t case_index);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
