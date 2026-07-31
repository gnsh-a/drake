#pragma once

#include <array>
#include <map>
#include <memory>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/voxel_sdf_geometry.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* Values at one dual-grid node used by the internal marching-cubes substrate.
 The position is measured and expressed in frame A. Pressures are scalars; this
 type owns its data and retains no reference to registered geometry. */
struct MarchingCubesNode {
  Vector3<double> p_AN_A;
  double pressure_A{};
  double pressure_B{};
};

/* Query-local mesh data ready for the Phase 4 constituent-gradient and
 ContactSurface finalization steps. Centroids correspond one-to-one and in
 order with the faces in builder_A. */
struct MarchingCubesMeshData {
  TriMeshBuilder<double> builder_A;
  std::vector<Vector3<double>> face_centroids_A;
};

/* Builds the retained marching-cubes triangles for one contact query.

 One automatic instance spans all cubes traversed in A. Its vertex caches map
 canonical value keys to integer builder indices only; they never point into
 builder storage that can reallocate. TakeMeshData() consumes the builder and
 centroids but deliberately leaves the caches behind to be destroyed with this
 query-local object. */
class MarchingCubesContactBuilder final {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MarchingCubesContactBuilder);

  explicit MarchingCubesContactBuilder(double voxel_width);

  /* Adds retained triangles for one dual-grid cube. `nodes_A` follows
   kMcCornerOffsets order. A case bit is set exactly when
   pressure_A - pressure_B < 0; exact zero is assigned to the unset side. */
  void AddCube(const Vector3<int>& cube_index,
               const std::array<MarchingCubesNode, 8>& nodes_A);

  /* Transfers the builder and ordered centroids. This may be called once and
   only as the final use of this object. */
  MarchingCubesMeshData TakeMeshData() &&;

 private:
  double voxel_width_{};
  bool consumed_{false};
  MarchingCubesMeshData mesh_data_;
  // {axis, lowest i, lowest j, lowest k} -> builder vertex index.
  std::map<std::array<int, 4>, int> edge_vertex_indices_;
  // Sorted pair of raw grid-edge keys -> clipped rim vertex index.
  std::map<std::array<std::array<int, 4>, 2>, int> boundary_vertex_indices_;
};

/* Calculates a triangular contact surface between two compliant voxel SDF
 representations using marching cubes. Geometry A's complete dual-grid cube
 lattice is traversed. All intermediate geometry is constructed in frame A,
 and B's primitive SDF is evaluated in frame B.

 The returned surface owns its mesh, pressure field, and constituent pressure
 gradients; it retains no references to either registered representation,
 query-local builder state, or pose. Geometry A can have either the lower or
 higher GeometryId; ContactSurface orders M and N by GeometryId.

 @returns nullptr if no cube produces a retained contact triangle.
 @pre Both geometries use primitive SDF evaluation and marching-cubes
      extraction. */
std::unique_ptr<ContactSurface<double>> CalcVoxelSdfMarchingCubesContact(
    const VoxelSdfGeometry& A, const math::RigidTransformd& X_WA,
    GeometryId id_A, const VoxelSdfGeometry& B,
    const math::RigidTransformd& X_WB, GeometryId id_B);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
