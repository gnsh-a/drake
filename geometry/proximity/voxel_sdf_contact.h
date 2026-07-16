#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/voxel_sdf_geometry.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* One affine signed-distance field at a voxel center. The quantities are
 expressed in the frame of the voxel. */
struct AffineSdfField {
  double value{};
  Vector3<double> gradient{};
  double pressure_scale{};
  double characteristic_length{};
};

/* The owned result of intersecting one voxel with the equal-pressure plane of
 two affine pressure fields. All vectors are expressed in frame A. */
struct VoxelSdfContactPolygon {
  std::vector<Vector3<double>> vertices_A;
  std::vector<double> pressures;
  Vector3<double> nhat_BA_A;
  Vector3<double> grad_p_A;
  Vector3<double> grad_p_B_A;
};

/* Calculates the contact polygon in one cubic voxel centered at `center_A`.

 Each selected branch defines an affine extension over this voxel. The Box
 branches are exact within their active regions, while a Sphere branch is a
 local approximation of its curved SDF. The returned normal points toward
 increasing A pressure and decreasing B pressure, i.e., out of B and into A.
 On the equal-pressure plane p_A = p_B, so clipping to nonnegative A pressure
 also clips to nonnegative B pressure.

 @param center_A       Voxel center, expressed in frame A.
 @param voxel_width    Edge length of the cubic voxel.
 @param sdf_A          A's SDF sample and scales at `center_A`.
 @param sdf_B_A        B's SDF sample at `center_A`, expressed in frame A.
 @returns No value if the equal-pressure plane is degenerate, misses the
          positive-pressure portion of the voxel, or produces negligible area.
 @pre `voxel_width`, both pressure scales, and both characteristic lengths are
      finite and strictly positive. All SDF values and gradients are finite.
 */
std::optional<VoxelSdfContactPolygon> CalcVoxelSdfContactPolygon(
    const Vector3<double>& center_A, double voxel_width,
    const AffineSdfField& sdf_A, const AffineSdfField& sdf_B_A);

/* Calculates a polygonal contact surface between two compliant voxel SDF
 representations. Geometry A's complete core grid is traversed, and all
 intermediate geometry is constructed in frame A before the result is
 transformed to World. Each geometry follows its own selected evaluation mode.

 The returned surface owns its mesh, pressure field, and constituent pressure
 gradients; it retains no references to either registered representation or
 pose. This calculator is double-only; SceneGraph dispatch supports it only for
 polygonal contact surfaces.

 Geometry A supplies the traversed grid and can have either the lower or higher
 GeometryId. The returned ContactSurface orders M and N by GeometryId.

 @returns nullptr if no A voxel produces a positive-area contact polygon.
 @pre When B uses sampled trilinear evaluation, A's voxel width is no greater
      than B's. ContactCalculator satisfies this for every dispatched voxel
      pair; direct callers must preserve the ordering.
 */
std::unique_ptr<ContactSurface<double>> CalcVoxelSdfCompliantContact(
    const VoxelSdfGeometry& A, const math::RigidTransformd& X_WA,
    GeometryId id_A, const VoxelSdfGeometry& B,
    const math::RigidTransformd& X_WB, GeometryId id_B);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
