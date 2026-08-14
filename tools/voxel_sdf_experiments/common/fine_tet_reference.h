#pragma once

#include <memory>

#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/tools/voxel_sdf_experiments/common/reference.h"

namespace drake {
namespace geometry {
template <typename T>
class ContactSurface;
}  // namespace geometry
namespace tools {
namespace voxel_sdf_experiments {

class FineTetReferenceTester;

/* A numerical reference built from Drake's tetrahedral hydroelastic contact
 surface at a deliberately fine resolution. All poses and query points are
 measured and expressed in a common reference frame R.

 The generated contact surface and its spatial index are owned by this object;
 the input shapes only need to remain alive for the duration of construction.
 force() is the norm of the integrated pressure traction, area() is the norm of
 the integrated area vector (the interface's projected-area quantity), and the
 centroid is weighted by true surface area. Pressure queries interpolate the
 field at the nearest point on the finite contact surface. */
class FineTetReference final : public Reference {
 public:
  FineTetReference(const geometry::Shape& shape_a,
                   const math::RigidTransformd& X_RA, double modulus_a,
                   const geometry::Shape& shape_b,
                   const math::RigidTransformd& X_RB, double modulus_b,
                   double tet_resolution_hint);
  ~FineTetReference() final;

  double force() const final;
  double area() const final;
  double distance_to_surface(const Eigen::Vector3d& p_RQ) const final;
  double pressure_at(const Eigen::Vector3d& p_RQ) const final;
  double patch_radius() const final;
  double peak_pressure() const final;
  Eigen::Vector3d centroid() const final;
  Eigen::Vector3d normal() const final;

 private:
  friend class FineTetReferenceTester;
  explicit FineTetReference(geometry::ContactSurface<double> surface);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
