#pragma once

#include <cstddef>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* Immutable registered geometry data for a Box voxel signed-distance field.
 Grid coordinates and gradients are expressed in the Box geometry frame. */
class VoxelSdfGeometry {
 public:
  struct SdfSample {
    double value{};
    Vector3<double> gradient{};
  };

  VoxelSdfGeometry(const Box& box, double voxel_width,
                   double hydroelastic_modulus);

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VoxelSdfGeometry);

  const Vector3<double>& half_widths() const { return half_widths_; }
  double voxel_width() const { return voxel_width_; }
  double hydroelastic_modulus() const { return hydroelastic_modulus_; }
  double characteristic_length() const { return characteristic_length_; }
  double pressure_scale() const { return pressure_scale_; }
  const Vector3<int>& cell_counts() const { return cell_counts_; }
  const Vector3<double>& lower_cell_boundary() const {
    return lower_cell_boundary_;
  }

  Vector3<double> cell_center(int i, int j, int k) const;
  const SdfSample& sample(int i, int j, int k) const;

 private:
  size_t linear_index(int i, int j, int k) const;

  Vector3<double> half_widths_;
  double voxel_width_{};
  double hydroelastic_modulus_{};
  double characteristic_length_{};
  double pressure_scale_{};
  Vector3<int> cell_counts_;
  Vector3<double> lower_cell_boundary_;
  // Samples are immutable registered geometry data, indexed with x varying
  // fastest, then y, then z.
  std::vector<SdfSample> samples_;
};

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
