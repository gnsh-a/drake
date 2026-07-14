#include "drake/geometry/proximity/voxel_sdf_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

int CalcCellCount(double extent, double voxel_width, int axis) {
  const double count = std::ceil(extent / voxel_width);
  if (!std::isfinite(count) ||
      count > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::logic_error(fmt::format(
        "The Box voxel SDF requires too many cells on axis {}", axis));
  }
  return std::max(1, static_cast<int>(count));
}

size_t CheckedMultiply(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::logic_error("The Box voxel SDF sample count overflows size_t");
  }
  return a * b;
}

}  // namespace

VoxelSdfGeometry::VoxelSdfGeometry(const Box& box, double voxel_width,
                                   double hydroelastic_modulus)
    : half_widths_(box.size() / 2.0),
      voxel_width_(voxel_width),
      hydroelastic_modulus_(hydroelastic_modulus),
      characteristic_length_(half_widths_.minCoeff()),
      pressure_scale_(hydroelastic_modulus / characteristic_length_) {
  if (!(voxel_width > 0.0 && std::isfinite(voxel_width))) {
    throw std::logic_error(
        "The Box voxel SDF width must be finite and strictly positive");
  }
  if (!(hydroelastic_modulus > 0.0 && std::isfinite(hydroelastic_modulus))) {
    throw std::logic_error(
        "The Box voxel SDF hydroelastic modulus must be finite and strictly "
        "positive");
  }

  const Vector3<double> extent = 2.0 * half_widths_;
  for (int a = 0; a < 3; ++a) {
    cell_counts_[a] = CalcCellCount(extent[a], voxel_width_, a);
    // Centering the padded grid about the Box makes its padding symmetric.
    lower_cell_boundary_[a] = -0.5 * cell_counts_[a] * voxel_width_;
  }

  size_t sample_count = CheckedMultiply(static_cast<size_t>(cell_counts_[0]),
                                        static_cast<size_t>(cell_counts_[1]));
  sample_count =
      CheckedMultiply(sample_count, static_cast<size_t>(cell_counts_[2]));
  if (sample_count > samples_.max_size()) {
    throw std::logic_error(
        "The Box voxel SDF sample count cannot be represented by a vector");
  }
  try {
    samples_.resize(sample_count);
  } catch (const std::bad_alloc&) {
    throw std::logic_error(
        "The Box voxel SDF samples cannot be allocated safely");
  } catch (const std::length_error&) {
    throw std::logic_error(
        "The Box voxel SDF samples cannot be allocated safely");
  }

  for (int k = 0; k < cell_counts_[2]; ++k) {
    for (int j = 0; j < cell_counts_[1]; ++j) {
      for (int i = 0; i < cell_counts_[0]; ++i) {
        const Vector3<double> center = cell_center(i, j, k);
        const auto distance =
            point_distance::DistanceToPoint<double>::ComputeDistanceToBox<3>(
                half_widths_, center);
        const Vector3<double>& nearest = std::get<0>(distance);
        const Vector3<double>& gradient = std::get<1>(distance);
        // At medial-axis ties, the chosen Box gradient follows Drake's public
        // point-distance implementation.
        samples_[linear_index(i, j, k)] =
            SdfSample{gradient.dot(center - nearest), gradient};
      }
    }
  }
}

Vector3<double> VoxelSdfGeometry::cell_center(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < cell_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < cell_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < cell_counts_[2]);
  return lower_cell_boundary_ +
         voxel_width_ *
             (Vector3<double>(static_cast<double>(i), static_cast<double>(j),
                              static_cast<double>(k)) +
              Vector3<double>::Constant(0.5));
}

const VoxelSdfGeometry::SdfSample& VoxelSdfGeometry::sample(int i, int j,
                                                            int k) const {
  return samples_[linear_index(i, j, k)];
}

size_t VoxelSdfGeometry::linear_index(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < cell_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < cell_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < cell_counts_[2]);
  return static_cast<size_t>(i) +
         static_cast<size_t>(cell_counts_[0]) *
             (static_cast<size_t>(j) +
              static_cast<size_t>(cell_counts_[1]) * static_cast<size_t>(k));
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
