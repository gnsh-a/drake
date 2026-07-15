#include "drake/geometry/proximity/voxel_sdf_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "drake/common/drake_assert.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

int CalcCellCount(double extent, double voxel_width, int axis,
                  std::string_view shape_name) {
  const double count = std::ceil(extent / voxel_width);
  if (!std::isfinite(count) ||
      count > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::logic_error(
        fmt::format("The {} voxel SDF requires too many cells on axis {}",
                    shape_name, axis));
  }
  return std::max(1, static_cast<int>(count));
}

size_t CheckedMultiply(size_t a, size_t b, std::string_view shape_name) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF sample count overflows size_t", shape_name));
  }
  return a * b;
}

}  // namespace

VoxelSdfGeometry::VoxelSdfGeometry(const Box& box, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(VoxelSdfShape(box), voxel_width, hydroelastic_modulus) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Sphere& sphere, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(VoxelSdfShape(sphere), voxel_width,
                       hydroelastic_modulus) {}

VoxelSdfGeometry::VoxelSdfGeometry(VoxelSdfShape shape, double voxel_width,
                                   double hydroelastic_modulus)
    : shape_(std::move(shape)),
      voxel_width_(voxel_width),
      hydroelastic_modulus_(hydroelastic_modulus),
      characteristic_length_(shape_.characteristic_length()),
      pressure_scale_(0.0) {
  const std::string shape_name(shape_.shape_name());
  if (!(voxel_width > 0.0 && std::isfinite(voxel_width))) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF width must be finite and strictly positive",
        shape_name));
  }
  if (!(hydroelastic_modulus > 0.0 && std::isfinite(hydroelastic_modulus))) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF hydroelastic modulus must be finite and strictly "
        "positive",
        shape_name));
  }
  if (!(characteristic_length_ > 0.0 &&
        std::isfinite(characteristic_length_))) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF characteristic length must be finite and strictly "
        "positive",
        shape_name));
  }
  pressure_scale_ = hydroelastic_modulus_ / characteristic_length_;
  if (!(pressure_scale_ > 0.0 && std::isfinite(pressure_scale_))) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF pressure scale must be finite and strictly positive",
        shape_name));
  }

  const Vector3<double> extent = 2.0 * shape_.bounding_box_half_widths();
  for (int a = 0; a < 3; ++a) {
    cell_counts_[a] = CalcCellCount(extent[a], voxel_width_, a, shape_name);
    // Centering the padded grid about the shape makes its padding symmetric.
    lower_cell_boundary_[a] = -0.5 * cell_counts_[a] * voxel_width_;
    if (!std::isfinite(lower_cell_boundary_[a])) {
      throw std::logic_error(
          fmt::format("The {} voxel SDF grid boundary is not finite on axis {}",
                      shape_name, a));
    }
  }

  size_t sample_count =
      CheckedMultiply(static_cast<size_t>(cell_counts_[0]),
                      static_cast<size_t>(cell_counts_[1]), shape_name);
  sample_count = CheckedMultiply(
      sample_count, static_cast<size_t>(cell_counts_[2]), shape_name);
  if (sample_count > samples_.max_size()) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF sample count cannot be represented by a vector",
        shape_name));
  }
  try {
    samples_.resize(sample_count);
  } catch (const std::bad_alloc&) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF samples cannot be allocated safely", shape_name));
  } catch (const std::length_error&) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF samples cannot be allocated safely", shape_name));
  }

  for (int k = 0; k < cell_counts_[2]; ++k) {
    for (int j = 0; j < cell_counts_[1]; ++j) {
      for (int i = 0; i < cell_counts_[0]; ++i) {
        const Vector3<double> center = cell_center(i, j, k);
        if (!center.allFinite()) {
          throw std::logic_error(fmt::format(
              "The {} voxel SDF cell center ({}, {}, {}) is not finite",
              shape_name, i, j, k));
        }
        const SdfSample sdf = EvaluateSdf(center);
        if (!sdf.gradient.allFinite() || !std::isfinite(sdf.value)) {
          throw std::logic_error(
              fmt::format("The {} voxel SDF sample ({}, {}, {}) is not finite",
                          shape_name, i, j, k));
        }
        // At non-unique gradients, including Box medial-axis ties and the
        // Sphere center, the choice follows Drake's point-distance code.
        samples_[linear_index(i, j, k)] = sdf;
      }
    }
  }
}

Vector3<double> VoxelSdfGeometry::cell_center(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < cell_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < cell_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < cell_counts_[2]);
  // Fused multiply-add avoids overflowing the intermediate h * (index + 0.5)
  // when its sum with the negative lower boundary is representable.
  return Vector3<double>(std::fma(voxel_width_, static_cast<double>(i) + 0.5,
                                  lower_cell_boundary_[0]),
                         std::fma(voxel_width_, static_cast<double>(j) + 0.5,
                                  lower_cell_boundary_[1]),
                         std::fma(voxel_width_, static_cast<double>(k) + 0.5,
                                  lower_cell_boundary_[2]));
}

const VoxelSdfGeometry::SdfSample& VoxelSdfGeometry::sample(int i, int j,
                                                            int k) const {
  return samples_[linear_index(i, j, k)];
}

VoxelSdfGeometry::SdfSample VoxelSdfGeometry::EvaluateSdf(
    const Vector3<double>& p_GQ) const {
  return shape_.Evaluate(p_GQ);
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
