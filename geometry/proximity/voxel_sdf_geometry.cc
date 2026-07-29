#include "drake/geometry/proximity/voxel_sdf_geometry.h"

#include <algorithm>
#include <array>
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

int CheckedCount(int core_count, int extra_count, std::string_view count_name,
                 std::string_view shape_name, int axis) {
  DRAKE_DEMAND(core_count > 0);
  DRAKE_DEMAND(extra_count >= 0);
  if (core_count > std::numeric_limits<int>::max() - extra_count) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF {} count overflows int on axis {}", shape_name,
        count_name, axis));
  }
  return core_count + extra_count;
}

void ValidateEvaluationMode(VoxelSdfEvaluationMode mode,
                            std::string_view shape_name) {
  switch (mode) {
    case VoxelSdfEvaluationMode::kPrimitiveAffine:
    case VoxelSdfEvaluationMode::kSampledTrilinear:
      return;
  }
  throw std::logic_error(
      fmt::format("The {} voxel SDF evaluation mode is invalid", shape_name));
}

}  // namespace

VoxelSdfGeometry::VoxelSdfGeometry(const Box& box, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(box, voxel_width, hydroelastic_modulus,
                       VoxelSdfEvaluationMode::kPrimitiveAffine) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Box& box, double voxel_width,
                                   double hydroelastic_modulus,
                                   VoxelSdfEvaluationMode evaluation_mode)
    : VoxelSdfGeometry(VoxelSdfShape(box), voxel_width, hydroelastic_modulus,
                       evaluation_mode) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Cylinder& cylinder, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(cylinder, voxel_width, hydroelastic_modulus,
                       VoxelSdfEvaluationMode::kPrimitiveAffine) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Cylinder& cylinder, double voxel_width,
                                   double hydroelastic_modulus,
                                   VoxelSdfEvaluationMode evaluation_mode)
    : VoxelSdfGeometry(VoxelSdfShape(cylinder), voxel_width,
                       hydroelastic_modulus, evaluation_mode) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Ellipsoid& ellipsoid,
                                   double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(ellipsoid, voxel_width, hydroelastic_modulus,
                       VoxelSdfEvaluationMode::kPrimitiveAffine) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Ellipsoid& ellipsoid,
                                   double voxel_width,
                                   double hydroelastic_modulus,
                                   VoxelSdfEvaluationMode evaluation_mode)
    : VoxelSdfGeometry(VoxelSdfShape(ellipsoid), voxel_width,
                       hydroelastic_modulus, evaluation_mode) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Sphere& sphere, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(sphere, voxel_width, hydroelastic_modulus,
                       VoxelSdfEvaluationMode::kPrimitiveAffine) {}

VoxelSdfGeometry::VoxelSdfGeometry(const Sphere& sphere, double voxel_width,
                                   double hydroelastic_modulus,
                                   VoxelSdfEvaluationMode evaluation_mode)
    : VoxelSdfGeometry(VoxelSdfShape(sphere), voxel_width, hydroelastic_modulus,
                       evaluation_mode) {}

VoxelSdfGeometry::VoxelSdfGeometry(VoxelSdfShape shape, double voxel_width,
                                   double hydroelastic_modulus)
    : VoxelSdfGeometry(std::move(shape), voxel_width, hydroelastic_modulus,
                       VoxelSdfEvaluationMode::kPrimitiveAffine) {}

VoxelSdfGeometry::VoxelSdfGeometry(VoxelSdfShape shape, double voxel_width,
                                   double hydroelastic_modulus,
                                   VoxelSdfEvaluationMode evaluation_mode)
    : shape_(std::move(shape)),
      voxel_width_(voxel_width),
      hydroelastic_modulus_(hydroelastic_modulus),
      characteristic_length_(shape_.characteristic_length()),
      pressure_scale_(0.0),
      evaluation_mode_(evaluation_mode) {
  const std::string shape_name(shape_.shape_name());
  ValidateEvaluationMode(evaluation_mode_, shape_name);
  if (evaluation_mode_ == VoxelSdfEvaluationMode::kSampledTrilinear &&
      !shape_.supports_sampled_trilinear()) {
    throw std::logic_error(fmt::format(
        "The {} voxel SDF does not support sampled trilinear evaluation",
        shape_name));
  }
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
  const int storage_padding = core_storage_offset();
  for (int a = 0; a < 3; ++a) {
    cell_counts_[a] = CalcCellCount(extent[a], voxel_width_, a, shape_name);
    // Center the core grid about the shape; fixed storage padding then extends
    // symmetrically beyond these unchanged original-cell boundaries.
    lower_cell_boundary_[a] = -0.5 * cell_counts_[a] * voxel_width_;
    if (!std::isfinite(lower_cell_boundary_[a])) {
      throw std::logic_error(
          fmt::format("The {} voxel SDF grid boundary is not finite on axis {}",
                      shape_name, a));
    }
    storage_counts_[a] =
        CheckedCount(cell_counts_[a], 2 * storage_padding, "padded sample",
                     shape_name, a);
    mc_node_counts_[a] =
        CheckedCount(cell_counts_[a], 2, "marching-cubes node", shape_name, a);
    mc_cube_counts_[a] =
        CheckedCount(cell_counts_[a], 1, "marching-cubes cube", shape_name, a);
  }

  // Check the padded coordinate extrema before allocating. All stored centers
  // lie between these two points on each axis. The MC nodes use either this
  // complete range (primitive mode) or a strict subset (sampled mode), so this
  // also proves that every dual-grid node and cube boundary is finite.
  const Vector3<double> first_center = stored_sample_center(0, 0, 0);
  const Vector3<double> last_center = stored_sample_center(
      storage_counts_[0] - 1, storage_counts_[1] - 1, storage_counts_[2] - 1);
  if (!first_center.allFinite() || !last_center.allFinite()) {
    throw std::logic_error(
        fmt::format("The {} voxel SDF padded sample coordinates are not finite",
                    shape_name));
  }

  size_t sample_count =
      CheckedMultiply(static_cast<size_t>(storage_counts_[0]),
                      static_cast<size_t>(storage_counts_[1]), shape_name);
  sample_count = CheckedMultiply(
      sample_count, static_cast<size_t>(storage_counts_[2]), shape_name);
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

  for (int k = 0; k < storage_counts_[2]; ++k) {
    for (int j = 0; j < storage_counts_[1]; ++j) {
      for (int i = 0; i < storage_counts_[0]; ++i) {
        const Vector3<double> center = stored_sample_center(i, j, k);
        if (!center.allFinite()) {
          throw std::logic_error(fmt::format(
              "The {} voxel SDF sample center ({}, {}, {}) is not finite",
              shape_name, i, j, k));
        }
        // Analytical shape evaluation is registration-time work. The complete
        // padded lattice is immutable registered geometry, not Context state
        // or query scratch. In sampled mode, all later off-grid queries use
        // this stored lattice.
        const SdfSample sdf = shape_.Evaluate(center);
        if (!sdf.gradient.allFinite() || !std::isfinite(sdf.value)) {
          throw std::logic_error(
              fmt::format("The {} voxel SDF sample ({}, {}, {}) is not finite",
                          shape_name, i, j, k));
        }
        // At non-unique gradients, including Box medial-axis ties and the
        // Sphere center, the choice follows Drake's point-distance code.
        samples_[storage_linear_index(i, j, k)] = sdf;
      }
    }
  }
}

Vector3<double> VoxelSdfGeometry::cell_center(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < cell_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < cell_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < cell_counts_[2]);
  const int offset = core_storage_offset();
  return stored_sample_center(i + offset, j + offset, k + offset);
}

const VoxelSdfGeometry::SdfSample& VoxelSdfGeometry::sample(int i, int j,
                                                            int k) const {
  DRAKE_DEMAND(i >= 0 && i < cell_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < cell_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < cell_counts_[2]);
  const int offset = core_storage_offset();
  return stored_sample(i + offset, j + offset, k + offset);
}

Vector3<double> VoxelSdfGeometry::mc_node_position(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < mc_node_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < mc_node_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < mc_node_counts_[2]);
  const int offset = mc_storage_offset();
  return stored_sample_center(i + offset, j + offset, k + offset);
}

double VoxelSdfGeometry::mc_node_value(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < mc_node_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < mc_node_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < mc_node_counts_[2]);
  const int offset = mc_storage_offset();
  return stored_sample(i + offset, j + offset, k + offset).value;
}

Vector3<double> VoxelSdfGeometry::stored_sample_center(int i, int j,
                                                       int k) const {
  DRAKE_DEMAND(i >= 0 && i < storage_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < storage_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < storage_counts_[2]);
  const int offset = core_storage_offset();
  // Fused multiply-add avoids overflowing the intermediate h * coordinate
  // when its sum with the negative lower boundary is representable.
  return Vector3<double>(
      std::fma(voxel_width_, static_cast<double>(i - offset) + 0.5,
               lower_cell_boundary_[0]),
      std::fma(voxel_width_, static_cast<double>(j - offset) + 0.5,
               lower_cell_boundary_[1]),
      std::fma(voxel_width_, static_cast<double>(k - offset) + 0.5,
               lower_cell_boundary_[2]));
}

const VoxelSdfGeometry::SdfSample& VoxelSdfGeometry::stored_sample(
    int i, int j, int k) const {
  return samples_[storage_linear_index(i, j, k)];
}

std::optional<VoxelSdfGeometry::SdfSample> VoxelSdfGeometry::InterpolateSdf(
    const Vector3<double>& p_GQ) const {
  DRAKE_DEMAND(evaluation_mode_ == VoxelSdfEvaluationMode::kSampledTrilinear);
  DRAKE_DEMAND(p_GQ.allFinite());

  struct AxisInterval {
    int lower{};
    double coordinate{};
  };
  std::array<AxisInterval, 3> intervals;
  const Vector3<double> first = stored_sample_center(0, 0, 0);
  const Vector3<double> last = stored_sample_center(
      storage_counts_[0] - 1, storage_counts_[1] - 1, storage_counts_[2] - 1);
  for (int a = 0; a < 3; ++a) {
    if (p_GQ[a] < first[a] || p_GQ[a] > last[a]) return std::nullopt;
    // Interpolation cells are half open: a query on an internal sample plane
    // belongs to the interval beginning at that sample. The final outer plane
    // is included by assigning it to the last interval at coordinate one.
    if (p_GQ[a] == last[a]) {
      intervals[a] = AxisInterval{storage_counts_[a] - 2, 1.0};
    } else {
      const double lattice_coordinate = (p_GQ[a] - first[a]) / voxel_width_;
      const int lower = static_cast<int>(std::floor(lattice_coordinate));
      if (lower < 0 || lower >= storage_counts_[a] - 1) return std::nullopt;
      intervals[a] =
          AxisInterval{lower, lattice_coordinate - static_cast<double>(lower)};
    }
  }

  const int i = intervals[0].lower;
  const int j = intervals[1].lower;
  const int k = intervals[2].lower;
  const double u = intervals[0].coordinate;
  const double v = intervals[1].coordinate;
  const double w = intervals[2].coordinate;
  const double one_u = 1.0 - u;
  const double one_v = 1.0 - v;
  const double one_w = 1.0 - w;
  const auto value = [this, i, j, k](int di, int dj, int dk) {
    return stored_sample(i + di, j + dj, k + dk).value;
  };
  const double f000 = value(0, 0, 0);
  const double f100 = value(1, 0, 0);
  const double f010 = value(0, 1, 0);
  const double f110 = value(1, 1, 0);
  const double f001 = value(0, 0, 1);
  const double f101 = value(1, 0, 1);
  const double f011 = value(0, 1, 1);
  const double f111 = value(1, 1, 1);

  // The scalar grid is the source of truth for off-grid evaluation. Its
  // gradient is the derivative of this same trilinear scalar field so pressure
  // values and gradients remain mutually consistent; stored gradients are not
  // blended. Trilinear gradients can jump across lattice cells or vanish at a
  // symmetric point, and neither condition warrants normalization or a
  // replacement value.
  const double interpolated_value =
      one_u * one_v * one_w * f000 + u * one_v * one_w * f100 +
      one_u * v * one_w * f010 + u * v * one_w * f110 +
      one_u * one_v * w * f001 + u * one_v * w * f101 + one_u * v * w * f011 +
      u * v * w * f111;
  const Vector3<double> parametric_gradient(
      one_v * one_w * (f100 - f000) + v * one_w * (f110 - f010) +
          one_v * w * (f101 - f001) + v * w * (f111 - f011),
      one_u * one_w * (f010 - f000) + u * one_w * (f110 - f100) +
          one_u * w * (f011 - f001) + u * w * (f111 - f101),
      one_u * one_v * (f001 - f000) + u * one_v * (f101 - f100) +
          one_u * v * (f011 - f010) + u * v * (f111 - f110));
  return SdfSample{interpolated_value, parametric_gradient / voxel_width_};
}

VoxelSdfGeometry::SdfSample VoxelSdfGeometry::EvaluateSdf(
    const Vector3<double>& p_GQ) const {
  return shape_.Evaluate(p_GQ);
}

std::vector<VoxelSdfGeometry::SdfBranch> VoxelSdfGeometry::CalcCellSdfBranches(
    int i, int j, int k) const {
  if (evaluation_mode_ == VoxelSdfEvaluationMode::kSampledTrilinear) {
    // TODO(gnsh-a): A sampled scalar SDF cannot recover the Box face branches
    // discarded during sampling, so this mode uses one local affine branch.
    return {SdfBranch{sample(i, j, k), {}, 0, false}};
  }
  return shape_.CalcAffineBranches(cell_center(i, j, k), sample(i, j, k));
}

std::vector<VoxelSdfGeometry::SdfBranch> VoxelSdfGeometry::EvaluateSdfBranches(
    const Vector3<double>& p_GQ) const {
  return shape_.CalcAffineBranches(p_GQ);
}

size_t VoxelSdfGeometry::storage_linear_index(int i, int j, int k) const {
  DRAKE_DEMAND(i >= 0 && i < storage_counts_[0]);
  DRAKE_DEMAND(j >= 0 && j < storage_counts_[1]);
  DRAKE_DEMAND(k >= 0 && k < storage_counts_[2]);
  return static_cast<size_t>(i) +
         static_cast<size_t>(storage_counts_[0]) *
             (static_cast<size_t>(j) +
              static_cast<size_t>(storage_counts_[1]) * static_cast<size_t>(k));
}

int VoxelSdfGeometry::core_storage_offset() const {
  return evaluation_mode_ == VoxelSdfEvaluationMode::kSampledTrilinear
             ? kSampledPadding
             : kPrimitivePadding;
}

int VoxelSdfGeometry::mc_storage_offset() const {
  // MC uses the nearest padding layer on either side of the core. Sampled
  // mode's second layer remains exclusively for interpolation coverage.
  return core_storage_offset() - 1;
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
