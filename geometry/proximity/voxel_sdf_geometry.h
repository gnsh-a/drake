#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/voxel_sdf_shape.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* Immutable registered geometry data for a voxel signed-distance field. Grid
 coordinates and gradients are expressed in the geometry frame. Sampling is
 registration-time work; the owned samples are neither Context state nor a
 query cache. */
class VoxelSdfGeometry {
 public:
  using SdfSample = VoxelSdfShape::Sample;
  using SdfBranch = VoxelSdfShape::AffineBranch;

  VoxelSdfGeometry(const Box& box, double voxel_width,
                   double hydroelastic_modulus);
  VoxelSdfGeometry(const Box& box, double voxel_width,
                   double hydroelastic_modulus,
                   VoxelSdfEvaluationMode evaluation_mode);
  VoxelSdfGeometry(const Sphere& sphere, double voxel_width,
                   double hydroelastic_modulus);
  VoxelSdfGeometry(const Sphere& sphere, double voxel_width,
                   double hydroelastic_modulus,
                   VoxelSdfEvaluationMode evaluation_mode);
  VoxelSdfGeometry(VoxelSdfShape shape, double voxel_width,
                   double hydroelastic_modulus);
  VoxelSdfGeometry(VoxelSdfShape shape, double voxel_width,
                   double hydroelastic_modulus,
                   VoxelSdfEvaluationMode evaluation_mode);

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VoxelSdfGeometry);

  double voxel_width() const { return voxel_width_; }
  double hydroelastic_modulus() const { return hydroelastic_modulus_; }
  double characteristic_length() const { return characteristic_length_; }
  double pressure_scale() const { return pressure_scale_; }
  VoxelSdfEvaluationMode evaluation_mode() const { return evaluation_mode_; }
  const Vector3<int>& cell_counts() const { return cell_counts_; }
  const Vector3<int>& storage_counts() const { return storage_counts_; }
  const Vector3<double>& lower_cell_boundary() const {
    return lower_cell_boundary_;
  }

  Vector3<double> cell_center(int i, int j, int k) const;
  const SdfSample& sample(int i, int j, int k) const;
  Vector3<double> stored_sample_center(int i, int j, int k) const;
  const SdfSample& stored_sample(int i, int j, int k) const;

  /* Interpolates the stored scalar field and returns the derivative of that
   same interpolant. Returns no value outside the stored sample-center domain.
   This is only available for kSampledTrilinear geometry. */
  std::optional<SdfSample> InterpolateSdf(const Vector3<double>& p_GQ) const;

  SdfSample EvaluateSdf(const Vector3<double>& p_GQ) const;
  std::vector<SdfBranch> CalcCellSdfBranches(int i, int j, int k) const;
  std::vector<SdfBranch> EvaluateSdfBranches(const Vector3<double>& p_GQ) const;

 private:
  static constexpr int kSampledPadding = 2;

  size_t storage_linear_index(int i, int j, int k) const;
  int core_storage_offset() const;

  VoxelSdfShape shape_;
  double voxel_width_{};
  double hydroelastic_modulus_{};
  double characteristic_length_{};
  double pressure_scale_{};
  VoxelSdfEvaluationMode evaluation_mode_{
      VoxelSdfEvaluationMode::kPrimitiveAffine};
  Vector3<int> cell_counts_;
  Vector3<int> storage_counts_;
  Vector3<double> lower_cell_boundary_;
  // The core lattice supplies traversed host cells. In sampled mode, the
  // larger stored lattice exists only to bracket transformed interpolation
  // queries. Its two immutable padding layers per side are never traversed as
  // contact cells; they are a fixed representation detail, not a hydroelastic
  // margin. Samples are indexed with x varying fastest, then y, then z.
  std::vector<SdfSample> samples_;
};

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
