#include "drake/geometry/proximity/voxel_sdf_shape.h"

#include <array>
#include <cmath>
#include <tuple>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

VoxelSdfShape::VoxelSdfShape(const Box& box)
    : data_(BoxData{box.size() / 2.0}) {}

VoxelSdfShape::VoxelSdfShape(const Sphere& sphere)
    : data_(SphereData{sphere.radius()}) {}

VoxelSdfShape::Sample VoxelSdfShape::Evaluate(
    const Vector3<double>& p_GQ) const {
  DRAKE_DEMAND(p_GQ.allFinite());
  return std::visit(
      [&p_GQ](const auto& data) {
        return DoEvaluate(data, p_GQ);
      },
      data_);
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::CalcAffineBranches(
    const Vector3<double>& p_GQ) const {
  DRAKE_DEMAND(p_GQ.allFinite());
  return std::visit(
      [&p_GQ](const auto& data) {
        return DoCalcAffineBranches(data, p_GQ);
      },
      data_);
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::CalcAffineBranches(
    const Vector3<double>& p_GQ,
    const VoxelSdfShape::Sample& single_branch_sample) const {
  DRAKE_DEMAND(p_GQ.allFinite());
  DRAKE_DEMAND(std::isfinite(single_branch_sample.value));
  DRAKE_DEMAND(single_branch_sample.gradient.allFinite());
  return std::visit(
      [&p_GQ, &single_branch_sample](const auto& data) {
        return DoCalcAffineBranches(data, p_GQ, single_branch_sample);
      },
      data_);
}

Vector3<double> VoxelSdfShape::bounding_box_half_widths() const {
  return std::visit(
      [](const auto& data) {
        return DoCalcBoundingBoxHalfWidths(data);
      },
      data_);
}

double VoxelSdfShape::characteristic_length() const {
  return std::visit(
      [](const auto& data) {
        return DoCalcCharacteristicLength(data);
      },
      data_);
}

std::string_view VoxelSdfShape::shape_name() const {
  return std::visit(
      [](const auto& data) {
        return DoGetShapeName(data);
      },
      data_);
}

std::string_view VoxelSdfShape::supported_shape_names() {
  return "Box and Sphere";
}

VoxelSdfShape::Sample VoxelSdfShape::DoEvaluate(const BoxData& box,
                                                const Vector3<double>& p_GQ) {
  const auto distance =
      point_distance::DistanceToPoint<double>::ComputeDistanceToBox<3>(
          box.half_widths, p_GQ);
  const Vector3<double>& nearest = std::get<0>(distance);
  const Vector3<double>& gradient = std::get<1>(distance);
  return Sample{gradient.dot(p_GQ - nearest), gradient};
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::DoCalcAffineBranches(
    const BoxData& box, const Vector3<double>& p_GQ) {
  std::array<Sample, 6> samples;
  int index = 0;
  // This ordering matches DistanceToPoint::ExtremalAxis():
  // +x, -x, +y, -y, +z, -z.
  for (int axis = 0; axis < 3; ++axis) {
    for (const double sign : {1.0, -1.0}) {
      Vector3<double> gradient = Vector3<double>::Zero();
      gradient[axis] = sign;
      samples[index++] =
          Sample{sign * p_GQ[axis] - box.half_widths[axis], gradient};
    }
  }

  std::vector<AffineBranch> result;
  result.reserve(samples.size());
  for (int f = 0; f < static_cast<int>(samples.size()); ++f) {
    AffineBranch branch{samples[f], {}, f, true};
    branch.active_region.reserve(samples.size() - 1);
    for (int g = 0; g < static_cast<int>(samples.size()); ++g) {
      if (g == f) continue;
      // The interior Box SDF is max(phi_f). Piece f is active where every
      // phi_g - phi_f is nonpositive.
      branch.active_region.push_back(AffineHalfSpace{
          samples[g].gradient - samples[f].gradient,
          samples[g].value - samples[f].value -
              (samples[g].gradient - samples[f].gradient).dot(p_GQ)});
    }
    result.push_back(std::move(branch));
  }
  return result;
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::DoCalcAffineBranches(
    const BoxData& box, const Vector3<double>& p_GQ,
    const VoxelSdfShape::Sample&) {
  return DoCalcAffineBranches(box, p_GQ);
}

Vector3<double> VoxelSdfShape::DoCalcBoundingBoxHalfWidths(const BoxData& box) {
  return box.half_widths;
}

double VoxelSdfShape::DoCalcCharacteristicLength(const BoxData& box) {
  return box.half_widths.minCoeff();
}

std::string_view VoxelSdfShape::DoGetShapeName(const BoxData&) {
  return "Box";
}

VoxelSdfShape::Sample VoxelSdfShape::DoEvaluate(const SphereData& sphere,
                                                const Vector3<double>& p_GQ) {
  const fcl::Sphered fcl_sphere(sphere.radius);
  Vector3<double> nearest;
  Vector3<double> gradient;
  double distance{};
  point_distance::SphereDistanceInSphereFrame(fcl_sphere, p_GQ, &nearest,
                                              &distance, &gradient);
  return Sample{distance, gradient};
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::DoCalcAffineBranches(
    const SphereData& sphere, const Vector3<double>& p_GQ) {
  return DoCalcAffineBranches(sphere, p_GQ, DoEvaluate(sphere, p_GQ));
}

std::vector<VoxelSdfShape::AffineBranch> VoxelSdfShape::DoCalcAffineBranches(
    const SphereData&, const Vector3<double>&,
    const VoxelSdfShape::Sample& single_branch_sample) {
  return {AffineBranch{single_branch_sample, {}, 0, false}};
}

Vector3<double> VoxelSdfShape::DoCalcBoundingBoxHalfWidths(
    const SphereData& sphere) {
  return Vector3<double>::Constant(sphere.radius);
}

double VoxelSdfShape::DoCalcCharacteristicLength(const SphereData& sphere) {
  return sphere.radius;
}

std::string_view VoxelSdfShape::DoGetShapeName(const SphereData&) {
  return "Sphere";
}

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
