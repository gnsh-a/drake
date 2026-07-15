#include "drake/geometry/proximity/voxel_sdf_shape.h"

#include <tuple>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/distance_to_point_callback.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

VoxelSdfShape::VoxelSdfShape(const Box& box)
    : data_(BoxData{box.size() / 2.0}) {}

VoxelSdfShape::Sample VoxelSdfShape::Evaluate(
    const Vector3<double>& p_GQ) const {
  DRAKE_DEMAND(p_GQ.allFinite());
  return std::visit(
      [&p_GQ](const auto& data) {
        return DoEvaluate(data, p_GQ);
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

VoxelSdfShape::Sample VoxelSdfShape::DoEvaluate(const BoxData& box,
                                                const Vector3<double>& p_GQ) {
  const auto distance =
      point_distance::DistanceToPoint<double>::ComputeDistanceToBox<3>(
          box.half_widths, p_GQ);
  const Vector3<double>& nearest = std::get<0>(distance);
  const Vector3<double>& gradient = std::get<1>(distance);
  return Sample{gradient.dot(p_GQ - nearest), gradient};
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

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
