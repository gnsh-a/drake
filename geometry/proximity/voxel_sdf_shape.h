#pragma once

#include <string_view>
#include <variant>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

/* Immutable, fully owned shape data used to construct and query a voxel SDF.
 Exact signed-distance evaluation delegates to Drake's existing point-distance
 implementation. All points and gradients are expressed in the shape's
 geometry frame. */
class VoxelSdfShape {
 public:
  struct Sample {
    double value{};
    Vector3<double> gradient{};
  };

  explicit VoxelSdfShape(const Box& box);
  explicit VoxelSdfShape(const Sphere& sphere);

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VoxelSdfShape);

  Sample Evaluate(const Vector3<double>& p_GQ) const;
  Vector3<double> bounding_box_half_widths() const;
  double characteristic_length() const;
  std::string_view shape_name() const;
  static std::string_view supported_shape_names();

 private:
  struct BoxData {
    Vector3<double> half_widths;
  };
  struct SphereData {
    double radius{};
  };

  static Sample DoEvaluate(const BoxData& box, const Vector3<double>& p_GQ);
  static Vector3<double> DoCalcBoundingBoxHalfWidths(const BoxData& box);
  static double DoCalcCharacteristicLength(const BoxData& box);
  static std::string_view DoGetShapeName(const BoxData& box);

  static Sample DoEvaluate(const SphereData& sphere,
                           const Vector3<double>& p_GQ);
  static Vector3<double> DoCalcBoundingBoxHalfWidths(const SphereData& sphere);
  static double DoCalcCharacteristicLength(const SphereData& sphere);
  static std::string_view DoGetShapeName(const SphereData& sphere);

  std::variant<BoxData, SphereData> data_;
};

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
