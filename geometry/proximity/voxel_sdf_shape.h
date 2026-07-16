#pragma once

#include <string_view>
#include <variant>
#include <vector>

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

  /* An affine half space `normal.dot(p_FQ) + offset <= 0`. The normal need not
   have unit length. The caller must express the normal and query point in the
   same frame F. Branches returned by this class use the shape's geometry
   frame. */
  struct AffineHalfSpace {
    Vector3<double> normal{};
    double offset{};

    double Evaluate(const Vector3<double>& p_FQ) const {
      return normal.dot(p_FQ) + offset;
    }
  };

  /* One affine piece of a shape's SDF at a query point. `active_region`
   contains the half spaces whose intersection is the region where this piece
   realizes the shape SDF. `index` gives the pieces a stable tie-breaking
   order. `is_cell_invariant` is true when the same affine function applies in
   every queried cell; its sample value can then change only by recentering that
   function. */
  struct AffineBranch {
    Sample sample;
    std::vector<AffineHalfSpace> active_region;
    int index{};
    bool is_cell_invariant{};
  };

  explicit VoxelSdfShape(const Box& box);
  explicit VoxelSdfShape(const Sphere& sphere);

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VoxelSdfShape);

  Sample Evaluate(const Vector3<double>& p_GQ) const;

  /* Calculates the affine SDF pieces at `p_GQ`. A Sphere has one local affine
   piece. A Box has six exact face-affine pieces; their active regions partition
   the Box interior, apart from shared boundaries. */
  std::vector<AffineBranch> CalcAffineBranches(
      const Vector3<double>& p_GQ) const;

  /* As above, but uses `single_branch_sample` for a shape represented by one
   local affine piece. This overload lets registered geometry use its cached
   cell sample. Piecewise-affine shapes construct all of their exact pieces and
   do not use the selected sample. */
  std::vector<AffineBranch> CalcAffineBranches(
      const Vector3<double>& p_GQ, const Sample& single_branch_sample) const;

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
  static std::vector<AffineBranch> DoCalcAffineBranches(
      const BoxData& box, const Vector3<double>& p_GQ);
  static std::vector<AffineBranch> DoCalcAffineBranches(
      const BoxData& box, const Vector3<double>& p_GQ,
      const Sample& single_branch_sample);
  static Vector3<double> DoCalcBoundingBoxHalfWidths(const BoxData& box);
  static double DoCalcCharacteristicLength(const BoxData& box);
  static std::string_view DoGetShapeName(const BoxData& box);

  static Sample DoEvaluate(const SphereData& sphere,
                           const Vector3<double>& p_GQ);
  static std::vector<AffineBranch> DoCalcAffineBranches(
      const SphereData& sphere, const Vector3<double>& p_GQ);
  static std::vector<AffineBranch> DoCalcAffineBranches(
      const SphereData& sphere, const Vector3<double>& p_GQ,
      const Sample& single_branch_sample);
  static Vector3<double> DoCalcBoundingBoxHalfWidths(const SphereData& sphere);
  static double DoCalcCharacteristicLength(const SphereData& sphere);
  static std::string_view DoGetShapeName(const SphereData& sphere);

  std::variant<BoxData, SphereData> data_;
};

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
