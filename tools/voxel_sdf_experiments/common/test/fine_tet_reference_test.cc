#include "drake/tools/voxel_sdf_experiments/common/fine_tet_reference.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/geometry/proximity/polygon_surface_mesh_field.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/shape_specification.h"
#include "drake/tools/voxel_sdf_experiments/common/reference.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

class FineTetReferenceTester {
 public:
  static FineTetReference Make(geometry::ContactSurface<double> surface) {
    return FineTetReference(std::move(surface));
  }
};

namespace {

using geometry::Box;
using geometry::ContactSurface;
using geometry::GeometryId;
using geometry::PolygonSurfaceMesh;
using geometry::PolygonSurfaceMeshFieldLinear;
using geometry::Sphere;
using math::RigidTransformd;

constexpr double kRadius = 0.1;
constexpr double kPenetration = 0.08 / 3.0;
constexpr double kModulus = 1.0e8;

FineTetReference MakeSphereBoxReference(double tet_resolution_hint) {
  const Sphere sphere(kRadius);
  const Box box(0.4, 0.4, 0.2);
  const RigidTransformd X_RS(
      Eigen::Vector3d(0.0, 0.0, 2.0 * kRadius - kPenetration));
  return FineTetReference(sphere, X_RS, kModulus, box, RigidTransformd(),
                          kModulus, tet_resolution_hint);
}

struct Errors {
  double force{};
  double area{};
  double centroid{};
};

Errors CalcErrors(const FineTetReference& fine,
                  const AnalyticParaboloid& analytic) {
  return {
      .force = std::abs(fine.force() - analytic.force()) / analytic.force(),
      .area = std::abs(fine.area() - analytic.area()) / analytic.area(),
      .centroid = (fine.centroid() - analytic.centroid()).norm(),
  };
}

FineTetReference MakeSquareReference() {
  std::vector<Eigen::Vector3d> vertices{
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  auto mesh = std::make_unique<PolygonSurfaceMesh<double>>(
      std::vector<int>{4, 0, 1, 2, 3}, std::move(vertices));
  auto pressure =
      std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
          std::vector<double>{1.0, 2.0, 3.0, 2.0}, mesh.get(),
          std::vector<Eigen::Vector3d>{{1.0, 1.0, 0.0}});
  return FineTetReferenceTester::Make(
      ContactSurface<double>(GeometryId::get_new_id(), GeometryId::get_new_id(),
                             std::move(mesh), std::move(pressure)));
}

GTEST_TEST(FineTetReferenceTest, UndefinedPatchDescriptorsAreNan) {
  const FineTetReference reference = MakeSphereBoxReference(0.01);
  EXPECT_TRUE(std::isnan(reference.patch_radius()));
  EXPECT_TRUE(std::isnan(reference.peak_pressure()));
  EXPECT_TRUE(reference.normal().array().isNaN().all());
  EXPECT_GT(reference.force(), 0.0);
  EXPECT_GT(reference.area(), 0.0);
  EXPECT_TRUE(reference.centroid().allFinite());
}

/* This test targets the spatial query itself using a known finite square.
 The three probes distinguish face-interior, edge, and corner nearest features;
 checking only an interior offset would not detect an implementation that
 measures distance to the infinite supporting plane. The synthetic surface is
 injected through a private test seam; production construction always builds
 the surface from the two supplied hydroelastic geometries. */
GTEST_TEST(FineTetReferenceTest,
           DistanceToSurfaceFindsNearestFaceEdgeAndCorner) {
  const FineTetReference reference = MakeSquareReference();
  EXPECT_NEAR(reference.distance_to_surface(Eigen::Vector3d(0.2, 0.3, 0.0)),
              0.0, 2e-14);
  EXPECT_NEAR(reference.distance_to_surface(Eigen::Vector3d(0.2, 0.3, 0.4)),
              0.4, 2e-14);
  EXPECT_NEAR(reference.distance_to_surface(Eigen::Vector3d(1.3, 0.4, 0.4)),
              0.5, 2e-14);
  EXPECT_NEAR(reference.distance_to_surface(Eigen::Vector3d(1.3, 1.4, 0.4)),
              std::sqrt(0.3 * 0.3 + 0.4 * 0.4 + 0.4 * 0.4), 2e-14);
  EXPECT_NEAR(reference.pressure_at(Eigen::Vector3d(0.2, 0.3, 0.4)), 1.5,
              2e-14);
}

GTEST_TEST(FineTetReferenceTest, PressureIsInterpolatedAtNearestSurfacePoint) {
  const FineTetReference reference = MakeSphereBoxReference(0.005);
  const AnalyticParaboloid analytic(kRadius, kPenetration, kModulus, kModulus);
  const Eigen::Vector3d axis_above(0.0, 0.0, kRadius + 0.004);
  EXPECT_NEAR(reference.pressure_at(axis_above), analytic.peak_pressure(),
              0.01 * analytic.peak_pressure());
  const Eigen::Vector3d outside(0.2, 0.0, kRadius);
  EXPECT_NEAR(reference.pressure_at(outside), 0.0,
              1e-12 * analytic.peak_pressure());
}

GTEST_TEST(FineTetReferenceTest, SphereBoxConvergesToAnalyticParaboloid) {
  const AnalyticParaboloid analytic(kRadius, kPenetration, kModulus, kModulus);
  const std::array<double, 3> hints{0.01, 0.005, 0.0025};
  std::array<Errors, 3> errors;
  for (int i = 0; i < ssize(hints); ++i) {
    errors[i] = CalcErrors(MakeSphereBoxReference(hints[i]), analytic);
    SCOPED_TRACE("tet_resolution_hint=" + std::to_string(hints[i]));
    EXPECT_TRUE(std::isfinite(errors[i].force));
    EXPECT_TRUE(std::isfinite(errors[i].area));
    EXPECT_TRUE(std::isfinite(errors[i].centroid));
  }
  for (int i = 1; i < ssize(errors); ++i) {
    SCOPED_TRACE("refinement_index=" + std::to_string(i));
    EXPECT_LT(errors[i].force, errors[i - 1].force);
    EXPECT_LT(errors[i].area, errors[i - 1].area);
    EXPECT_LT(errors[i].centroid, errors[i - 1].centroid);
  }
  EXPECT_LT(errors.back().force, 0.002);
  EXPECT_LT(errors.back().area, 0.002);
  EXPECT_LT(errors.back().centroid, 2.0e-4);
}

/* Opt-in measurement mode used to price the reference without making the full
 test suite pay for sub-millimeter meshes. Run the built test binary directly
 with FINE_TET_REFERENCE_HINT_M set; /usr/bin/time -v supplies peak RSS. */
GTEST_TEST(FineTetReferenceTest, MeasureSphereBoxWhenRequested) {
  const char* const hint_text = std::getenv("FINE_TET_REFERENCE_HINT_M");
  if (hint_text == nullptr) GTEST_SKIP();
  const double hint = std::stod(hint_text);
  const AnalyticParaboloid analytic(kRadius, kPenetration, kModulus, kModulus);
  const auto start = std::chrono::steady_clock::now();
  const FineTetReference fine = MakeSphereBoxReference(hint);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  const Errors errors = CalcErrors(fine, analytic);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "FINE_TET_MEASUREMENT hint_m=" << hint
            << " force_N=" << fine.force()
            << " force_relative_error=" << errors.force
            << " area_m2=" << fine.area()
            << " area_relative_error=" << errors.area
            << " centroid_error_m=" << errors.centroid
            << " construction_wall_s=" << elapsed << std::endl;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
