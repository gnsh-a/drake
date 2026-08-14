#include "drake/tools/voxel_sdf_experiments/frozen_spatula/frozen_spatula.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;

GTEST_TEST(FrozenSpatulaTest, DemoPoseIsLoadedFromTheSdfScene) {
  FrozenSpatulaConfig config;
  config.pose = SpatulaPose::kFirstTouch;
  const FrozenSpatulaResult result = RunFrozenSpatula(config);
  const Matrix3d expected_rotation =
      (Matrix3d() << 0.000000634136, 0.999999682932, -0.000796326458,
       0.999596538461, -0.000023252360, -0.028403516625, -0.028403526136,
       -0.000795987159, -0.999596221535)
          .finished();
  const Vector3d expected_translation(-0.051798766552, 0.003847168533,
                                      -0.023488225020);
  EXPECT_TRUE(CompareMatrices(result.X_CE_demo.rotation().matrix(),
                              expected_rotation, 1.0e-12));
  EXPECT_TRUE(CompareMatrices(result.X_CE_demo.translation(),
                              expected_translation, 1.0e-12));
  EXPECT_EQ(result.ellipsoid_base_resolution, 0.04);
  EXPECT_EQ(result.cylinder_base_resolution, 0.005);
  EXPECT_EQ(result.ellipsoid_modulus, 1.0e5);
  EXPECT_EQ(result.cylinder_modulus, 1.0e8);
  EXPECT_GE(result.rigid_signed_distance, 0.0);
  EXPECT_LE(result.rigid_signed_distance, 1.0e-6);
  EXPECT_FALSE(result.reference.available);
  for (const FrozenSpatulaRepresentationResult& representation :
       result.representations) {
    EXPECT_FALSE(representation.metrics.has_nonfinite)
        << to_string(representation.representation);
  }
}

/* This is the end-to-end correctness gate for the study. It checks all three
 representation keys at the actual demo pose, including the required output
 surface type, and verifies that every physical error channel is finite against
 an independently constructed finer tetrahedral reference. */
GTEST_TEST(FrozenSpatulaTest, DemoPoseIsGenuinelyThreeWayAndReferenced) {
  FrozenSpatulaConfig config;
  config.fine_tet_resolution_hint = 0.0025;
  const FrozenSpatulaResult result = RunFrozenSpatula(config);
  EXPECT_TRUE(result.reference.available);
  EXPECT_GT(result.reference.force, 0.0);
  EXPECT_GT(result.reference.projected_area, 0.0);
  EXPECT_LT(result.rigid_signed_distance, 0.0);
  const std::array<Representation, 3> expected{Representation::kTet,
                                               Representation::kPlaneClip,
                                               Representation::kMarchingCubes};
  for (int i = 0; i < ssize(expected); ++i) {
    const auto& representation = result.representations[i];
    const FrozenSpatulaMetrics& metrics = representation.metrics;
    const std::string label(to_string(representation.representation));
    EXPECT_EQ(representation.representation, expected[i]);
    EXPECT_EQ(metrics.is_triangle,
              expected[i] == Representation::kMarchingCubes)
        << label;
    EXPECT_TRUE(metrics.in_contact) << label;
    EXPECT_GT(metrics.num_faces, 0) << label;
    EXPECT_GT(metrics.num_vertices, 0) << label;
    EXPECT_GT(metrics.total_surface_area, 0.0) << label;
    EXPECT_GT(metrics.projected_area, 0.0) << label;
    EXPECT_GT(metrics.force_norm, 0.0) << label;
    EXPECT_TRUE(std::isfinite(metrics.force_relative_error)) << label;
    EXPECT_TRUE(std::isfinite(metrics.area_relative_error)) << label;
    EXPECT_TRUE(std::isfinite(metrics.surface_distance_rms)) << label;
    EXPECT_TRUE(std::isfinite(metrics.surface_distance_max)) << label;
    EXPECT_TRUE(std::isfinite(metrics.pressure_error_rms)) << label;
    EXPECT_TRUE(std::isfinite(metrics.pressure_error_max)) << label;
    EXPECT_TRUE(std::isfinite(metrics.centroid_position_error)) << label;
    EXPECT_GE(metrics.connected_components, 1) << label;
    EXPECT_GT(metrics.largest_component_area_fraction, 0.0) << label;
    EXPECT_LE(metrics.largest_component_area_fraction, 1.0) << label;
    EXPECT_FALSE(metrics.has_nonfinite) << label;
    EXPECT_FALSE(metrics.has_negative_pressure) << label;
  }
}

GTEST_TEST(FrozenSpatulaTest, RejectsInvalidInputs) {
  EXPECT_THROW(ParseSpatulaPose("other"), std::logic_error);
  FrozenSpatulaConfig config;
  config.penetration = -0.001;
  EXPECT_THROW(RunFrozenSpatula(config), std::logic_error);
  config.penetration = 0.0;
  config.resolution_scale = 0.0;
  EXPECT_THROW(RunFrozenSpatula(config), std::logic_error);
}

/* Opt-in measurement mode for the geometry-specific FineTetReference floor.
 Run the built test binary directly with FROZEN_SPATULA_FINE_TET_HINT_M set;
 /usr/bin/time -v supplies peak RSS without burdening ordinary test runs. */
GTEST_TEST(FrozenSpatulaTest, MeasureFineTetFloorWhenRequested) {
  const char* const hint_text = std::getenv("FROZEN_SPATULA_FINE_TET_HINT_M");
  if (hint_text == nullptr) GTEST_SKIP();
  const double hint = std::stod(hint_text);
  const FineSpatulaReferenceValues reference =
      MeasureFrozenSpatulaReference(FrozenSpatulaConfig{}, hint);
  ASSERT_TRUE(reference.available);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "FROZEN_SPATULA_FINE_TET_MEASUREMENT hint_m=" << hint
            << " force_N=" << reference.force
            << " projected_area_m2=" << reference.projected_area
            << " centroid_x_m=" << reference.centroid_C.x()
            << " centroid_y_m=" << reference.centroid_C.y()
            << " centroid_z_m=" << reference.centroid_C.z()
            << " construction_wall_s=" << reference.construction_wall_time
            << std::endl;
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
