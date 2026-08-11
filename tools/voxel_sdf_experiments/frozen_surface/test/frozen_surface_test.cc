#include "drake/tools/voxel_sdf_experiments/frozen_surface/frozen_surface.h"

#include <cmath>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

GTEST_TEST(FrozenSurfaceTest, CoarseEndToEndSmoke) {
  FrozenSurfaceConfig config;
  config.scene = Scene::kSphereSphere;
  config.representation = Representation::kPlaneClip;
  config.voxel_width = 0.02;
  config.tet_resolution_hint = 0.02;
  const Metrics metrics = RunOne(config);
  EXPECT_GT(metrics.num_faces, 0);
  EXPECT_GT(metrics.num_vertices, 0);
  EXPECT_TRUE(std::isfinite(metrics.normal_force_relative_error));
  EXPECT_LT(metrics.normal_force_relative_error, 0.5);
  EXPECT_GT(metrics.largest_component_area_fraction, 0.0);
  EXPECT_LE(metrics.largest_component_area_fraction, 1.0);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
