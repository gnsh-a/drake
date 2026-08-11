#include "drake/tools/voxel_sdf_experiments/common/metrics.h"

#include <cmath>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

class SquareReference final : public Reference {
 public:
  double force() const final { return 2.0; }
  double area() const final { return 1.0; }
  double distance_to_surface(const Eigen::Vector3d& p_RQ) const final {
    return std::abs(p_RQ.z());
  }
  double pressure_at(const Eigen::Vector3d&) const final { return 2.0; }
  double patch_radius() const final { return std::sqrt(0.5); }
  double peak_pressure() const final { return 2.0; }
  Eigen::Vector3d centroid() const final {
    return Eigen::Vector3d(0.5, 0.5, 0.0);
  }
  Eigen::Vector3d normal() const final { return Eigen::Vector3d::UnitZ(); }
};

SurfaceView MakeExactSquare() {
  return SurfaceView{.num_vertices = 4,
                     .vertices_W = {{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {0.0, 1.0, 0.0}},
                     .faces = {
                         Face{.centroid_W = {2.0 / 3.0, 1.0 / 3.0, 0.0},
                              .area = 0.5,
                              .pressure = 2.0,
                              .normal_W = Eigen::Vector3d::UnitZ(),
                              .vertex_indices = {0, 1, 2}},
                         Face{.centroid_W = {1.0 / 3.0, 2.0 / 3.0, 0.0},
                              .area = 0.5,
                              .pressure = 2.0,
                              .normal_W = Eigen::Vector3d::UnitZ(),
                              .vertex_indices = {2, 3, 0}},
                     }};
}

GTEST_TEST(MetricsTest, ExactFlatPatchHasZeroErrors) {
  const Metrics metrics = CalcMetrics(MakeExactSquare(), SquareReference());
  EXPECT_NEAR(metrics.surface_distance_rms, 0.0, 1e-15);
  EXPECT_NEAR(metrics.surface_distance_max, 0.0, 1e-15);
  EXPECT_NEAR(metrics.normal_force_relative_error, 0.0, 1e-15);
  EXPECT_NEAR(metrics.pressure_error_rms, 0.0, 1e-15);
  EXPECT_NEAR(metrics.pressure_error_max, 0.0, 1e-15);
  EXPECT_NEAR(metrics.peak_pressure_relative_error, 0.0, 1e-15);
  EXPECT_NEAR(metrics.area_relative_error, 0.0, 1e-15);
  EXPECT_NEAR(metrics.patch_radius_relative_error, 0.0, 1e-15);
  EXPECT_NEAR(metrics.centroid_position_error, 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(metrics.largest_component_area_fraction, 1.0);
  EXPECT_EQ(metrics.num_faces, 2);
  EXPECT_EQ(metrics.num_vertices, 4);
}

GTEST_TEST(MetricsTest, FragmentedPatchIsDetected) {
  SurfaceView fragmented = MakeExactSquare();
  fragmented.num_vertices = 6;
  fragmented.vertices_W.push_back({2.0, 0.0, 0.0});
  fragmented.vertices_W.push_back({2.0, 1.0, 0.0});
  fragmented.faces[1].vertex_indices = {3, 4, 5};
  const Metrics metrics = CalcMetrics(fragmented, SquareReference());
  EXPECT_DOUBLE_EQ(metrics.largest_component_area_fraction, 0.5);
}

GTEST_TEST(MetricsTest, GeometricallySharedVerticesConnectPolygonFaces) {
  SurfaceView duplicated = MakeExactSquare();
  duplicated.num_vertices = 7;
  duplicated.vertices_W.push_back(duplicated.vertices_W[2]);
  duplicated.vertices_W.push_back(duplicated.vertices_W[3]);
  duplicated.vertices_W.push_back(duplicated.vertices_W[0]);
  duplicated.faces[1].vertex_indices = {4, 5, 6};
  const Metrics metrics = CalcMetrics(duplicated, SquareReference());
  EXPECT_DOUBLE_EQ(metrics.largest_component_area_fraction, 1.0);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
