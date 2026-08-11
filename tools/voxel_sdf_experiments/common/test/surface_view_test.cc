#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/geometry/proximity/polygon_surface_mesh_field.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh_field.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using Eigen::Vector3d;
using geometry::ContactSurface;
using geometry::GeometryId;
using geometry::PolygonSurfaceMesh;
using geometry::PolygonSurfaceMeshFieldLinear;
using geometry::SurfaceTriangle;
using geometry::TriangleSurfaceMesh;
using geometry::TriangleSurfaceMeshFieldLinear;

std::vector<Vector3d> MakeVertices() {
  return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
}

ContactSurface<double> MakeTriangleSurface() {
  auto mesh = std::make_unique<TriangleSurfaceMesh<double>>(
      std::vector<SurfaceTriangle>{{0, 1, 2}, {2, 3, 0}}, MakeVertices());
  auto field = std::make_unique<TriangleSurfaceMeshFieldLinear<double, double>>(
      std::vector<double>{1.0, 2.0, 3.0, 2.0}, mesh.get());
  return ContactSurface<double>(GeometryId::get_new_id(),
                                GeometryId::get_new_id(), std::move(mesh),
                                std::move(field));
}

ContactSurface<double> MakePolygonSurface() {
  auto mesh = std::make_unique<PolygonSurfaceMesh<double>>(
      std::vector<int>{3, 0, 1, 2, 3, 2, 3, 0}, MakeVertices());
  auto field = std::make_unique<PolygonSurfaceMeshFieldLinear<double, double>>(
      std::vector<double>{1.0, 2.0, 3.0, 2.0}, mesh.get(),
      std::vector<Vector3d>{Vector3d(1.0, 1.0, 0.0), Vector3d(1.0, 1.0, 0.0)});
  return ContactSurface<double>(GeometryId::get_new_id(),
                                GeometryId::get_new_id(), std::move(mesh),
                                std::move(field));
}

std::pair<double, double> TotalAreaAndMeanPressure(const SurfaceView& view) {
  double total_area = 0.0;
  double pressure_integral = 0.0;
  for (const Face& face : view.faces) {
    total_area += face.area;
    pressure_integral += face.area * face.pressure;
  }
  return {total_area, pressure_integral / total_area};
}

GTEST_TEST(SurfaceViewTest, TriangleAndPolygonAreEquivalent) {
  const SurfaceView triangle = MakeSurfaceView(MakeTriangleSurface());
  const SurfaceView polygon = MakeSurfaceView(MakePolygonSurface());
  const auto [triangle_area, triangle_pressure] =
      TotalAreaAndMeanPressure(triangle);
  const auto [polygon_area, polygon_pressure] =
      TotalAreaAndMeanPressure(polygon);
  EXPECT_EQ(triangle.num_vertices, polygon.num_vertices);
  EXPECT_EQ(triangle.faces.size(), polygon.faces.size());
  EXPECT_DOUBLE_EQ(triangle_area, 1.0);
  EXPECT_DOUBLE_EQ(polygon_area, triangle_area);
  EXPECT_DOUBLE_EQ(triangle_pressure, 2.0);
  EXPECT_DOUBLE_EQ(polygon_pressure, triangle_pressure);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
