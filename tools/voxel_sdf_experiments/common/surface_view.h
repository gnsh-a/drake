#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/query_results/contact_surface.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

struct Face {
  Eigen::Vector3d centroid_W{Eigen::Vector3d::Zero()};
  double area{};
  double pressure{};
  Eigen::Vector3d normal_W{Eigen::Vector3d::Zero()};
  std::vector<int> vertex_indices;
};

struct SurfaceView {
  int num_vertices{};
  std::vector<Eigen::Vector3d> vertices_W;
  std::vector<Face> faces;
};

SurfaceView MakeSurfaceView(const geometry::ContactSurface<double>& surface);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
