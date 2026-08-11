#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

#include <utility>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

SurfaceView MakeSurfaceView(const geometry::ContactSurface<double>& surface) {
  SurfaceView result;
  result.num_vertices = surface.num_vertices();
  result.vertices_W.reserve(surface.num_vertices());
  result.faces.reserve(surface.num_faces());
  if (surface.is_triangle()) {
    const auto& mesh_W = surface.tri_mesh_W();
    const auto& pressure = surface.tri_e_MN();
    for (int vertex = 0; vertex < mesh_W.num_vertices(); ++vertex) {
      result.vertices_W.push_back(mesh_W.vertex(vertex));
    }
    for (int face_index = 0; face_index < mesh_W.num_elements(); ++face_index) {
      const Eigen::Vector3d centroid_W = mesh_W.element_centroid(face_index);
      const auto& triangle = mesh_W.element(face_index);
      Face face{
          .centroid_W = centroid_W,
          .area = mesh_W.area(face_index),
          .pressure = pressure.EvaluateCartesian(face_index, centroid_W),
          .normal_W = mesh_W.face_normal(face_index),
          .vertex_indices = {triangle.vertex(0), triangle.vertex(1),
                             triangle.vertex(2)},
      };
      result.faces.push_back(std::move(face));
    }
    return result;
  }

  const auto& mesh_W = surface.poly_mesh_W();
  const auto& pressure = surface.poly_e_MN();
  for (int vertex = 0; vertex < mesh_W.num_vertices(); ++vertex) {
    result.vertices_W.push_back(mesh_W.vertex(vertex));
  }
  for (int face_index = 0; face_index < mesh_W.num_elements(); ++face_index) {
    const Eigen::Vector3d centroid_W = mesh_W.element_centroid(face_index);
    const auto polygon = mesh_W.element(face_index);
    std::vector<int> vertex_indices;
    vertex_indices.reserve(polygon.num_vertices());
    for (int vertex = 0; vertex < polygon.num_vertices(); ++vertex) {
      vertex_indices.push_back(polygon.vertex(vertex));
    }
    Face face{
        .centroid_W = centroid_W,
        .area = mesh_W.area(face_index),
        .pressure = pressure.EvaluateCartesian(face_index, centroid_W),
        .normal_W = mesh_W.face_normal(face_index),
        .vertex_indices = std::move(vertex_indices),
    };
    result.faces.push_back(std::move(face));
  }
  return result;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
