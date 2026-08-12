#include "drake/tools/voxel_sdf_experiments/common/mesh_export.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "drake/tools/voxel_sdf_experiments/common/components.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

void WriteSurfaceVtk(const std::filesystem::path& path,
                     const SurfaceView& surface, std::string_view title) {
  if (surface.faces.empty()) {
    throw std::logic_error("Cannot write a mesh for an empty surface");
  }
  if (surface.num_vertices != std::ssize(surface.vertices_W)) {
    throw std::logic_error("Surface vertex count does not match its data");
  }
  if (title.find('\n') != std::string_view::npos) {
    throw std::logic_error("VTK title must be a single line");
  }

  int connectivity_size = 0;
  for (const Face& face : surface.faces) {
    if (face.vertex_indices.size() < 3) {
      throw std::logic_error("Surface face has fewer than three vertices");
    }
    for (const int vertex : face.vertex_indices) {
      if (vertex < 0 || vertex >= std::ssize(surface.vertices_W)) {
        throw std::logic_error("Surface face has an invalid vertex index");
      }
    }
    connectivity_size += 1 + std::ssize(face.vertex_indices);
  }

  const std::vector<int> component_ids =
      CalcFaceComponentIds(surface, DefaultComponentTolerance(surface));

  if (path.has_parent_path() && !path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error(
        fmt::format("Cannot open '{}' for writing", path.string()));
  }

  file << "# vtk DataFile Version 3.0\n";
  file << title << "\n";
  file << "ASCII\n";
  file << "DATASET POLYDATA\n";

  file << fmt::format("POINTS {} double\n", surface.vertices_W.size());
  for (const Eigen::Vector3d& vertex_W : surface.vertices_W) {
    file << fmt::format("{:.17g} {:.17g} {:.17g}\n", vertex_W.x(), vertex_W.y(),
                        vertex_W.z());
  }

  file << fmt::format("POLYGONS {} {}\n", surface.faces.size(),
                      connectivity_size);
  for (const Face& face : surface.faces) {
    file << face.vertex_indices.size();
    for (const int vertex : face.vertex_indices) {
      file << " " << vertex;
    }
    file << "\n";
  }

  file << fmt::format("CELL_DATA {}\n", surface.faces.size());
  file << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
  for (const Face& face : surface.faces) {
    file << fmt::format("{:.17g}\n", face.pressure);
  }
  file << "SCALARS area double 1\nLOOKUP_TABLE default\n";
  for (const Face& face : surface.faces) {
    file << fmt::format("{:.17g}\n", face.area);
  }
  file << "SCALARS component_id int 1\nLOOKUP_TABLE default\n";
  for (const int component_id : component_ids) {
    file << component_id << "\n";
  }

  file.flush();
  if (!file) {
    throw std::runtime_error(
        fmt::format("Failed while writing '{}'", path.string()));
  }
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
