#include "drake/tools/voxel_sdf_experiments/common/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

double RelativeError(double value, double reference) {
  if (!(std::isfinite(reference) && reference > 0.0)) {
    throw std::logic_error("Metric reference must be finite and positive");
  }
  return std::abs(value - reference) / reference;
}

class DisjointSet {
 public:
  explicit DisjointSet(int size) : parent_(size), rank_(size) {
    for (int i = 0; i < size; ++i) parent_[i] = i;
  }

  int Find(int value) {
    if (parent_[value] != value) parent_[value] = Find(parent_[value]);
    return parent_[value];
  }

  void Union(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
  }

 private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

struct VertexBucket {
  std::array<int64_t, 3> coordinate{};

  bool operator==(const VertexBucket&) const = default;
};

struct VertexBucketHash {
  size_t operator()(const VertexBucket& bucket) const {
    size_t result = 0;
    for (const int64_t coordinate : bucket.coordinate) {
      result ^= std::hash<int64_t>{}(coordinate) + 0x9e3779b9 + (result << 6) +
                (result >> 2);
    }
    return result;
  }
};

struct VertexOccurrence {
  Eigen::Vector3d position_W;
  int face{};
};

double CalcLargestComponentAreaFraction(const SurfaceView& surface,
                                        double total_area) {
  DisjointSet components(surface.faces.size());
  double coordinate_scale = 1.0;
  for (const Eigen::Vector3d& vertex_W : surface.vertices_W) {
    coordinate_scale =
        std::max(coordinate_scale, vertex_W.cwiseAbs().maxCoeff());
  }
  const double tolerance = 1e-10 * coordinate_scale;
  std::unordered_map<VertexBucket, std::vector<VertexOccurrence>,
                     VertexBucketHash>
      occurrences;
  for (int face_index = 0; face_index < ssize(surface.faces); ++face_index) {
    for (const int vertex : surface.faces[face_index].vertex_indices) {
      if (vertex < 0 || vertex >= ssize(surface.vertices_W)) {
        throw std::logic_error("Surface face has an invalid vertex index");
      }
      const Eigen::Vector3d& position_W = surface.vertices_W[vertex];
      VertexBucket bucket;
      for (int axis = 0; axis < 3; ++axis) {
        bucket.coordinate[axis] =
            static_cast<int64_t>(std::floor(position_W[axis] / tolerance));
      }
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            VertexBucket neighbor = bucket;
            neighbor.coordinate[0] += dx;
            neighbor.coordinate[1] += dy;
            neighbor.coordinate[2] += dz;
            const auto iter = occurrences.find(neighbor);
            if (iter == occurrences.end()) continue;
            for (const VertexOccurrence& occurrence : iter->second) {
              if ((occurrence.position_W - position_W).norm() <= tolerance) {
                components.Union(face_index, occurrence.face);
              }
            }
          }
        }
      }
      occurrences[bucket].push_back({position_W, face_index});
    }
  }
  std::vector<double> component_area(surface.faces.size(), 0.0);
  for (int face_index = 0; face_index < ssize(surface.faces); ++face_index) {
    component_area[components.Find(face_index)] +=
        surface.faces[face_index].area;
  }
  return *std::max_element(component_area.begin(), component_area.end()) /
         total_area;
}

}  // namespace

Metrics CalcMetrics(const SurfaceView& surface, const Reference& reference) {
  if (surface.faces.empty()) {
    throw std::logic_error("Cannot calculate metrics for an empty surface");
  }
  if (surface.num_vertices != ssize(surface.vertices_W)) {
    throw std::logic_error("Surface vertex count does not match its data");
  }

  Metrics result;
  result.num_faces = surface.faces.size();
  result.num_vertices = surface.num_vertices;
  result.reference_force = reference.force();
  result.reference_peak_pressure = reference.peak_pressure();
  result.reference_area = reference.area();
  result.reference_patch_radius = reference.patch_radius();
  result.peak_pressure = -std::numeric_limits<double>::infinity();

  const Eigen::Vector3d reference_normal = reference.normal().normalized();
  double total_area = 0.0;
  double squared_surface_error_integral = 0.0;
  double squared_pressure_error_integral = 0.0;
  Eigen::Vector3d force_W = Eigen::Vector3d::Zero();
  Eigen::Vector3d centroid_integral_W = Eigen::Vector3d::Zero();
  for (const Face& face : surface.faces) {
    if (!(std::isfinite(face.area) && face.area >= 0.0 &&
          std::isfinite(face.pressure) && face.centroid_W.allFinite() &&
          face.normal_W.allFinite())) {
      throw std::logic_error("Surface face data must be finite");
    }
    total_area += face.area;
    centroid_integral_W += face.area * face.centroid_W;
    const double surface_error = reference.distance_to_surface(face.centroid_W);
    squared_surface_error_integral += face.area * surface_error * surface_error;
    result.surface_distance_max =
        std::max(result.surface_distance_max, surface_error);

    const double pressure_error =
        face.pressure - reference.pressure_at(face.centroid_W);
    squared_pressure_error_integral +=
        face.area * pressure_error * pressure_error;
    result.pressure_error_max =
        std::max(result.pressure_error_max, std::abs(pressure_error));
    result.peak_pressure = std::max(result.peak_pressure, face.pressure);

    force_W += face.area * face.pressure * face.normal_W;
    result.projected_area +=
        face.area * std::abs(face.normal_W.dot(reference_normal));
  }
  if (!(total_area > 0.0 && result.projected_area > 0.0)) {
    throw std::logic_error("Surface must have positive area and projection");
  }

  result.surface_distance_rms =
      std::sqrt(squared_surface_error_integral / total_area);
  result.pressure_error_rms =
      std::sqrt(squared_pressure_error_integral / total_area);
  result.normal_force = std::abs(force_W.dot(reference_normal));
  result.normal_force_relative_error =
      RelativeError(result.normal_force, result.reference_force);
  result.peak_pressure_relative_error =
      RelativeError(result.peak_pressure, result.reference_peak_pressure);
  result.area_relative_error =
      RelativeError(result.projected_area, result.reference_area);

  result.centroid_W = centroid_integral_W / total_area;
  result.centroid_position_error =
      (result.centroid_W - reference.centroid()).norm();

  const Eigen::Vector3d reference_centroid = reference.centroid();
  for (const Eigen::Vector3d& vertex_W : surface.vertices_W) {
    const Eigen::Vector3d offset = vertex_W - reference_centroid;
    const double radial_distance =
        (offset - offset.dot(reference_normal) * reference_normal).norm();
    result.patch_radius = std::max(result.patch_radius, radial_distance);
  }
  result.patch_radius_relative_error =
      RelativeError(result.patch_radius, result.reference_patch_radius);
  result.largest_component_area_fraction =
      CalcLargestComponentAreaFraction(surface, total_area);
  return result;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
