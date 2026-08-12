#include "drake/tools/voxel_sdf_experiments/common/components.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

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
}  // namespace

double DefaultComponentTolerance(const SurfaceView& surface) {
  double coordinate_scale = 1.0;
  for (const Eigen::Vector3d& vertex_W : surface.vertices_W) {
    coordinate_scale =
        std::max(coordinate_scale, vertex_W.cwiseAbs().maxCoeff());
  }
  return 1e-10 * coordinate_scale;
}

std::vector<int> CalcFaceComponentIds(const SurfaceView& surface,
                                      double tolerance) {
  if (!(std::isfinite(tolerance) && tolerance > 0.0)) {
    throw std::logic_error("Component tolerance must be finite and positive");
  }
  DisjointSet components(surface.faces.size());
  std::unordered_map<VertexBucket, std::vector<VertexOccurrence>,
                     VertexBucketHash>
      occurrences;
  for (int face_index = 0; face_index < std::ssize(surface.faces);
       ++face_index) {
    for (const int vertex : surface.faces[face_index].vertex_indices) {
      if (vertex < 0 || vertex >= std::ssize(surface.vertices_W)) {
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
  std::vector<int> ids(surface.faces.size());
  for (int face_index = 0; face_index < std::ssize(surface.faces);
       ++face_index) {
    ids[face_index] = components.Find(face_index);
  }
  return ids;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
