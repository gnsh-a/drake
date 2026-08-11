#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "drake/tools/voxel_sdf_experiments/common/metrics.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

struct RunMetadata {
  std::string scene;
  std::string representation;
  double penetration{};
  double voxel_width{};
  double tet_resolution_hint{};
  double radius{};
  Eigen::Vector3d box_half_widths{Eigen::Vector3d::Zero()};
  double modulus_a{};
  double modulus_b{};
  Eigen::Vector3d grid_rpy_deg{Eigen::Vector3d::Zero()};
};

struct GitProvenance {
  std::string commit;
  bool dirty{};
};

struct RunRecord {
  RunMetadata metadata;
  GitProvenance provenance;
  Metrics metrics;
};

std::string_view CsvHeader();
void EmitCsv(const std::filesystem::path& path, const RunRecord& record);
GitProvenance ReadGitProvenance();

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
