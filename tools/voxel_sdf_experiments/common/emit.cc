#include "drake/tools/voxel_sdf_experiments/common/emit.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

constexpr std::string_view kCsvHeader =
    "schema_version,git_commit,git_dirty,scene,representation,penetration_m,"
    "voxel_width_m,tet_resolution_hint_m,radius_m,box_half_width_x_m,"
    "box_half_width_y_m,box_half_width_z_m,modulus_a_pa,modulus_b_pa,"
    "grid_roll_deg,grid_pitch_deg,grid_yaw_deg,surface_distance_rms_m,"
    "surface_distance_max_m,normal_force_n,reference_force_n,"
    "normal_force_relative_error,pressure_error_rms_pa,"
    "pressure_error_max_pa,peak_pressure_pa,reference_peak_pressure_pa,"
    "peak_pressure_relative_error,projected_area_m2,reference_area_m2,"
    "area_relative_error,total_area_m2,reference_surface_area_m2,"
    "total_area_relative_error,"
    "patch_radius_m,reference_patch_radius_m,"
    "patch_radius_relative_error,num_faces,num_vertices,centroid_x_m,"
    "centroid_y_m,centroid_z_m,centroid_position_error_m,"
    "center_of_pressure_x_m,center_of_pressure_y_m,center_of_pressure_z_m,"
    "center_of_pressure_error_m,spurious_moment_nm,"
    "spurious_moment_normalized,"
    "largest_component_area_fraction,num_components";

std::string EscapeCsv(std::string_view value) {
  if (value.find_first_of(",\"\n\r") == std::string_view::npos) {
    return std::string(value);
  }
  std::string result{"\""};
  for (const char character : value) {
    if (character == '\"') result.push_back('\"');
    result.push_back(character);
  }
  result.push_back('\"');
  return result;
}

std::string RunCommand(const char* command) {
  FILE* const pipe = popen(command, "r");
  if (pipe == nullptr) {
    throw std::runtime_error("Unable to launch git for provenance");
  }
  std::array<char, 256> buffer{};
  std::string output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output += buffer.data();
  }
  if (pclose(pipe) != 0) {
    throw std::runtime_error("Git provenance command failed");
  }
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

}  // namespace

std::string_view CsvHeader() {
  return kCsvHeader;
}

void EmitCsv(const std::filesystem::path& path, const RunRecord& record) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to create CSV file '" + path.string() +
                             "'");
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  const RunMetadata& input = record.metadata;
  const Metrics& metrics = record.metrics;
  output << kCsvHeader << '\n';
  output << "2," << EscapeCsv(record.provenance.commit) << ','
         << (record.provenance.dirty ? "true" : "false") << ','
         << EscapeCsv(input.scene) << ',' << EscapeCsv(input.representation)
         << ',' << input.penetration << ',' << input.voxel_width << ','
         << input.tet_resolution_hint << ',' << input.radius << ','
         << input.box_half_widths.x() << ',' << input.box_half_widths.y() << ','
         << input.box_half_widths.z() << ',' << input.modulus_a << ','
         << input.modulus_b << ',' << input.grid_rpy_deg.x() << ','
         << input.grid_rpy_deg.y() << ',' << input.grid_rpy_deg.z() << ','
         << metrics.surface_distance_rms << ',' << metrics.surface_distance_max
         << ',' << metrics.normal_force << ',' << metrics.reference_force << ','
         << metrics.normal_force_relative_error << ','
         << metrics.pressure_error_rms << ',' << metrics.pressure_error_max
         << ',' << metrics.peak_pressure << ','
         << metrics.reference_peak_pressure << ','
         << metrics.peak_pressure_relative_error << ','
         << metrics.projected_area << ',' << metrics.reference_area << ','
         << metrics.area_relative_error << ',' << metrics.total_area << ','
         << metrics.reference_surface_area << ','
         << metrics.total_area_relative_error << ',' << metrics.patch_radius
         << ',' << metrics.reference_patch_radius << ','
         << metrics.patch_radius_relative_error << ',' << metrics.num_faces
         << ',' << metrics.num_vertices << ',' << metrics.centroid_W.x() << ','
         << metrics.centroid_W.y() << ',' << metrics.centroid_W.z() << ','
         << metrics.centroid_position_error << ','
         << metrics.center_of_pressure_W.x() << ','
         << metrics.center_of_pressure_W.y() << ','
         << metrics.center_of_pressure_W.z() << ','
         << metrics.center_of_pressure_error << ',' << metrics.spurious_moment
         << ',' << metrics.spurious_moment_normalized << ','
         << metrics.largest_component_area_fraction << ','
         << metrics.num_components << '\n';
  if (!output) {
    throw std::runtime_error("Failed while writing CSV file '" + path.string() +
                             "'");
  }
}

GitProvenance ReadGitProvenance() {
  GitProvenance result;
  result.commit = RunCommand("git rev-parse HEAD 2>/dev/null");
  if (result.commit.empty()) {
    throw std::runtime_error("git rev-parse did not return a commit");
  }
  result.dirty =
      !RunCommand("git status --porcelain --untracked-files=normal 2>/dev/null")
           .empty();
  return result;
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
