#include "drake/tools/voxel_sdf_experiments/common/emit.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

GTEST_TEST(EmitTest, SchemaIsStable) {
  constexpr std::string_view kExpectedHeader =
      "schema_version,git_commit,git_dirty,scene,representation,penetration_m,"
      "voxel_width_m,tet_resolution_hint_m,radius_m,box_half_width_x_m,"
      "box_half_width_y_m,box_half_width_z_m,modulus_a_pa,modulus_b_pa,"
      "grid_roll_deg,grid_pitch_deg,grid_yaw_deg,surface_distance_rms_m,"
      "surface_distance_max_m,normal_force_n,reference_force_n,"
      "normal_force_relative_error,pressure_error_rms_pa,"
      "pressure_error_max_pa,peak_pressure_pa,reference_peak_pressure_pa,"
      "peak_pressure_relative_error,projected_area_m2,reference_area_m2,"
      "area_relative_error,patch_radius_m,reference_patch_radius_m,"
      "patch_radius_relative_error,num_faces,num_vertices,centroid_x_m,"
      "centroid_y_m,centroid_z_m,centroid_position_error_m,"
      "largest_component_area_fraction";
  EXPECT_EQ(CsvHeader(), kExpectedHeader);

  RunRecord record;
  record.metadata.scene = "sphere_sphere";
  record.metadata.representation = "marching_cubes";
  record.provenance = {.commit = "0123456789abcdef", .dirty = true};
  record.metrics.num_faces = 12;
  record.metrics.num_vertices = 10;
  const char* const test_tmpdir = std::getenv("TEST_TMPDIR");
  ASSERT_NE(test_tmpdir, nullptr);
  const std::filesystem::path output_path =
      std::filesystem::path(test_tmpdir) / "emit_schema.csv";
  EmitCsv(output_path, record);

  std::ifstream input(output_path);
  ASSERT_TRUE(input);
  std::string header;
  std::string row;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, header)));
  ASSERT_TRUE(static_cast<bool>(std::getline(input, row)));
  EXPECT_EQ(header, kExpectedHeader);
  EXPECT_FALSE(row.empty());
  std::string extra;
  EXPECT_FALSE(static_cast<bool>(std::getline(input, extra)));
  EXPECT_EQ(std::count(header.begin(), header.end(), ','),
            std::count(row.begin(), row.end(), ','));
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
