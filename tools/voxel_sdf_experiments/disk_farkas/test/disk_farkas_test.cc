#include "drake/tools/voxel_sdf_experiments/disk_farkas/disk_farkas.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

std::filesystem::path ScratchPath(const std::string& name) {
  const char* const directory = std::getenv("TEST_TMPDIR");
  return std::filesystem::path(directory != nullptr ? directory : ".") / name;
}

DiskFarkasConfig MakeSmokeConfig(Representation representation) {
  DiskFarkasConfig config;
  config.scene =
      LoadDiskScene("tools/voxel_sdf_experiments/disk_farkas/disk_plane.yaml");
  config.representation = representation;
  config.resolution = config.scene.mesh.sdf_target_voxel_size;
  config.time_step = 1.25e-4;
  config.settle_time = 0.05;
  config.settle_time_step = 6.25e-5;
  config.frames_per_second = 2000.0;
  config.num_frames = 300;
  config.output =
      ScratchPath(std::string(to_string(representation)) + "_disk.csv");
  return config;
}

/* This is deliberately an infrastructure smoke test, not a convergence gate.
 It uses the YAML's inexpensive 2.5 mm grid and loose physical bands. The key
 correctness check is the settle-to-measurement handoff: a discrete plant fixes
 its time step at construction, so the 62.5 us settle plant and 125 us motion
 plant are distinct, and every discrete-state group must copy exactly before
 the kick. A bad handoff can still produce a plausible eps curve. */
GTEST_TEST(DiskFarkasTest, AllRepresentationsSettleSlideSpinAndDecay) {
  const std::array<Representation, 3> representations{
      Representation::kTet, Representation::kPlaneClip,
      Representation::kMarchingCubes};
  for (const Representation representation : representations) {
    const DiskFarkasConfig config = MakeSmokeConfig(representation);
    const DiskFarkasResult result = RunDiskFarkas(config);
    const std::string label(to_string(representation));

    EXPECT_TRUE(result.used_distinct_settle_plant) << label;
    EXPECT_GT(result.transferred_discrete_state_groups, 0) << label;
    EXPECT_TRUE(result.state_transfer_exact) << label;
    EXPECT_TRUE(result.contact_acquired) << label;
    EXPECT_GT(result.post_kick_contact_samples, 0) << label;
    EXPECT_NEAR(result.settled_normal_load, result.weight, 0.35 * result.weight)
        << label;

    EXPECT_TRUE(std::isfinite(result.initial_epsilon)) << label;
    EXPECT_TRUE(std::isfinite(result.terminal_epsilon)) << label;
    EXPECT_GT(result.initial_epsilon, 1.2) << label;
    EXPECT_LT(result.terminal_epsilon, result.initial_epsilon - 0.1) << label;
    EXPECT_LT(std::abs(result.terminal_epsilon - 0.653),
              std::abs(result.initial_epsilon - 0.653))
        << label;
    EXPECT_GT(result.initial_linear_speed, 0.1) << label;
    EXPECT_GT(result.initial_spin_speed, 5.0) << label;
    EXPECT_LT(result.final_linear_speed, 0.8 * result.initial_linear_speed)
        << label;
    EXPECT_LT(result.final_spin_speed, 0.8 * result.initial_spin_speed)
        << label;

    const auto first_post_kick = std::find_if(
        result.rows.begin(), result.rows.end(), [](const DiskFarkasRow& row) {
          return row.post_kick;
        });
    ASSERT_NE(first_post_kick, result.rows.end()) << label;
    EXPECT_GT(first_post_kick->linear_speed, 0.1) << label;
    EXPECT_GT(std::abs(first_post_kick->angular_velocity_WD.z()), 5.0) << label;
    EXPECT_GT(
        result.rows.back().position_WD.x() - first_post_kick->position_WD.x(),
        0.001)
        << label;

    std::ifstream input(config.output);
    ASSERT_TRUE(input) << config.output;
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
    EXPECT_EQ(line, DiskFarkasCsvHeader());
    int csv_rows = 0;
    int post_kick_rows = 0;
    while (std::getline(input, line)) {
      ++csv_rows;
      if (line.ends_with(",true")) ++post_kick_rows;
    }
    EXPECT_EQ(csv_rows, result.rows.size()) << label;
    EXPECT_EQ(post_kick_rows, config.num_frames) << label;
  }
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
