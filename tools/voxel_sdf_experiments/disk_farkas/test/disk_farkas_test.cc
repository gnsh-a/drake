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

/* The initial slip-to-spin ratio is what the invariance study sweeps, and it
 is set indirectly: eps0 = |v| / (omega_z * R), so a driver picks a slide for a
 fixed spin. If the scene override were dropped the run would still produce a
 plausible trajectory at the YAML's own ratio, so this pins that the requested
 ratio is the one actually simulated. */
GTEST_TEST(DiskFarkasTest, SceneOverridesSetTheInitialRatio) {
  constexpr double kRadius = 0.01213;
  constexpr double kSpin = 12.0;
  for (const double target : {0.75, 2.0}) {
    DiskFarkasConfig config = MakeSmokeConfig(Representation::kPlaneClip);
    config.num_frames = 20;
    config.scene.disk.initial_linear_velocity =
        Eigen::Vector3d(target * kSpin * kRadius, 0.0, 0.0);
    config.scene.disk.initial_angular_velocity =
        Eigen::Vector3d(0.0, 0.0, kSpin);
    config.output = ScratchPath("eps0_" + std::to_string(target) + ".csv");
    const DiskFarkasResult result = RunDiskFarkas(config);
    EXPECT_NEAR(result.initial_epsilon, target, 1e-3) << target;
  }
}

/* Fragmentation of the contact patch, newly recorded here. The disk's flat
 face against a flat box is the geometry least likely to fragment, and the
 affine kernel does return one conforming component on it -- unlike the curved
 sphere contact in the settling study, where the same kernel shatters into
 hundreds. That contrast is the finding; what would be a regression is the
 columns going empty. */
GTEST_TEST(DiskFarkasTest, FragmentationIsRecordedPerFrame) {
  DiskFarkasConfig config = MakeSmokeConfig(Representation::kPlaneClip);
  config.num_frames = 20;
  config.output = ScratchPath("fragmentation.csv");
  const DiskFarkasResult result = RunDiskFarkas(config);

  int frames_with_contact = 0;
  for (const DiskFarkasRow& row : result.rows) {
    if (row.surface_faces == 0) continue;
    ++frames_with_contact;
    EXPECT_GE(row.num_components, 1);
    EXPECT_GT(row.largest_component_area_fraction, 0.0);
    EXPECT_LE(row.largest_component_area_fraction, 1.0 + 1e-12);
    /* One component must hold all of the area; many cannot. */
    if (row.num_components == 1) {
      EXPECT_NEAR(row.largest_component_area_fraction, 1.0, 1e-9);
    }
  }
  EXPECT_GT(frames_with_contact, 0);

  /* The header and every row must agree on width, including the two new
   columns. */
  std::ifstream input(config.output);
  ASSERT_TRUE(input);
  std::string header;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, header)));
  EXPECT_NE(header.find("num_components"), std::string::npos);
  EXPECT_NE(header.find("largest_component_area_fraction"), std::string::npos);
  const auto commas = std::count(header.begin(), header.end(), ',');
  std::string row;
  while (std::getline(input, row)) {
    ASSERT_EQ(std::count(row.begin(), row.end(), ','), commas);
  }
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
