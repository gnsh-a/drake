#include "drake/tools/voxel_sdf_experiments/spatula_slip/spatula_slip.h"

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

/* This is intentionally a cheap smoke tier, not a trajectory-accuracy gate.
 It proves that the exact dynamic scene runs with all three representation /
 surface-type pairings and produces finite, physically sane contact data. */
GTEST_TEST(SpatulaSlipTest, AllRepresentationsProduceSaneTrajectories) {
  const std::array<Representation, 3> representations{
      Representation::kTet, Representation::kPlaneClip,
      Representation::kMarchingCubes};
  for (const Representation representation : representations) {
    SpatulaSlipConfig config;
    config.representation = representation;
    config.resolution_scale = 1.0;
    config.duration = 0.12;
    config.output = ScratchPath(std::string(to_string(representation)) +
                                "_spatula_slip.csv");
    const SpatulaSlipResult result = RunSpatulaSlip(config);
    const std::string label(to_string(representation));

    EXPECT_EQ(result.rows.size(), 4) << label;
    EXPECT_EQ(result.ellipsoid_base_resolution, 0.04) << label;
    EXPECT_EQ(result.cylinder_base_resolution, 0.005) << label;
    EXPECT_GT(result.contact_samples, 0) << label;
    EXPECT_GT(result.two_finger_contact_samples, 0) << label;
    EXPECT_GT(result.max_contact_force, 0.0) << label;
    EXPECT_LT(result.max_contact_force, 1000.0) << label;
    EXPECT_GT(result.max_contact_area, 0.0) << label;
    EXPECT_LT(result.max_contact_area, 0.1) << label;
    EXPECT_GE(result.max_handle_axis_swing, 0.0) << label;
    EXPECT_LT(result.max_handle_axis_swing, M_PI) << label;
    if (representation == Representation::kMarchingCubes) {
      EXPECT_GT(result.triangle_surface_samples, 0) << label;
    } else {
      EXPECT_EQ(result.triangle_surface_samples, 0) << label;
    }
    for (const SpatulaSlipRow& row : result.rows) {
      EXPECT_TRUE(row.position_WB.allFinite()) << label;
      EXPECT_NEAR(row.quaternion_WB.norm(), 1.0, 1.0e-12) << label;
      EXPECT_NEAR(row.handle_axis_W.norm(), 1.0, 1.0e-12) << label;
      EXPECT_GT(row.position_WB.norm(), 0.1) << label;
      EXPECT_LT(row.position_WB.norm(), 1.0) << label;
      EXPECT_LT(std::abs(row.left_finger_position), 0.1) << label;
      EXPECT_LT(std::abs(row.right_finger_position), 0.1) << label;
      EXPECT_EQ(row.point_contacts, 0) << label;
      EXPECT_GE(row.spatula_contacts, 0) << label;
      EXPECT_LE(row.spatula_contacts, 2) << label;
      EXPECT_TRUE(row.left_contact_force_W.allFinite()) << label;
      EXPECT_TRUE(std::isfinite(row.left_handle_axis_torque)) << label;
      EXPECT_TRUE(std::isfinite(row.left_contact_area)) << label;
      EXPECT_TRUE(row.right_contact_force_W.allFinite()) << label;
      EXPECT_TRUE(std::isfinite(row.right_handle_axis_torque)) << label;
      EXPECT_TRUE(std::isfinite(row.right_contact_area)) << label;
    }

    std::ifstream input(config.output);
    ASSERT_TRUE(input) << config.output;
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
    EXPECT_EQ(line, SpatulaSlipCsvHeader());
    int csv_rows = 0;
    while (std::getline(input, line)) ++csv_rows;
    EXPECT_EQ(csv_rows, result.rows.size()) << label;
  }
}

GTEST_TEST(SpatulaSlipTest, RejectsUnalignedSampling) {
  SpatulaSlipConfig config;
  config.sample_period = 0.03;
  EXPECT_THROW(RunSpatulaSlip(config), std::logic_error);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
