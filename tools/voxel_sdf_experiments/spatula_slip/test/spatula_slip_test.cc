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

/* Fragmentation is measured per representation and per finger, and the values
 asserted here are the measured behaviour rather than an expectation: affine
 voxels emit independent per-cell faces on this curved frictional contact and
 shatter into dozens of components, while both conforming meshers give one.
 This is the scene section 5 named as untested, so the count existing at all is
 the point; asserting the direction keeps a regression from silently removing
 the finding. */
GTEST_TEST(SpatulaSlipTest, FragmentationIsMeasuredPerRepresentation) {
  for (const Representation representation :
       {Representation::kTet, Representation::kMarchingCubes,
        Representation::kPlaneClip}) {
    SpatulaSlipConfig config;
    config.representation = representation;
    config.duration = 0.4;
    const SpatulaSlipResult result = RunSpatulaSlip(config);
    const std::string label(to_string(representation));
    ASSERT_GT(result.max_num_components, 0) << label;
    if (representation == Representation::kPlaneClip) {
      EXPECT_GT(result.max_num_components, 5) << label;
      EXPECT_LT(result.min_largest_component_area_fraction, 0.5) << label;
    } else {
      EXPECT_EQ(result.max_num_components, 1) << label;
      EXPECT_EQ(result.min_largest_component_area_fraction, 1.0) << label;
    }
  }
}

/* Solver effort is recorded, and unlike the settling and disk scenes it has
 room to vary here: this is the only frictional stick-slip scene in the study.
 The assertion is only that the statistics arrive and are sane, because whether
 they differ between representations is the result, not a precondition. */
GTEST_TEST(SpatulaSlipTest, SapEffortIsRecorded) {
  SpatulaSlipConfig config;
  config.duration = 0.4;
  const SpatulaSlipResult result = RunSpatulaSlip(config);
  EXPECT_GT(result.max_sap_iters, 0);
  EXPECT_EQ(result.sap_nonconverged_steps, 0);
  for (const SpatulaSlipRow& row : result.rows) {
    EXPECT_GE(row.sap_iters, 0);
    EXPECT_GE(row.sap_line_search_iters, 0);
  }
}

/* The task outcome. A run this short cannot drop the spatula, so what is
 checked is that holding is reported as holding, that no loss time is invented
 for it, and that slip is a small nonnegative number rather than the raw
 displacement -- the two differ once the spatula rotates. */
GTEST_TEST(SpatulaSlipTest, HoldingRunReportsNoDrop) {
  SpatulaSlipConfig config;
  config.duration = 0.4;
  const SpatulaSlipResult result = RunSpatulaSlip(config);
  EXPECT_TRUE(result.held);
  EXPECT_TRUE(std::isnan(result.first_contact_loss_time));
  EXPECT_GE(result.handle_axis_slip, 0.0);
  EXPECT_LT(result.handle_axis_slip, 0.05);
}

/* A grip far too weak to hold the spatula must be reported as a drop with the
 time it happened, and the slip must stop accumulating at that point rather
 than following the fall. Without this the analysis cannot tell a bad
 trajectory from a failed task, which is how a 47 m "position error" ended up
 in a results table. */
GTEST_TEST(SpatulaSlipTest, DroppedRunReportsWhenAndStopsMeasuringSlip) {
  SpatulaSlipConfig config;
  config.representation = Representation::kTet;
  config.gripper_force = 0.0;
  config.amplitude = 0.0;
  config.duration = 4.0;
  const SpatulaSlipResult result = RunSpatulaSlip(config);
  ASSERT_FALSE(result.held);
  EXPECT_TRUE(std::isfinite(result.first_contact_loss_time));
  EXPECT_GT(result.first_contact_loss_time, 0.0);
  EXPECT_LT(result.first_contact_loss_time, config.duration);
  // Free fall over the remaining seconds covers metres; slip must not.
  EXPECT_LT(result.handle_axis_slip, 0.5);
  EXPECT_GT(result.max_position_displacement, result.handle_axis_slip);
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
