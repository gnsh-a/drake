#include "drake/multibody/plant/tamsi_driver.h"

#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "drake/multibody/plant/test_utilities/rigid_body_on_compliant_ground.h"

constexpr double kEps = std::numeric_limits<double>::epsilon();

namespace drake {
namespace multibody {
namespace internal {

// This test verifies contact results in the equilibrium configuration.
TEST_P(RigidBodyOnCompliantGround, VerifyEquilibriumConfiguration) {
  const ContactTestConfig& config = GetParam();
  EXPECT_EQ(plant_->num_velocities(), 2);
  contact_solvers::internal::ContactSolverResults<double> results;
  tamsi_driver_->CalcContactSolverResults(*plant_context_, &results);

  EXPECT_EQ(results.v_next.size(), plant_->num_velocities());
  const int num_contacts = config.point_contact ? 1 : kNumberOfTriangles_;
  EXPECT_EQ(results.fn.size(), num_contacts);

  const double normal_force_expected = CalcBodyWeight();
  const double normal_force = results.fn.sum();
  EXPECT_NEAR(normal_force, normal_force_expected, kEps);
}

TEST_P(RigidBodyOnCompliantGround, ReportsTamsiStatistics) {
  contact_solvers::internal::ContactSolverResults<double> results;
  tamsi_driver_->CalcContactSolverResults(*plant_context_, &results);

  EXPECT_FALSE(results.sap_statistics.has_value());
  ASSERT_TRUE(results.tamsi_statistics.has_value());
  const contact_solvers::internal::TamsiStatistics& stats =
      *results.tamsi_statistics;
  EXPECT_EQ(stats.result, TamsiSolverResult::kSuccess);
  EXPECT_EQ(stats.accepted_num_substeps, 1);
  EXPECT_EQ(stats.num_substep_attempts, 1);
  EXPECT_EQ(stats.num_solve_calls, 1);
  EXPECT_GT(stats.total_iterations, 0);
  EXPECT_GE(stats.max_iterations_per_solve, 1);
  EXPECT_LE(stats.max_iterations_per_solve, stats.total_iterations);
  EXPECT_GE(stats.final_vt_residual, 0.0);
}

TEST_P(RigidBodyOnCompliantGround, PlantEvaluatesTamsiStatistics) {
  EXPECT_FALSE(
      plant_->EvalTamsiSolverStatistics(*plant_context_).has_value());

  Simulate(1);

  const std::optional<contact_solvers::internal::TamsiStatistics> stats =
      plant_->EvalTamsiSolverStatistics(*plant_context_);
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->result, TamsiSolverResult::kSuccess);
  EXPECT_EQ(stats->accepted_num_substeps, 1);
  EXPECT_EQ(stats->num_substep_attempts, 1);
  EXPECT_EQ(stats->num_solve_calls, 1);
  EXPECT_GT(stats->total_iterations, 0);
}

// Setup test cases using point and hydroelastic contact.
std::vector<ContactTestConfig> MakeTestCases() {
  return std::vector<ContactTestConfig>{
      {.description = "HydroelasticContact", .point_contact = false},
      {.description = "PointContact", .point_contact = true},
  };
}

INSTANTIATE_TEST_SUITE_P(TamsiDriverTests, RigidBodyOnCompliantGround,
                         testing::ValuesIn(MakeTestCases()),
                         testing::PrintToStringParamName());

}  // namespace internal
}  // namespace multibody
}  // namespace drake
