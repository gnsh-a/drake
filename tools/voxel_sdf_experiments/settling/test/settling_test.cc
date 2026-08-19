#include "drake/tools/voxel_sdf_experiments/settling/settling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/drake_assert.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

std::filesystem::path ScratchPath(const std::string& name) {
  const char* const directory = std::getenv("TEST_TMPDIR");
  return std::filesystem::path(directory != nullptr ? directory : ".") / name;
}

GTEST_TEST(SettlingTest, AutoDerivationUsesMassAndSceneReference) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  for (const SettlingScene scene :
       {SettlingScene::kSphereSphere, SettlingScene::kSphereBox}) {
    SettlingConfig config;
    config.scene = scene;
    config.mass = DefaultSettlingMass(scene, config.hydroelastic_modulus);
    const SettlingDerived derived = CalcSettlingDerived(config);
    EXPECT_NEAR(derived.analytic_equilibrium_penetration, 0.08 / 3.0, 2e-15);

    /* The equal-sphere closed form pins sphere_sphere exactly. It must NOT be
     applied to sphere_box: the paraboloid is measurably stiffer at the same
     penetration, and using one expression for both understated sphere_box's
     stiffness by 1.85x, which inflated its auto duration by 36%. */
    const double x = 0.1 - derived.analytic_equilibrium_penetration / 2.0;
    const double equal_sphere_stiffness =
        kPi * config.hydroelastic_modulus * x *
        derived.analytic_equilibrium_penetration / 0.2;
    if (scene == SettlingScene::kSphereSphere) {
      EXPECT_NEAR(derived.contact_stiffness, equal_sphere_stiffness,
                  1e-8 * equal_sphere_stiffness);
    } else {
      EXPECT_GT(derived.contact_stiffness, 1.8 * equal_sphere_stiffness);
      EXPECT_LT(derived.contact_stiffness, 1.9 * equal_sphere_stiffness);
    }

    const double expected_period =
        2.0 * kPi * std::sqrt(config.mass / derived.contact_stiffness);
    EXPECT_NEAR(derived.natural_period, expected_period,
                1e-14 * expected_period);
    EXPECT_NEAR(derived.duration, 15.0 * expected_period,
                1e-14 * derived.duration);
    EXPECT_NEAR(derived.settling_window, 1.25 * expected_period,
                1e-14 * derived.settling_window);
  }
}

/* Configures a deliberately coarse, short run. The smoke coverage cares that
 every representation acquires contact and emits a complete row, not that the
 penetration has converged. */
SettlingConfig MakeSmokeConfig(SettlingScene scene,
                               Representation representation) {
  SettlingConfig config;
  config.scene = scene;
  config.representation = representation;
  config.mass = DefaultSettlingMass(scene, config.hydroelastic_modulus);
  config.voxel_width = 0.02;
  config.tet_resolution_hint = 0.02;
  config.time_step = 1.0e-3;
  config.duration = 0.6;
  config.settling_window = 0.1;
  return config;
}

GTEST_TEST(SettlingTest, AllRepresentationsAcquireContactAndEmitCompleteRow) {
  const std::array<Representation, 3> representations{
      Representation::kTet, Representation::kPlaneClip,
      Representation::kMarchingCubes};
  /* Runs on sphere_box, whose contact surface is flat and therefore laterally
   neutral, so every representation stays in contact here. See the
   SymmetryIsPreserved test below for why sphere_sphere cannot serve this role
   for marching cubes. */
  for (const Representation representation : representations) {
    SettlingConfig config =
        MakeSmokeConfig(SettlingScene::kSphereBox, representation);
    config.output =
        ScratchPath(std::string(to_string(representation)) + "_settling.csv");
    const SettlingResult result = RunSettling(config);

    /* Free fall over the initial gap is a hard lower bound, but the upper bound
     is representation-dependent: a surface is reported only once the kernel can
     extract one, and marching cubes needs its dual-grid nodes to straddle the
     zero level set, so it acquires contact several milliseconds late at this
     coarse grid. The earlier study saw the same split on sphere_box -- 14.3 ms
     for the affine kernel against 20.3 ms for marching cubes. */
    const double free_fall_time = std::sqrt(2.0 * 0.001 / 9.81);
    ASSERT_TRUE(result.first_contact_time.has_value());
    EXPECT_GE(*result.first_contact_time, free_fall_time - config.time_step);
    EXPECT_LE(*result.first_contact_time, free_fall_time + 0.010);
    EXPECT_NEAR(result.mean_support_force, result.derived.weight,
                0.5 * result.derived.weight);
    EXPECT_TRUE(std::isfinite(result.equilibrium_penetration));
    EXPECT_GT(result.mean_faces, 0.0);
    EXPECT_GT(result.mean_contact_area, 0.0);

    std::ifstream input(config.output);
    ASSERT_TRUE(input);
    std::string header;
    std::string row;
    ASSERT_TRUE(static_cast<bool>(std::getline(input, header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(input, row)));
    EXPECT_EQ(header, SettlingCsvHeader());
    EXPECT_EQ(std::count(header.begin(), header.end(), ','),
              std::count(row.begin(), row.end(), ','));
    EXPECT_FALSE(row.empty());
    std::string extra;
    EXPECT_FALSE(static_cast<bool>(std::getline(input, extra)));
  }
}

/* Both scenes are axisymmetric about the vertical and frictionless, so lateral
 drift and spin are pure discretization error, and this test pins how much of it
 each combination produces at a coarse grid. The bounds are not uniform, because
 the measured behaviour is not: each one records a finding.

 Symmetry survives when the discretization shares the scene's symmetry. On
 sphere_sphere the two tetrahedral meshes are mirror images, so their errors
 cancel and tet holds machine precision. On sphere_box that cancellation is
 gone -- an irregularly meshed sphere against a box has no such pairing -- and
 tet drifts by of order 1e-4. The affine kernel holds machine precision in both,
 because a cube lattice aligned with an axis-aligned box is symmetric under the
 scene's own symmetry group while the primitive SDF is exactly axisymmetric.

 Marching cubes is the loosest: of order a millimetre at this grid, matching the
 earlier study's h = 10 mm sphere_box numbers (2.6e-4 m, 1.1e-2 rad/s). That
 drift converges away at roughly 15x per halving of h, so it is a coarse-grid
 artifact rather than a persistent defect.

 Scene stability decides whether any of that is survivable. Sphere-box contact
 is flat, hence laterally neutral: drift neither grows nor is corrected.
 Sphere-on-sphere apex contact is laterally *unstable*, so drift tilts the
 normal and gravity drives it further off. Marching cubes therefore loses
 contact entirely on sphere_sphere at coarse resolution, which is why that
 combination is asserted to fail rather than given a bound. */
GTEST_TEST(SettlingTest, SymmetryIsPreserved) {
  struct Expectation {
    SettlingScene scene;
    Representation representation;
    double lateral_bound;
    double angular_bound;
  };
  constexpr double kMachine = 1.0e-12;
  const std::array<Expectation, 5> expectations{{
      {SettlingScene::kSphereSphere, Representation::kTet, kMachine, kMachine},
      {SettlingScene::kSphereSphere, Representation::kPlaneClip, kMachine,
       kMachine},
      {SettlingScene::kSphereBox, Representation::kTet, 5.0e-4, 5.0e-2},
      {SettlingScene::kSphereBox, Representation::kPlaneClip, kMachine,
       kMachine},
      {SettlingScene::kSphereBox, Representation::kMarchingCubes, 5.0e-3,
       2.0e-1},
  }};
  for (const Expectation& expectation : expectations) {
    const SettlingResult result = RunSettling(
        MakeSmokeConfig(expectation.scene, expectation.representation));
    const std::string label =
        std::string(to_string(expectation.scene)) + ' ' +
        std::string(to_string(expectation.representation));
    EXPECT_LT(result.max_lateral_offset, expectation.lateral_bound) << label;
    EXPECT_LT(result.max_angular_speed, expectation.angular_bound) << label;
  }

  /* Marching cubes on sphere_sphere is expected to lose contact outright. If
   this ever starts holding contact, the exclusion above should be revisited. */
  const SettlingResult unstable = RunSettling(MakeSmokeConfig(
      SettlingScene::kSphereSphere, Representation::kMarchingCubes));
  EXPECT_GT(unstable.max_lateral_offset, 1.0e-6);
}

/* Splits one CSV line on commas. No field here is ever quoted or contains a
 comma, so this does not need to be a real CSV parser. */
std::vector<std::string> SplitFields(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

/* Reads a one-row summary CSV into a column-name to value map. */
std::map<std::string, std::string> ReadSummaryRow(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  DRAKE_DEMAND(static_cast<bool>(input));
  std::string header_line;
  std::string row_line;
  DRAKE_DEMAND(static_cast<bool>(std::getline(input, header_line)));
  DRAKE_DEMAND(static_cast<bool>(std::getline(input, row_line)));
  const std::vector<std::string> names = SplitFields(header_line);
  const std::vector<std::string> values = SplitFields(row_line);
  DRAKE_DEMAND(names.size() == values.size());
  std::map<std::string, std::string> row;
  for (size_t index = 0; index < names.size(); ++index) {
    row[names[index]] = values[index];
  }
  return row;
}

/* The row is written field by field against a header written as one string
 literal, so the two can drift apart without changing the field count. Counting
 commas -- which the completeness test above does -- cannot see a swap between
 two columns of the same type, and the columns most at risk of that are the
 adjacent same-typed pairs added together: mean_faces/faces_span,
 mean_contact_area/contact_area_span, and the four SAP columns. Values are
 emitted at max_digits10, so they round-trip exactly and can be compared with
 EXPECT_EQ rather than a tolerance. */
GTEST_TEST(SettlingTest, SummaryColumnsCarryTheFieldsTheyName) {
  SettlingConfig config =
      MakeSmokeConfig(SettlingScene::kSphereBox, Representation::kPlaneClip);
  config.trajectory_stride = 4;
  config.output = ScratchPath("named_columns_settling.csv");
  const SettlingResult result = RunSettling(config);
  const std::map<std::string, std::string> row = ReadSummaryRow(config.output);

  auto value = [&row](const std::string& name) {
    const auto iterator = row.find(name);
    DRAKE_DEMAND(iterator != row.end());
    return std::stod(iterator->second);
  };
  EXPECT_EQ(value("trajectory_stride"), config.trajectory_stride);
  EXPECT_EQ(value("mean_faces"), result.mean_faces);
  EXPECT_EQ(value("faces_span"), result.faces_span);
  EXPECT_EQ(value("mean_contact_area_m2"), result.mean_contact_area);
  EXPECT_EQ(value("contact_area_span_m2"), result.contact_area_span);
  EXPECT_EQ(value("largest_component_area_fraction"),
            result.largest_component_area_fraction);
  EXPECT_EQ(value("mean_num_components"), result.mean_num_components);
  EXPECT_EQ(value("mean_sap_iters"), result.mean_sap_iters);
  EXPECT_EQ(value("max_sap_iters"), result.max_sap_iters);
  EXPECT_EQ(value("sap_nonconverged_steps"), result.sap_nonconverged_steps);
  EXPECT_EQ(value("max_sap_momentum_residual"),
            result.max_sap_momentum_residual);
  EXPECT_EQ(value("max_penetration_m"), result.max_penetration);
}

/* The SAP columns reach the row through a chain -- SapSolver populates the
 statistics, SapDriver copies them into the contact-solver results, the plant
 caches those in its step memory, and MultibodyPlant::EvalSapSolverStatistics
 reads them back out under the plant's output-port mode. Any break in that
 chain degrades silently to zeros in a converged column, which is
 indistinguishable from a well-behaved solve to anything reading the CSV. This
 asserts the statistics are live rather than defaulted. */
GTEST_TEST(SettlingTest, SapStatisticsAreLive) {
  const SettlingResult result = RunSettling(
      MakeSmokeConfig(SettlingScene::kSphereBox, Representation::kPlaneClip));
  EXPECT_GT(result.mean_sap_iters, 0.0);
  EXPECT_GE(result.max_sap_iters, result.mean_sap_iters);
  /* A settled, converging contact must not exhaust SAP's iteration budget. */
  EXPECT_EQ(result.sap_nonconverged_steps, 0.0);
  EXPECT_TRUE(std::isfinite(result.max_sap_momentum_residual));
}

/* Striding decides how much of a long run reaches disk, and getting it wrong
 is invisible in the file itself -- a strided trajectory looks like a
 well-formed trajectory at a coarser time step. Comparing against an unstrided
 run of the same configuration pins the count exactly. */
GTEST_TEST(SettlingTest, TrajectoryStrideKeepsEveryNthSample) {
  auto row_count = [](const std::filesystem::path& path) {
    std::ifstream input(path);
    DRAKE_DEMAND(static_cast<bool>(input));
    std::string line;
    int64_t lines = 0;
    while (std::getline(input, line)) ++lines;
    return lines - 1;  // Drop the header.
  };

  SettlingConfig dense =
      MakeSmokeConfig(SettlingScene::kSphereBox, Representation::kPlaneClip);
  dense.output = ScratchPath("dense_settling.csv");
  dense.trajectory = ScratchPath("dense_trajectory.csv");
  RunSettling(dense);

  SettlingConfig strided = dense;
  strided.trajectory_stride = 3;
  strided.output = ScratchPath("strided_settling.csv");
  strided.trajectory = ScratchPath("strided_trajectory.csv");
  RunSettling(strided);

  const int64_t dense_rows = row_count(dense.trajectory);
  EXPECT_GT(dense_rows, 100);
  EXPECT_EQ(
      row_count(strided.trajectory),
      (dense_rows + strided.trajectory_stride - 1) / strided.trajectory_stride);

  /* Every trajectory row must carry as many fields as the header names, and
   the kept samples must be spaced by the stride. */
  std::ifstream input(strided.trajectory);
  std::string header_line;
  std::string first_line;
  std::string second_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, header_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(input, first_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(input, second_line)));
  const std::vector<std::string> names = SplitFields(header_line);
  EXPECT_EQ(SplitFields(first_line).size(), names.size());
  EXPECT_EQ(names.front(), "time_s");
  EXPECT_NEAR(std::stod(SplitFields(second_line).front()) -
                  std::stod(SplitFields(first_line).front()),
              strided.trajectory_stride * strided.time_step,
              0.1 * strided.time_step);
}

/* The duration and window are configured in natural periods, and the period is
 itself derived, so a config field that is silently ignored still produces a
 plausible run. The auto-derivation test above pins only the defaults, which it
 would keep doing if these fields were dropped entirely. */
GTEST_TEST(SettlingTest, PeriodCountsAreHonored) {
  SettlingConfig config;
  config.mass = DefaultSettlingMass(config.scene, config.hydroelastic_modulus);
  config.duration_periods = 4.0;
  config.settling_window_periods = 0.5;
  const SettlingDerived derived = CalcSettlingDerived(config);
  EXPECT_NEAR(derived.duration, 4.0 * derived.natural_period,
              1e-14 * derived.duration);
  EXPECT_NEAR(derived.settling_window, 0.5 * derived.natural_period,
              1e-14 * derived.settling_window);

  /* An explicit absolute time still wins over the period count. */
  config.duration = 1.0;
  config.settling_window = 0.25;
  const SettlingDerived overridden = CalcSettlingDerived(config);
  EXPECT_EQ(overridden.duration, 1.0);
  EXPECT_EQ(overridden.settling_window, 0.25);
}

/* The load is what places the equal-pressure plane relative to the voxel
 lattice, so choosing the target penetration is the only handle the settling
 scene offers on cell phase. Two facts are worth pinning because the
 on-boundary study depends on both.

 First, 0.02 m puts the affine kernel exactly on a cell boundary at every
 dyadic rung: phase is remainder(delta / 2h, 1), and 0.02 / 2h is 1, 2, 4 as h
 halves from 10 mm. Second, no target does that for marching cubes at every
 rung, because its degeneracy sits at delta = (2i + 1) h -- the half-cell shift
 below -- and halving h turns an odd multiple into an even one. At 0.02 m
 marching cubes is instead maximally *off* boundary at all three rungs, which
 is why the boundary axis needs a per-rung target rather than one value. */
GTEST_TEST(SettlingTest, TargetPenetrationControlsCellPhase) {
  constexpr double kOnAffineBoundary = 0.02;
  for (const double voxel_width : {0.01, 0.005, 0.0025}) {
    SettlingConfig config;
    config.voxel_width = voxel_width;
    config.mass = DefaultSettlingMass(config.scene, config.hydroelastic_modulus,
                                      kOnAffineBoundary);

    config.representation = Representation::kPlaneClip;
    SettlingDerived derived = CalcSettlingDerived(config);
    EXPECT_NEAR(derived.analytic_equilibrium_penetration, kOnAffineBoundary,
                2e-15);
    EXPECT_NEAR(derived.contact_plane_voxel_phase, 0.0, 1e-12)
        << "h = " << voxel_width;

    config.representation = Representation::kMarchingCubes;
    derived = CalcSettlingDerived(config);
    EXPECT_NEAR(std::abs(derived.contact_plane_voxel_phase), 0.5, 1e-12)
        << "h = " << voxel_width;
  }

  /* The default target is the complement of the above: 1/3 of a cell off an
   affine boundary and 1/6 off a marching-cubes one, invariant under halving
   h, which is the whole reason for the 0.08 / 3 value. */
  for (const double voxel_width : {0.01, 0.005, 0.0025}) {
    SettlingConfig config;
    config.voxel_width = voxel_width;
    config.mass =
        DefaultSettlingMass(config.scene, config.hydroelastic_modulus);

    config.representation = Representation::kPlaneClip;
    EXPECT_NEAR(std::abs(CalcSettlingDerived(config).contact_plane_voxel_phase),
                1.0 / 3.0, 1e-12)
        << "h = " << voxel_width;

    config.representation = Representation::kMarchingCubes;
    EXPECT_NEAR(std::abs(CalcSettlingDerived(config).contact_plane_voxel_phase),
                1.0 / 6.0, 1e-12)
        << "h = " << voxel_width;
  }

  /* Tet has no lattice, so it has no phase to report. */
  SettlingConfig tet_config;
  tet_config.representation = Representation::kTet;
  tet_config.mass =
      DefaultSettlingMass(tet_config.scene, tet_config.hydroelastic_modulus);
  EXPECT_TRUE(
      std::isnan(CalcSettlingDerived(tet_config).contact_plane_voxel_phase));
}

/* Component counting shares its labelling pass with the largest-component area
 fraction, so the two are checked together: a count that silently collapsed to
 one would leave the fraction reading a perfect 1.0, and neither number alone
 would look wrong.

 Sphere-on-box contact is a single flat disc, and tet resolves it as one
 connected patch holding all of the area. The affine kernel does not, even
 though it converges on penetration at second order: at this coarse grid the
 patch spans about five cells and comes back as tens of disjoint pieces with
 the largest holding a fifth of the area. That split is the fragmentation the
 settling study reports, so it is asserted here rather than treated as a
 defect -- what would be a regression is the affine kernel suddenly reporting
 one clean component, which would mean the metric had stopped measuring. */
GTEST_TEST(SettlingTest, FragmentationIsMeasuredPerRepresentation) {
  const SettlingResult tet = RunSettling(
      MakeSmokeConfig(SettlingScene::kSphereBox, Representation::kTet));
  EXPECT_GE(tet.mean_num_components, 1.0);
  EXPECT_LT(tet.mean_num_components, 1.5);
  EXPECT_GT(tet.largest_component_area_fraction, 0.99);

  const SettlingResult affine = RunSettling(
      MakeSmokeConfig(SettlingScene::kSphereBox, Representation::kPlaneClip));
  EXPECT_GT(affine.mean_num_components, 5.0);
  EXPECT_LT(affine.largest_component_area_fraction, 0.5);
  /* The two metrics are computed from one labelling, so they must agree: many
   components cannot coexist with one component holding nearly all the area. */
  EXPECT_LT(affine.largest_component_area_fraction,
            1.0 - 1.0 / affine.mean_num_components);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
