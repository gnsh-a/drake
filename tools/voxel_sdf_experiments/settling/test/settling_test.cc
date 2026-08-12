#include "drake/tools/voxel_sdf_experiments/settling/settling.h"

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

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
