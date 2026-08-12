#include "drake/tools/voxel_sdf_experiments/common/reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double AdaptiveSimpson(const std::function<double(double)>& function,
                       double lower, double upper, double lower_value,
                       double middle_value, double upper_value,
                       double whole_interval, double tolerance, int depth) {
  const double middle = 0.5 * (lower + upper);
  const double left_middle = 0.5 * (lower + middle);
  const double right_middle = 0.5 * (middle + upper);
  const double left_middle_value = function(left_middle);
  const double right_middle_value = function(right_middle);
  const double left_interval =
      (middle - lower) *
      (lower_value + 4.0 * left_middle_value + middle_value) / 6.0;
  const double right_interval =
      (upper - middle) *
      (middle_value + 4.0 * right_middle_value + upper_value) / 6.0;
  const double split_interval = left_interval + right_interval;
  if (depth == 0 ||
      std::abs(split_interval - whole_interval) <= 15.0 * tolerance) {
    return split_interval + (split_interval - whole_interval) / 15.0;
  }
  return AdaptiveSimpson(function, lower, middle, lower_value,
                         left_middle_value, middle_value, left_interval,
                         tolerance / 2.0, depth - 1) +
         AdaptiveSimpson(function, middle, upper, middle_value,
                         right_middle_value, upper_value, right_interval,
                         tolerance / 2.0, depth - 1);
}

double IntegratePressureOverProjectedPatch(const Reference& reference) {
  const double radius = reference.patch_radius();
  const double centroid_height = reference.centroid().z();
  const auto pressure_in_squared_radius = [&reference, radius,
                                           centroid_height](double value) {
    const double radial_distance = radius * std::sqrt(value);
    return reference.pressure_at(
        Eigen::Vector3d(radial_distance, 0.0, centroid_height));
  };
  const double lower_value = pressure_in_squared_radius(0.0);
  const double middle_value = pressure_in_squared_radius(0.5);
  const double upper_value = pressure_in_squared_radius(1.0);
  const double whole_interval =
      (lower_value + 4.0 * middle_value + upper_value) / 6.0;
  const double pressure_integral =
      AdaptiveSimpson(pressure_in_squared_radius, 0.0, 1.0, lower_value,
                      middle_value, upper_value, whole_interval,
                      2e-13 * std::max(1.0, reference.peak_pressure()), 24);
  return kPi * radius * radius * pressure_integral;
}

template <typename ReferenceType>
void RunClosedFormGate() {
  constexpr double kRadius = 0.1;
  const std::array<double, 4> penetrations{0.002, 0.01, 0.0199, 0.08};
  const std::array<std::array<double, 2>, 4> moduli{{
      {1.0e5, 1.0e5},
      {1.0e8, 1.0e8},
      {2.5e7, 4.0e8},
      {4.0e10, 7.0e7},
  }};
  for (const double penetration : penetrations) {
    for (const auto& modulus : moduli) {
      const ReferenceType reference(kRadius, penetration, modulus[0],
                                    modulus[1]);
      const double integrated_force =
          IntegratePressureOverProjectedPatch(reference);
      EXPECT_NEAR(integrated_force, reference.force(),
                  2e-12 * reference.force())
          << "penetration=" << penetration << ", moduli=" << modulus[0] << ","
          << modulus[1];
    }
  }
}

template <typename ReferenceType>
void RunEquilibriumRoundTripGate() {
  constexpr double kRadius = 0.1;
  constexpr double kModulus = 1.0e5;
  const ReferenceFactory factory = [=](double penetration) {
    return std::make_unique<ReferenceType>(kRadius, penetration, kModulus,
                                           kModulus);
  };
  const std::array<double, 5> source_penetrations{0.0005, 0.002, 0.01, 0.0199,
                                                  0.08};
  double previous_penetration = 0.0;
  double previous_mass = 0.0;
  for (const double source_penetration : source_penetrations) {
    const double target_force = ForceAtPenetration(factory, source_penetration);
    const double mass = target_force / 9.81;
    const double penetration =
        EquilibriumPenetration(factory, mass * 9.81, kRadius);
    EXPECT_NEAR(ForceAtPenetration(factory, penetration), target_force,
                1e-12 * target_force)
        << "mass=" << mass;
    EXPECT_GT(penetration, previous_penetration);
    EXPECT_GT(mass, previous_mass);
    previous_penetration = penetration;
    previous_mass = mass;
  }
}

GTEST_TEST(ReferenceTest, AnalyticPlaneClosedFormGate) {
  RunClosedFormGate<AnalyticPlane>();
}

GTEST_TEST(ReferenceTest, AnalyticParaboloidClosedFormGate) {
  RunClosedFormGate<AnalyticParaboloid>();
}

GTEST_TEST(ReferenceTest, EquilibriumPenetrationRoundTripAndMonotonicity) {
  RunEquilibriumRoundTripGate<AnalyticPlane>();
  RunEquilibriumRoundTripGate<AnalyticParaboloid>();
}

/* The equal-sphere stiffness has a closed form, pi E x delta / (2 R) with
 x = R - delta / 2, which pins the central difference exactly. The paraboloid
 has no such simple form here, so it is checked for self-consistency against a
 much wider difference step, and for being stiffer than the plane at equal
 penetration. The sphere-box scene carries roughly 1.85x the load and stiffness
 of the equal-sphere scene at this operating point, which is precisely why one
 shared analytic expression cannot serve both. */
GTEST_TEST(ReferenceTest, StiffnessMatchesClosedFormAndIsSceneSpecific) {
  constexpr double kRadius = 0.1;
  constexpr double kModulus = 1.0e5;
  const ReferenceFactory plane = [=](double penetration) {
    return std::make_unique<AnalyticPlane>(kRadius, penetration, kModulus,
                                           kModulus);
  };
  const ReferenceFactory paraboloid = [=](double penetration) {
    return std::make_unique<AnalyticParaboloid>(kRadius, penetration, kModulus,
                                                kModulus);
  };
  for (const double penetration : {0.002, 0.0199, 0.08 / 3.0, 0.08}) {
    const double x = kRadius - penetration / 2.0;
    const double expected = kPi * kModulus * x * penetration / (2.0 * kRadius);
    const double plane_stiffness = StiffnessAtPenetration(plane, penetration);
    EXPECT_NEAR(plane_stiffness, expected, 1e-8 * expected)
        << "penetration=" << penetration;

    const double wide = 1.0e-3 * penetration;
    const double reference_slope =
        (ForceAtPenetration(paraboloid, penetration + wide) -
         ForceAtPenetration(paraboloid, penetration - wide)) /
        (2.0 * wide);
    const double paraboloid_stiffness =
        StiffnessAtPenetration(paraboloid, penetration);
    EXPECT_NEAR(paraboloid_stiffness, reference_slope, 1e-5 * reference_slope)
        << "penetration=" << penetration;
    EXPECT_GT(paraboloid_stiffness, plane_stiffness)
        << "penetration=" << penetration;
  }
}

GTEST_TEST(ReferenceTest, EqualModulusSurfaceShapes) {
  constexpr double kRadius = 0.1;
  constexpr double kPenetration = 0.02;
  constexpr double kModulus = 1.0e8;
  const AnalyticPlane plane(kRadius, kPenetration, kModulus, kModulus);
  const double plane_height = (2.0 * kRadius - kPenetration) / 2.0;
  EXPECT_NEAR(plane.distance_to_surface(Eigen::Vector3d(
                  0.5 * plane.patch_radius(), 0.0, plane_height + 0.001)),
              0.001, 1e-14);

  const AnalyticParaboloid paraboloid(kRadius, kPenetration, kModulus,
                                      kModulus);
  const double radial_distance = 0.5 * paraboloid.patch_radius();
  const double sphere_center_height = 2.0 * kRadius - kPenetration;
  const double paraboloid_height =
      (radial_distance * radial_distance +
       sphere_center_height * sphere_center_height) /
      (2.0 * sphere_center_height);
  EXPECT_NEAR(paraboloid.distance_to_surface(
                  Eigen::Vector3d(radial_distance, 0.0, paraboloid_height)),
              0.0, 1e-12);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
