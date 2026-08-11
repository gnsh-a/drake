#include "drake/tools/voxel_sdf_experiments/common/reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>

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

GTEST_TEST(ReferenceTest, AnalyticPlaneClosedFormGate) {
  RunClosedFormGate<AnalyticPlane>();
}

GTEST_TEST(ReferenceTest, AnalyticParaboloidClosedFormGate) {
  RunClosedFormGate<AnalyticParaboloid>();
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
