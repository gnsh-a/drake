#include "drake/tools/voxel_sdf_experiments/common/reference.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void ThrowUnlessFinitePositive(double value, std::string_view name) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw std::logic_error(std::string(name) +
                           " must be finite and strictly positive");
  }
}

void ValidateInputs(double radius, double penetration, double modulus_a,
                    double modulus_b) {
  ThrowUnlessFinitePositive(radius, "radius");
  ThrowUnlessFinitePositive(penetration, "penetration");
  ThrowUnlessFinitePositive(modulus_a, "modulus_a");
  ThrowUnlessFinitePositive(modulus_b, "modulus_b");
  if (!(penetration < 2.0 * radius)) {
    throw std::logic_error("penetration must be smaller than the diameter");
  }
}

template <size_t N>
double EvaluatePolynomial(const std::array<double, N>& coefficients,
                          double value) {
  double result = 0.0;
  for (auto iter = coefficients.rbegin(); iter != coefficients.rend(); ++iter) {
    result = result * value + *iter;
  }
  return result;
}

template <size_t N>
double IntegratePolynomialOnUnitInterval(
    const std::array<double, N>& coefficients) {
  double result = 0.0;
  for (size_t i = 0; i < N; ++i) {
    result += coefficients[i] / static_cast<double>(i + 1);
  }
  return result;
}

template <size_t N>
double SolveMonotoneRadiusSquared(
    const std::array<double, N>& radius_squared_coefficients,
    double radius_squared) {
  const double patch_radius_squared = radius_squared_coefficients[0];
  if (radius_squared >= patch_radius_squared) return 0.0;
  if (radius_squared <= 0.0) return 1.0;
  double lower = 0.0;
  double upper = 1.0;
  for (int i = 0; i < 80; ++i) {
    const double middle = 0.5 * (lower + upper);
    if (EvaluatePolynomial(radius_squared_coefficients, middle) >
        radius_squared) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return 0.5 * (lower + upper);
}

template <typename RadiusSquared, typename Height>
double DistanceToAxisymmetricPatch(const Eigen::Vector3d& p_RQ,
                                   RadiusSquared radius_squared,
                                   Height height) {
  const double radial_distance = std::hypot(p_RQ.x(), p_RQ.y());
  const auto squared_distance = [&](double fraction) {
    const double reference_radius =
        std::sqrt(std::max(0.0, radius_squared(fraction)));
    const double radial_error = radial_distance - reference_radius;
    const double height_error = p_RQ.z() - height(fraction);
    return radial_error * radial_error + height_error * height_error;
  };

  // Golden-section search gives the Euclidean distance in the meridional
  // plane. Axisymmetry guarantees that the closest point has the same azimuth.
  constexpr double kInversePhi = 0.6180339887498948482;
  double lower = 0.0;
  double upper = 1.0;
  double left = upper - kInversePhi * (upper - lower);
  double right = lower + kInversePhi * (upper - lower);
  double left_value = squared_distance(left);
  double right_value = squared_distance(right);
  for (int i = 0; i < 96; ++i) {
    if (left_value < right_value) {
      upper = right;
      right = left;
      right_value = left_value;
      left = upper - kInversePhi * (upper - lower);
      left_value = squared_distance(left);
    } else {
      lower = left;
      left = right;
      left_value = right_value;
      right = lower + kInversePhi * (upper - lower);
      right_value = squared_distance(right);
    }
  }
  const double minimum = std::min(
      {left_value, right_value, squared_distance(0.0), squared_distance(1.0)});
  return std::sqrt(std::max(0.0, minimum));
}

template <size_t Q, size_t Z>
double SurfaceCentroidHeight(
    const std::array<double, Q>& radius_squared_coefficients,
    const std::array<double, Z>& height_coefficients) {
  constexpr int kIntervals = 4096;
  const auto sample = [&](double fraction) {
    double radius_squared = 0.0;
    double radius_squared_derivative = 0.0;
    for (size_t i = 0; i < Q; ++i) {
      radius_squared += radius_squared_coefficients[i] *
                        std::pow(fraction, static_cast<int>(i));
      if (i > 0) {
        radius_squared_derivative +=
            static_cast<double>(i) * radius_squared_coefficients[i] *
            std::pow(fraction, static_cast<int>(i - 1));
      }
    }
    double height_derivative = 0.0;
    for (size_t i = 1; i < Z; ++i) {
      height_derivative += static_cast<double>(i) * height_coefficients[i] *
                           std::pow(fraction, static_cast<int>(i - 1));
    }
    const double area_density =
        std::sqrt(radius_squared_derivative * radius_squared_derivative +
                  4.0 * std::max(0.0, radius_squared) * height_derivative *
                      height_derivative);
    return std::array<double, 2>{
        area_density,
        area_density * EvaluatePolynomial(height_coefficients, fraction)};
  };
  std::array<double, 2> integral{};
  for (int i = 0; i <= kIntervals; ++i) {
    const double fraction =
        static_cast<double>(i) / static_cast<double>(kIntervals);
    const double weight =
        i == 0 || i == kIntervals ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
    const auto value = sample(fraction);
    integral[0] += weight * value[0];
    integral[1] += weight * value[1];
  }
  return integral[1] / integral[0];
}

}  // namespace

Reference::~Reference() = default;

AnalyticPlane::AnalyticPlane(double radius, double penetration,
                             double modulus_lower, double modulus_upper)
    : radius_(radius),
      penetration_(penetration),
      modulus_lower_(modulus_lower),
      modulus_upper_(modulus_upper),
      center_distance_(2.0 * radius - penetration) {
  ValidateInputs(radius, penetration, modulus_lower, modulus_upper);
  peak_pressure_ = modulus_lower_ * modulus_upper_ * penetration_ /
                   (radius_ * (modulus_lower_ + modulus_upper_));

  const double lower_depth_at_peak =
      penetration_ * modulus_upper_ / (modulus_lower_ + modulus_upper_);
  const double upper_depth_at_peak = penetration_ - lower_depth_at_peak;
  height_coefficients_[0] = center_distance_ / 2.0;
  height_coefficients_[1] =
      -radius_ * (lower_depth_at_peak - upper_depth_at_peak) / center_distance_;
  height_coefficients_[2] = (lower_depth_at_peak * lower_depth_at_peak -
                             upper_depth_at_peak * upper_depth_at_peak) /
                            (2.0 * center_distance_);

  const std::array<double, 3> lower_radius_squared{
      radius_ * radius_, -2.0 * radius_ * lower_depth_at_peak,
      lower_depth_at_peak * lower_depth_at_peak};
  std::array<double, 5> height_squared{};
  for (size_t i = 0; i < height_coefficients_.size(); ++i) {
    for (size_t j = 0; j < height_coefficients_.size(); ++j) {
      height_squared[i + j] +=
          height_coefficients_[i] * height_coefficients_[j];
    }
  }
  for (size_t i = 0; i < radius_squared_coefficients_.size(); ++i) {
    radius_squared_coefficients_[i] =
        (i < lower_radius_squared.size() ? lower_radius_squared[i] : 0.0) -
        height_squared[i];
  }
}

double AnalyticPlane::force() const {
  return kPi * peak_pressure_ *
         IntegratePolynomialOnUnitInterval(radius_squared_coefficients_);
}

double AnalyticPlane::area() const {
  return kPi * radius_squared_coefficients_[0];
}

double AnalyticPlane::distance_to_surface(const Eigen::Vector3d& p_RQ) const {
  return DistanceToAxisymmetricPatch(
      p_RQ,
      [this](double fraction) {
        return RadiusSquaredAtPressureFraction(fraction);
      },
      [this](double fraction) {
        return HeightAtPressureFraction(fraction);
      });
}

double AnalyticPlane::pressure_at(const Eigen::Vector3d& p_RQ) const {
  return peak_pressure_ * PressureFractionAtRadiusSquared(p_RQ.x() * p_RQ.x() +
                                                          p_RQ.y() * p_RQ.y());
}

double AnalyticPlane::patch_radius() const {
  return std::sqrt(radius_squared_coefficients_[0]);
}

double AnalyticPlane::peak_pressure() const {
  return peak_pressure_;
}

Eigen::Vector3d AnalyticPlane::centroid() const {
  return Eigen::Vector3d(0.0, 0.0,
                         SurfaceCentroidHeight(radius_squared_coefficients_,
                                               height_coefficients_));
}

Eigen::Vector3d AnalyticPlane::normal() const {
  return Eigen::Vector3d::UnitZ();
}

double AnalyticPlane::RadiusSquaredAtPressureFraction(double fraction) const {
  return EvaluatePolynomial(radius_squared_coefficients_, fraction);
}

double AnalyticPlane::HeightAtPressureFraction(double fraction) const {
  return EvaluatePolynomial(height_coefficients_, fraction);
}

double AnalyticPlane::PressureFractionAtRadiusSquared(
    double radius_squared) const {
  if (modulus_lower_ == modulus_upper_) {
    const double half_distance = center_distance_ / 2.0;
    const double distance_from_center = std::sqrt(
        std::max(0.0, radius_squared + half_distance * half_distance));
    return std::clamp((radius_ - distance_from_center) / (penetration_ / 2.0),
                      0.0, 1.0);
  }
  return SolveMonotoneRadiusSquared(radius_squared_coefficients_,
                                    radius_squared);
}

AnalyticParaboloid::AnalyticParaboloid(double radius, double penetration,
                                       double sphere_modulus,
                                       double box_modulus)
    : radius_(radius),
      penetration_(penetration),
      sphere_modulus_(sphere_modulus),
      box_modulus_(box_modulus),
      sphere_center_height_(2.0 * radius - penetration) {
  ValidateInputs(radius, penetration, sphere_modulus, box_modulus);
  peak_pressure_ = sphere_modulus_ * box_modulus_ * penetration_ /
                   (radius_ * (sphere_modulus_ + box_modulus_));

  const double sphere_depth_at_peak =
      penetration_ * box_modulus_ / (sphere_modulus_ + box_modulus_);
  const double box_depth_at_peak = penetration_ - sphere_depth_at_peak;
  height_coefficients_[0] = radius_;
  height_coefficients_[1] = -box_depth_at_peak;

  const double sphere_axis_distance = radius_ - penetration_;
  radius_squared_coefficients_[0] =
      radius_ * radius_ - sphere_axis_distance * sphere_axis_distance;
  radius_squared_coefficients_[1] =
      -2.0 * radius_ * sphere_depth_at_peak -
      2.0 * sphere_axis_distance * box_depth_at_peak;
  radius_squared_coefficients_[2] =
      sphere_depth_at_peak * sphere_depth_at_peak -
      box_depth_at_peak * box_depth_at_peak;
}

double AnalyticParaboloid::force() const {
  return kPi * peak_pressure_ *
         IntegratePolynomialOnUnitInterval(radius_squared_coefficients_);
}

double AnalyticParaboloid::area() const {
  return kPi * radius_squared_coefficients_[0];
}

double AnalyticParaboloid::distance_to_surface(
    const Eigen::Vector3d& p_RQ) const {
  return DistanceToAxisymmetricPatch(
      p_RQ,
      [this](double fraction) {
        return RadiusSquaredAtPressureFraction(fraction);
      },
      [this](double fraction) {
        return HeightAtPressureFraction(fraction);
      });
}

double AnalyticParaboloid::pressure_at(const Eigen::Vector3d& p_RQ) const {
  return peak_pressure_ * PressureFractionAtRadiusSquared(p_RQ.x() * p_RQ.x() +
                                                          p_RQ.y() * p_RQ.y());
}

double AnalyticParaboloid::patch_radius() const {
  return std::sqrt(radius_squared_coefficients_[0]);
}

double AnalyticParaboloid::peak_pressure() const {
  return peak_pressure_;
}

Eigen::Vector3d AnalyticParaboloid::centroid() const {
  return Eigen::Vector3d(0.0, 0.0,
                         SurfaceCentroidHeight(radius_squared_coefficients_,
                                               height_coefficients_));
}

Eigen::Vector3d AnalyticParaboloid::normal() const {
  return Eigen::Vector3d::UnitZ();
}

double AnalyticParaboloid::RadiusSquaredAtPressureFraction(
    double fraction) const {
  return EvaluatePolynomial(radius_squared_coefficients_, fraction);
}

double AnalyticParaboloid::HeightAtPressureFraction(double fraction) const {
  return EvaluatePolynomial(height_coefficients_, fraction);
}

double AnalyticParaboloid::PressureFractionAtRadiusSquared(
    double radius_squared) const {
  if (sphere_modulus_ == box_modulus_) {
    return std::clamp(1.0 - radius_squared / radius_squared_coefficients_[0],
                      0.0, 1.0);
  }
  return SolveMonotoneRadiusSquared(radius_squared_coefficients_,
                                    radius_squared);
}

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
