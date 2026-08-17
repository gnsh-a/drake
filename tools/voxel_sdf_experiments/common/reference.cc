#include "drake/tools/voxel_sdf_experiments/common/reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

/* Integrals over the true equal-pressure surface, which is axisymmetric and
 parameterized here by pressure fraction rather than by radius. The pressure at
 parameter `fraction` is exactly `peak_pressure * fraction`, so a
 pressure-weighted integrand is the area-weighted one times `fraction`.

 Simpson over 4096 intervals on a smooth integrand leaves a relative error near
 1e-15, so these are exact for every purpose here. The quadrature is over the
 analytic surface and never touches a contact discretization. */
struct SurfaceIntegrals {
  double area{};             // Integral of dA.
  double height{};           // Integral of z dA.
  double pressure{};         // Integral of (p / peak) dA.
  double pressure_height{};  // Integral of z (p / peak) dA.

  double centroid_height() const { return height / area; }
  double pressure_centroid_height() const { return pressure_height / pressure; }
};

template <size_t Q, size_t Z>
SurfaceIntegrals CalcSurfaceIntegrals(
    const std::array<double, Q>& radius_squared_coefficients,
    const std::array<double, Z>& height_coefficients) {
  constexpr int kIntervals = 4096;
  // The axisymmetric area element is dA = pi * area_density * d(fraction),
  // with area_density as computed below.
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
    const double height = EvaluatePolynomial(height_coefficients, fraction);
    return std::array<double, 4>{area_density, area_density * height,
                                 area_density * fraction,
                                 area_density * fraction * height};
  };
  std::array<double, 4> integral{};
  for (int i = 0; i <= kIntervals; ++i) {
    const double fraction =
        static_cast<double>(i) / static_cast<double>(kIntervals);
    const double weight =
        i == 0 || i == kIntervals ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
    const auto value = sample(fraction);
    for (size_t term = 0; term < integral.size(); ++term) {
      integral[term] += weight * value[term];
    }
  }
  // Simpson's h/3, times the pi from the axisymmetric area element. Ratios
  // taken from these are unaffected by the scale; the areas are not.
  const double scale = kPi / (3.0 * static_cast<double>(kIntervals));
  return SurfaceIntegrals{scale * integral[0], scale * integral[1],
                          scale * integral[2], scale * integral[3]};
}

}  // namespace

Reference::~Reference() = default;

double ForceAtPenetration(const ReferenceFactory& factory, double penetration) {
  if (!factory) {
    throw std::logic_error("Reference factory must not be empty");
  }
  ThrowUnlessFinitePositive(penetration, "penetration");
  const std::unique_ptr<Reference> reference = factory(penetration);
  if (reference == nullptr) {
    throw std::logic_error("Reference factory returned null");
  }
  const double force = reference->force();
  if (!(std::isfinite(force) && force > 0.0)) {
    throw std::logic_error("Reference force must be finite and positive");
  }
  return force;
}

double EquilibriumPenetration(const ReferenceFactory& factory,
                              double target_force, double radius) {
  if (!factory) {
    throw std::logic_error("Reference factory must not be empty");
  }
  ThrowUnlessFinitePositive(target_force, "target_force");
  ThrowUnlessFinitePositive(radius, "radius");

  double lower = 0.0;
  double upper = 0.0;
  constexpr int kBracketIntervals = 512;
  for (int i = 1; i <= kBracketIntervals; ++i) {
    const double fraction = static_cast<double>(i) / kBracketIntervals;
    const double candidate = i == kBracketIntervals
                                 ? std::nextafter(2.0 * radius, 0.0)
                                 : 2.0 * radius * fraction;
    if (ForceAtPenetration(factory, candidate) >= target_force) {
      upper = candidate;
      break;
    }
    lower = candidate;
  }
  if (upper == 0.0) {
    throw std::logic_error(
        "target_force has no reference root below the "
        "diameter");
  }

  for (int iteration = 0; iteration < 128; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    if (middle == lower || middle == upper) break;
    if (ForceAtPenetration(factory, middle) < target_force) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return 0.5 * (lower + upper);
}

double StiffnessAtPenetration(const ReferenceFactory& factory,
                              double penetration) {
  ThrowUnlessFinitePositive(penetration, "penetration");
  /* A relative step keeps the balance between truncation and roundoff fixed
   across operating points. At 1e-6 the central difference is accurate to well
   under 1e-9 relative for these smooth loads. */
  const double step = 1.0e-6 * penetration;
  const double upper = ForceAtPenetration(factory, penetration + step);
  const double lower = ForceAtPenetration(factory, penetration - step);
  const double stiffness = (upper - lower) / (2.0 * step);
  if (!(std::isfinite(stiffness) && stiffness > 0.0)) {
    throw std::logic_error("Reference stiffness must be finite and positive");
  }
  return stiffness;
}

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

double AnalyticPlane::surface_area() const {
  return CalcSurfaceIntegrals(radius_squared_coefficients_,
                              height_coefficients_)
      .area;
}

Eigen::Vector3d AnalyticPlane::centroid() const {
  return Eigen::Vector3d(
      0.0, 0.0,
      CalcSurfaceIntegrals(radius_squared_coefficients_, height_coefficients_)
          .centroid_height());
}

Eigen::Vector3d AnalyticPlane::pressure_centroid() const {
  return Eigen::Vector3d(
      0.0, 0.0,
      CalcSurfaceIntegrals(radius_squared_coefficients_, height_coefficients_)
          .pressure_centroid_height());
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

double AnalyticParaboloid::surface_area() const {
  return CalcSurfaceIntegrals(radius_squared_coefficients_,
                              height_coefficients_)
      .area;
}

Eigen::Vector3d AnalyticParaboloid::centroid() const {
  return Eigen::Vector3d(
      0.0, 0.0,
      CalcSurfaceIntegrals(radius_squared_coefficients_, height_coefficients_)
          .centroid_height());
}

Eigen::Vector3d AnalyticParaboloid::pressure_centroid() const {
  return Eigen::Vector3d(
      0.0, 0.0,
      CalcSurfaceIntegrals(radius_squared_coefficients_, height_coefficients_)
          .pressure_centroid_height());
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
