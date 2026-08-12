#pragma once

#include <array>
#include <functional>
#include <memory>

#include "drake/common/eigen_types.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

/* Analytic reference for an axisymmetric frozen contact patch. The area is
 projected onto the xy plane, matching normal-force integration. The centroid
 is weighted by the true surface area. */
class Reference {
 public:
  virtual ~Reference();

  virtual double force() const = 0;
  virtual double area() const = 0;
  virtual double distance_to_surface(const Eigen::Vector3d& p_RQ) const = 0;
  virtual double pressure_at(const Eigen::Vector3d& p_RQ) const = 0;
  virtual double patch_radius() const = 0;
  virtual double peak_pressure() const = 0;
  virtual Eigen::Vector3d centroid() const = 0;
  virtual Eigen::Vector3d normal() const = 0;
};

/* Builds an analytic reference at the requested penetration. Callers capture
 scene-specific dimensions and moduli, leaving penetration as the independent
 variable for load inversion. */
using ReferenceFactory =
    std::function<std::unique_ptr<Reference>(double penetration)>;

/* Evaluates the reference load at `penetration`. */
double ForceAtPenetration(const ReferenceFactory& factory, double penetration);

/* Returns the unique penetration in (0, 2 * radius) whose reference load is
 `target_force`. Reference::force() must be monotone increasing over that
 interval. */
double EquilibriumPenetration(const ReferenceFactory& factory,
                              double target_force, double radius);

/* Returns the contact stiffness dF/d(penetration) at `penetration`, by central
 difference on the reference load.

 This is deliberately numeric rather than closed form. The closed-form stiffness
 differs per scene -- for two equal spheres it is pi E x delta / (2 R) with
 x = R - delta / 2, but the sphere-box paraboloid is roughly 1.85x stiffer at
 the same penetration -- so a single analytic expression silently misreports
 every scene but one. Differencing Reference::force() stays correct for whatever
 reference it is handed. */
double StiffnessAtPenetration(const ReferenceFactory& factory,
                              double penetration);

/* Exact reference for two equal-radius spheres. The lower sphere is centered
 at the reference origin and the upper sphere is centered on +z. With equal
 moduli, the equilibrium surface is the flat mid-plane. Unequal moduli are
 also supported; the exact equilibrium surface is then axisymmetric. */
class AnalyticPlane final : public Reference {
 public:
  AnalyticPlane(double radius, double penetration, double modulus_lower,
                double modulus_upper);

  double force() const final;
  double area() const final;
  double distance_to_surface(const Eigen::Vector3d& p_RQ) const final;
  double pressure_at(const Eigen::Vector3d& p_RQ) const final;
  double patch_radius() const final;
  double peak_pressure() const final;
  Eigen::Vector3d centroid() const final;
  Eigen::Vector3d normal() const final;

 private:
  double RadiusSquaredAtPressureFraction(double fraction) const;
  double HeightAtPressureFraction(double fraction) const;
  double PressureFractionAtRadiusSquared(double radius_squared) const;

  double radius_{};
  double penetration_{};
  double modulus_lower_{};
  double modulus_upper_{};
  double center_distance_{};
  double peak_pressure_{};
  std::array<double, 5> radius_squared_coefficients_{};
  std::array<double, 3> height_coefficients_{};
};

/* Exact reference for a sphere above a box whose smallest half-width equals
 the sphere radius. The box is centered at the reference origin with its top
 face at +radius; the sphere center lies on +z. With equal moduli, the
 equilibrium surface is a paraboloid. Unequal moduli are also supported. */
class AnalyticParaboloid final : public Reference {
 public:
  AnalyticParaboloid(double radius, double penetration, double sphere_modulus,
                     double box_modulus);

  double force() const final;
  double area() const final;
  double distance_to_surface(const Eigen::Vector3d& p_RQ) const final;
  double pressure_at(const Eigen::Vector3d& p_RQ) const final;
  double patch_radius() const final;
  double peak_pressure() const final;
  Eigen::Vector3d centroid() const final;
  Eigen::Vector3d normal() const final;

 private:
  double RadiusSquaredAtPressureFraction(double fraction) const;
  double HeightAtPressureFraction(double fraction) const;
  double PressureFractionAtRadiusSquared(double radius_squared) const;

  double radius_{};
  double penetration_{};
  double sphere_modulus_{};
  double box_modulus_{};
  double sphere_center_height_{};
  double peak_pressure_{};
  std::array<double, 3> radius_squared_coefficients_{};
  std::array<double, 2> height_coefficients_{};
};

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
