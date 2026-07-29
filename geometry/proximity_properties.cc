#include "drake/geometry/proximity_properties.h"

#include <array>
#include <cmath>
#include <string>

namespace drake {
namespace geometry {
namespace internal {

const char* const kMaterialGroup = "material";
const char* const kFriction = "coulomb_friction";
const char* const kHcDissipation = "hunt_crossley_dissipation";
const char* const kRelaxationTime = "relaxation_time";
const char* const kPointStiffness = "point_contact_stiffness";

const char* const kHydroGroup = "hydroelastic";
const char* const kElastic = "hydroelastic_modulus";
const char* const kRezHint = "resolution_hint";
const char* const kComplianceType = "compliance_type";
const char* const kSlabThickness = "slab_thickness";
const char* const kMargin = "margin";
const char* const kCompliantRepresentation = "compliant_representation";
const char* const kVoxelSdfEvaluationMode = "voxel_sdf_evaluation_mode";
const char* const kVoxelSdfExtractionMethod = "voxel_sdf_extraction_method";

namespace {

// Use a switch() statement here, to ensure the compiler sends us a reminder
// when somebody adds a new value to the enum. New values must be listed here
// as well as in the list of kHydroelasticTypes below.
constexpr const char* EnumToChars(HydroelasticType enum_value) {
  switch (enum_value) {
    case HydroelasticType::kUndefined:
      return "undefined";
    case HydroelasticType::kRigid:
      return "rigid";
    case HydroelasticType::kCompliant:
      return "compliant";
  }
  DRAKE_UNREACHABLE();
}

template <typename Enum>
struct NamedEnum {
  // An implicit conversion here enables the convenient initializer_list syntax
  // that's used below, so we'll say NOLINTNEXTLINE(runtime/explicit).
  constexpr NamedEnum(Enum value_in)
      : value(value_in), name(EnumToChars(value_in)) {}
  const Enum value;
  const char* const name;
};

constexpr std::array<NamedEnum<HydroelasticType>, 3> kHydroelasticTypes{{
    {HydroelasticType::kUndefined},
    {HydroelasticType::kRigid},
    {HydroelasticType::kCompliant},
}};

}  // namespace

HydroelasticType GetHydroelasticTypeFromString(
    std::string_view hydroelastic_type) {
  for (const auto& [value, name] : kHydroelasticTypes) {
    if (name == hydroelastic_type) {
      return value;
    }
  }
  throw std::logic_error(
      fmt::format("Unknown hydroelastic_type: '{}'", hydroelastic_type));
}

std::string GetStringFromHydroelasticType(HydroelasticType hydroelastic_type) {
  for (const auto& [value, name] : kHydroelasticTypes) {
    if (value == hydroelastic_type) {
      return name;
    }
  }
  DRAKE_UNREACHABLE();
}

std::string_view to_string(const HydroelasticType& type) {
  return EnumToChars(type);
}

}  // namespace internal

void AddContactMaterial(
    std::optional<double> dissipation, std::optional<double> point_stiffness,
    const std::optional<multibody::CoulombFriction<double>>& friction,
    ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  if (dissipation.has_value()) {
    if (*dissipation < 0) {
      throw std::logic_error(fmt::format(
          "The dissipation can't be negative; given {}", *dissipation));
    }
    properties->AddProperty(internal::kMaterialGroup, internal::kHcDissipation,
                            *dissipation);
  }

  if (point_stiffness.has_value()) {
    if (*point_stiffness <= 0) {
      throw std::logic_error(fmt::format(
          "The point_contact_stiffness must be strictly positive; given {}",
          *point_stiffness));
    }
    properties->AddProperty(internal::kMaterialGroup, internal::kPointStiffness,
                            *point_stiffness);
  }

  if (friction.has_value()) {
    properties->AddProperty(internal::kMaterialGroup, internal::kFriction,
                            *friction);
  }
}

// NOTE: Although these functions currently do the same thing, we're leaving
// the two functions in place to facilitate future differences.

void AddRigidHydroelasticProperties(double resolution_hint,
                                    ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  properties->AddProperty(internal::kHydroGroup, internal::kRezHint,
                          resolution_hint);
  AddRigidHydroelasticProperties(properties);
}

void AddRigidHydroelasticProperties(ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  // The bare minimum of defining a rigid geometry is to declare its compliance
  // type. Downstream consumers (ProximityEngine) will determine if this is
  // sufficient.
  properties->AddProperty(internal::kHydroGroup, internal::kComplianceType,
                          internal::HydroelasticType::kRigid);
}

namespace {
void AddCompliantHydroelasticProperties(double hydroelastic_modulus,
                                        ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  // The bare minimum of defining a compliant geometry is to declare its
  // compliance type. Downstream consumers (ProximityEngine) will determine
  // if this is sufficient.
  if (hydroelastic_modulus <= 0) {
    throw std::logic_error(
        fmt::format("The hydroelastic modulus must be positive; given {}",
                    hydroelastic_modulus));
  }
  properties->AddProperty(internal::kHydroGroup, internal::kElastic,
                          hydroelastic_modulus);
  properties->AddProperty(internal::kHydroGroup, internal::kComplianceType,
                          internal::HydroelasticType::kCompliant);
}

void ValidateVoxelSdfEvaluationMode(VoxelSdfEvaluationMode mode) {
  switch (mode) {
    case VoxelSdfEvaluationMode::kPrimitiveSdf:
    case VoxelSdfEvaluationMode::kStoredGridTrilinear:
      return;
  }
  throw std::logic_error("The voxel SDF evaluation mode is invalid");
}

void ValidateVoxelSdfExtractionMethod(VoxelSdfExtractionMethod method) {
  switch (method) {
    case VoxelSdfExtractionMethod::kPlaneClip:
    case VoxelSdfExtractionMethod::kMarchingCubes:
      return;
  }
  throw std::logic_error("The voxel SDF extraction method is invalid");
}

void ValidateVoxelSdfProperties(double voxel_width,
                                double hydroelastic_modulus) {
  if (!(voxel_width > 0.0 && std::isfinite(voxel_width))) {
    throw std::logic_error(fmt::format(
        "The voxel width must be finite and strictly positive; given {}",
        voxel_width));
  }
  if (!(hydroelastic_modulus > 0.0 && std::isfinite(hydroelastic_modulus))) {
    throw std::logic_error(fmt::format(
        "The hydroelastic modulus must be finite and strictly positive; given "
        "{}",
        hydroelastic_modulus));
  }
}

void AddVoxelSdfProperties(
    double voxel_width, double hydroelastic_modulus,
    std::optional<VoxelSdfEvaluationMode> evaluation_mode,
    std::optional<VoxelSdfExtractionMethod> extraction_method,
    ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  // Validate all caller values before adding any property so a bad request
  // cannot partially modify the caller's set.
  ValidateVoxelSdfProperties(voxel_width, hydroelastic_modulus);
  if (evaluation_mode.has_value()) {
    ValidateVoxelSdfEvaluationMode(*evaluation_mode);
  }
  if (extraction_method.has_value()) {
    ValidateVoxelSdfExtractionMethod(*extraction_method);
  }
  const VoxelSdfEvaluationMode effective_evaluation =
      evaluation_mode.value_or(VoxelSdfEvaluationMode::kPrimitiveSdf);
  const VoxelSdfExtractionMethod effective_extraction =
      extraction_method.value_or(VoxelSdfExtractionMethod::kPlaneClip);
  if (effective_evaluation == VoxelSdfEvaluationMode::kStoredGridTrilinear &&
      effective_extraction == VoxelSdfExtractionMethod::kMarchingCubes) {
    throw std::logic_error(
        "Marching-cubes voxel SDF extraction requires primitive SDF "
        "evaluation");
  }
  properties->AddProperty(internal::kHydroGroup, internal::kRezHint,
                          voxel_width);
  properties->AddProperty(internal::kHydroGroup, internal::kElastic,
                          hydroelastic_modulus);
  properties->AddProperty(internal::kHydroGroup, internal::kComplianceType,
                          internal::HydroelasticType::kCompliant);
  properties->AddProperty(internal::kHydroGroup,
                          internal::kCompliantRepresentation,
                          std::string("voxel_sdf"));
  if (evaluation_mode.has_value()) {
    properties->AddProperty(internal::kHydroGroup,
                            internal::kVoxelSdfEvaluationMode,
                            *evaluation_mode);
  }
  if (extraction_method.has_value()) {
    properties->AddProperty(internal::kHydroGroup,
                            internal::kVoxelSdfExtractionMethod,
                            *extraction_method);
  }
}
}  // namespace

void AddCompliantHydroelasticProperties(double resolution_hint,
                                        double hydroelastic_modulus,
                                        ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  properties->AddProperty(internal::kHydroGroup, internal::kRezHint,
                          resolution_hint);
  AddCompliantHydroelasticProperties(hydroelastic_modulus, properties);
}

void AddCompliantHydroelasticVoxelSdfProperties(
    double voxel_width, double hydroelastic_modulus,
    ProximityProperties* properties) {
  AddVoxelSdfProperties(voxel_width, hydroelastic_modulus, std::nullopt,
                        std::nullopt, properties);
}

void AddCompliantHydroelasticVoxelSdfProperties(
    double voxel_width, double hydroelastic_modulus,
    VoxelSdfEvaluationMode mode, ProximityProperties* properties) {
  AddVoxelSdfProperties(voxel_width, hydroelastic_modulus, mode, std::nullopt,
                        properties);
}

void AddCompliantHydroelasticVoxelSdfProperties(
    double voxel_width, double hydroelastic_modulus,
    VoxelSdfEvaluationMode evaluation_mode,
    VoxelSdfExtractionMethod extraction_method,
    ProximityProperties* properties) {
  AddVoxelSdfProperties(voxel_width, hydroelastic_modulus, evaluation_mode,
                        extraction_method, properties);
}

void AddCompliantHydroelasticPropertiesForHalfSpace(
    double slab_thickness, double hydroelastic_modulus,
    ProximityProperties* properties) {
  DRAKE_DEMAND(properties != nullptr);
  properties->AddProperty(internal::kHydroGroup, internal::kSlabThickness,
                          slab_thickness);
  AddCompliantHydroelasticProperties(hydroelastic_modulus, properties);
}

}  // namespace geometry
}  // namespace drake
