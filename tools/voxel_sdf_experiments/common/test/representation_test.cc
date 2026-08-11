#include "drake/tools/voxel_sdf_experiments/common/representation.h"

#include <string>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

using geometry::HydroelasticContactRepresentation;
using geometry::VoxelSdfEvaluationMode;
using geometry::VoxelSdfExtractionMethod;
using geometry::internal::kCompliantRepresentation;
using geometry::internal::kElastic;
using geometry::internal::kHydroGroup;
using geometry::internal::kRezHint;
using geometry::internal::kVoxelSdfEvaluationMode;
using geometry::internal::kVoxelSdfExtractionMethod;

GTEST_TEST(RepresentationTest, PropertiesRoundTrip) {
  constexpr double kResolution = 0.0125;
  constexpr double kModulus = 2.5e8;
  for (const Representation representation :
       {Representation::kTet, Representation::kPlaneClip,
        Representation::kMarchingCubes}) {
    const geometry::ProximityProperties properties =
        MakeProperties(representation, kResolution, kModulus);
    EXPECT_EQ(properties.GetProperty<double>(kHydroGroup, kRezHint),
              kResolution);
    EXPECT_EQ(properties.GetProperty<double>(kHydroGroup, kElastic), kModulus);
    if (representation == Representation::kTet) {
      EXPECT_FALSE(
          properties.HasProperty(kHydroGroup, kCompliantRepresentation));
      EXPECT_FALSE(
          properties.HasProperty(kHydroGroup, kVoxelSdfEvaluationMode));
      EXPECT_FALSE(
          properties.HasProperty(kHydroGroup, kVoxelSdfExtractionMethod));
      continue;
    }
    EXPECT_EQ(properties.GetProperty<std::string>(kHydroGroup,
                                                  kCompliantRepresentation),
              "voxel_sdf");
    EXPECT_EQ(properties.GetProperty<VoxelSdfEvaluationMode>(
                  kHydroGroup, kVoxelSdfEvaluationMode),
              VoxelSdfEvaluationMode::kPrimitiveSdf);
    const auto expected_extraction =
        representation == Representation::kPlaneClip
            ? VoxelSdfExtractionMethod::kPlaneClip
            : VoxelSdfExtractionMethod::kMarchingCubes;
    EXPECT_EQ(properties.GetProperty<VoxelSdfExtractionMethod>(
                  kHydroGroup, kVoxelSdfExtractionMethod),
              expected_extraction);
  }
}

GTEST_TEST(RepresentationTest, SurfaceTypes) {
  EXPECT_EQ(SurfaceTypeFor(Representation::kTet),
            HydroelasticContactRepresentation::kPolygon);
  EXPECT_EQ(SurfaceTypeFor(Representation::kPlaneClip),
            HydroelasticContactRepresentation::kPolygon);
  EXPECT_EQ(SurfaceTypeFor(Representation::kMarchingCubes),
            HydroelasticContactRepresentation::kTriangle);
}

GTEST_TEST(RepresentationTest, StringRoundTrip) {
  for (const std::string value : {"tet", "plane_clip", "marching_cubes"}) {
    EXPECT_EQ(to_string(ParseRepresentation(value)), value);
  }
  EXPECT_THROW(ParseRepresentation("stored_grid"), std::logic_error);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
