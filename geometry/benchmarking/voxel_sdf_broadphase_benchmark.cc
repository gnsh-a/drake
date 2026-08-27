#include <memory>

#include <benchmark/benchmark.h>

#include "drake/geometry/proximity/voxel_sdf_contact_common.h"
#include "drake/geometry/proximity/voxel_sdf_geometry.h"
#include "drake/geometry/proximity/voxel_sdf_marching_cubes_contact.h"
#include "drake/geometry/proximity/voxel_sdf_polygon_contact.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using math::RigidTransformd;

constexpr double kRadius = 0.1;
constexpr double kPenetration = 0.0199;
constexpr double kModulus = 1.0e8;
constexpr double kVoxelWidths[] = {0.01, 0.008, 0.005, 0.0025};

class VoxelSdfBroadphaseBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    const auto method = state.range(0) == 0
                            ? VoxelSdfExtractionMethod::kPlaneClip
                            : VoxelSdfExtractionMethod::kMarchingCubes;
    voxel_width_ = kVoxelWidths[state.range(1)];
    sphere_ = std::make_unique<VoxelSdfGeometry>(
        Sphere(kRadius), voxel_width_, kModulus,
        VoxelSdfEvaluationMode::kPrimitiveSdf, method);
    box_ = std::make_unique<VoxelSdfGeometry>(
        Box(4.0 * kRadius, 4.0 * kRadius, 2.0 * kRadius), voxel_width_,
        kModulus, VoxelSdfEvaluationMode::kPrimitiveSdf, method);
    X_WS_ = RigidTransformd(
        Vector3<double>(0.0, 0.0, 2.0 * kRadius - kPenetration));
  }

 protected:
  double voxel_width_{};
  std::unique_ptr<VoxelSdfGeometry> sphere_;
  std::unique_ptr<VoxelSdfGeometry> box_;
  RigidTransformd X_WS_;
  const GeometryId sphere_id_{GeometryId::get_new_id()};
  const GeometryId box_id_{GeometryId::get_new_id()};
};

BENCHMARK_DEFINE_F(VoxelSdfBroadphaseBenchmark, SphereBox)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  std::unique_ptr<ContactSurface<double>> surface;
  for (auto _ : state) {
    if (state.range(0) == 0) {
      surface = CalcVoxelSdfPolygonContact(*sphere_, X_WS_, sphere_id_, *box_,
                                           RigidTransformd(), box_id_);
    } else {
      surface = CalcVoxelSdfMarchingCubesContact(
          *sphere_, X_WS_, sphere_id_, *box_, RigidTransformd(), box_id_);
    }
    benchmark::DoNotOptimize(surface.get());
  }
  const auto grid = state.range(0) == 0 ? VoxelSdfTraversalGrid::kPlaneClipCells
                                        : VoxelSdfTraversalGrid::kMarchingCubes;
  const VoxelSdfIndexRange full = MakeFullVoxelSdfIndexRange(*sphere_, grid);
  const VoxelSdfIndexRange candidate =
      CalcVoxelSdfCandidateRange(*sphere_, *box_, X_WS_.inverse(), grid);
  state.counters["candidate_units"] = candidate.num_elements();
  state.counters["full_units"] = full.num_elements();
  state.SetLabel(state.range(0) == 0 ? "plane_clip" : "marching_cubes");
}

BENCHMARK_DEFINE_F(VoxelSdfBroadphaseBenchmark, SphereBoxFullTraversal)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const auto grid = state.range(0) == 0 ? VoxelSdfTraversalGrid::kPlaneClipCells
                                        : VoxelSdfTraversalGrid::kMarchingCubes;
  const VoxelSdfIndexRange full = MakeFullVoxelSdfIndexRange(*sphere_, grid);
  std::unique_ptr<ContactSurface<double>> surface;
  for (auto _ : state) {
    if (state.range(0) == 0) {
      surface = CalcVoxelSdfPolygonContactOverRange(
          *sphere_, X_WS_, sphere_id_, *box_, RigidTransformd(), box_id_, full);
    } else {
      surface = CalcVoxelSdfMarchingCubesContactOverRange(
          *sphere_, X_WS_, sphere_id_, *box_, RigidTransformd(), box_id_, full);
    }
    benchmark::DoNotOptimize(surface.get());
  }
  state.counters["full_units"] = full.num_elements();
  state.SetLabel(state.range(0) == 0 ? "plane_clip_full"
                                     : "marching_cubes_full");
}

BENCHMARK_REGISTER_F(VoxelSdfBroadphaseBenchmark, SphereBox)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->MinWarmUpTime(0.05)
    ->Args({0, 0})
    ->Args({0, 1})
    ->Args({0, 2})
    ->Args({0, 3})
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({1, 2})
    ->Args({1, 3});

BENCHMARK_REGISTER_F(VoxelSdfBroadphaseBenchmark, SphereBoxFullTraversal)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(0.1)
    ->MinWarmUpTime(0.05)
    ->Args({0, 0})
    ->Args({0, 1})
    ->Args({0, 2})
    ->Args({0, 3})
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({1, 2})
    ->Args({1, 3});

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
