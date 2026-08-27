#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/marching_cubes_table.h"
#include "drake/geometry/proximity/voxel_sdf_contact_common.h"
#include "drake/geometry/proximity/voxel_sdf_geometry.h"
#include "drake/geometry/proximity/voxel_sdf_marching_cubes_contact.h"
#include "drake/geometry/proximity/voxel_sdf_polygon_contact.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {
namespace {

using Eigen::Vector3d;
using math::RigidTransformd;

constexpr double kRadius = 1.0;
constexpr double kHydroelasticModulus = 1.0e5;
constexpr std::array<double, 3> kSeparations{2.4, 1.95, 1.7};

enum class Method {
  kTet,
  kAffine,
  kMarchingCubes,
};

struct SurfaceSummary {
  bool has_surface{};
  int faces{};
  int vertices{};
  double area{};
};

SurfaceSummary Summarize(const ContactSurface<double>* surface, Method method) {
  if (surface == nullptr) return {};
  DRAKE_DEMAND(surface->is_triangle() == (method == Method::kMarchingCubes));
  DRAKE_DEMAND(surface->HasGradE_M());
  DRAKE_DEMAND(surface->HasGradE_N());
  DRAKE_DEMAND(std::isfinite(surface->total_area()));
  DRAKE_DEMAND(surface->total_area() > 0.0);
  return SurfaceSummary{true, surface->num_faces(), surface->num_vertices(),
                        surface->total_area()};
}

class ExtractionTimingBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) final {
    Method method{};
    if (state.name().find("Tet") != std::string::npos) {
      method = Method::kTet;
    } else if (state.name().find("Affine") != std::string::npos) {
      method = Method::kAffine;
    } else {
      method = Method::kMarchingCubes;
    }
    SetUpCase(state, method);
  }

  void RunBenchmark(benchmark::State* state, Method method) {
    DRAKE_DEMAND(state != nullptr);
    std::unique_ptr<ContactSurface<double>> surface;
    for (auto _ : *state) {
      // Surface destruction is not part of extraction. Doing this at the start
      // also makes every measured iteration allocate a fresh owning result.
      state->PauseTiming();
      surface.reset();
      state->ResumeTiming();
      surface = Extract(method);
      benchmark::DoNotOptimize(surface.get());
    }
    const int64_t candidates = CandidateCount(method);
    state->SetItemsProcessed(state->iterations() * candidates);
    state->counters["area"] = summary_.area;
    state->counters["candidates"] = static_cast<double>(candidates);
    state->counters["faces"] = summary_.faces;
    state->counters["has_surface"] = summary_.has_surface ? 1.0 : 0.0;
    state->counters["vertices"] = summary_.vertices;
  }

 private:
  void SetUpCase(const benchmark::State& state, Method method) {
    const int cells_per_axis = state.range(0);
    const int pose = state.range(1);
    DRAKE_DEMAND(cells_per_axis > 0);
    DRAKE_DEMAND(pose >= 0 && pose < static_cast<int>(kSeparations.size()));

    const double resolution = 2.0 * kRadius / cells_per_axis;
    const Vector3d direction = Vector3d(1.0, 0.37, 0.19).normalized();
    X_WB_ = RigidTransformd(kSeparations[pose] * direction);
    X_AB_ = X_WA_.InvertAndCompose(X_WB_);
    X_BA_ = X_AB_.inverse();

    if (method == Method::kTet) {
      BuildTetRepresentations(resolution);
      PrepareTetCandidates();
    } else {
      const VoxelSdfExtractionMethod extraction_method =
          method == Method::kAffine ? VoxelSdfExtractionMethod::kPlaneClip
                                    : VoxelSdfExtractionMethod::kMarchingCubes;
      voxel_A_ = std::make_unique<VoxelSdfGeometry>(
          Sphere(kRadius), resolution, kHydroelasticModulus,
          VoxelSdfEvaluationMode::kPrimitiveSdf, extraction_method);
      voxel_B_ = std::make_unique<VoxelSdfGeometry>(
          Sphere(kRadius), resolution, kHydroelasticModulus,
          VoxelSdfEvaluationMode::kPrimitiveSdf, extraction_method);
      PrepareVoxelCandidates(method);
    }

    const std::unique_ptr<ContactSurface<double>> production =
        CalcProductionSurface(method);
    const std::unique_ptr<ContactSurface<double>> prepared = Extract(method);
    DRAKE_DEMAND((production == nullptr) == (prepared == nullptr));
    if (production != nullptr) {
      DRAKE_DEMAND(production->Equal(*prepared));
    }
    summary_ = Summarize(prepared.get(), method);
    if (pose != 0) DRAKE_DEMAND(summary_.has_surface);
  }

  void BuildTetRepresentations(double resolution_hint) {
    auto make_compliant_sphere = [resolution_hint]() {
      auto mesh =
          std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
              Sphere(kRadius), resolution_hint,
              TessellationStrategy::kDenseInteriorVertices));
      auto pressure = std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeSpherePressureField<double>(Sphere(kRadius), mesh.get(),
                                          kHydroelasticModulus));
      return std::make_unique<CompliantMesh>(std::move(mesh),
                                             std::move(pressure));
    };
    tet_A_ = make_compliant_sphere();
    tet_B_ = make_compliant_sphere();
  }

  void PrepareTetCandidates() {
    tet_candidates_.clear();
    auto callback = [this](int tet_A, int tet_B) {
      tet_candidates_.emplace_back(tet_A, tet_B);
      return BvttCallbackResult::Continue;
    };
    tet_A_->bvh().Collide(tet_B_->bvh(), X_AB_, callback);
  }

  bool UnitMayOverlapB(const Vector3d& center_A,
                       const Vector3d& half_width_A) const {
    const Vector3d half_width_B = -voxel_B_->lower_cell_boundary();
    const Vector3d extent_B_A =
        X_AB_.rotation().matrix().cwiseAbs() * half_width_B;
    return ((center_A - X_AB_.translation()).cwiseAbs().array() <=
            (extent_B_A + half_width_A).array())
        .all();
  }

  void PrepareVoxelCandidates(Method method) {
    voxel_candidates_.clear();
    const Vector3<int> counts = method == Method::kAffine
                                    ? voxel_A_->cell_counts()
                                    : voxel_A_->mc_cube_counts();
    for (int k = 0; k < counts[2]; ++k) {
      for (int j = 0; j < counts[1]; ++j) {
        for (int i = 0; i < counts[0]; ++i) {
          Vector3d center_A;
          Vector3d half_width_A;
          if (method == Method::kAffine) {
            center_A = voxel_A_->cell_center(i, j, k);
            half_width_A = Vector3d::Constant(0.5 * voxel_A_->voxel_width());
          } else {
            const Vector3d lower_A = voxel_A_->mc_node_position(i, j, k);
            const Vector3d upper_A =
                voxel_A_->mc_node_position(i + 1, j + 1, k + 1);
            center_A = 0.5 * (lower_A + upper_A);
            half_width_A = 0.5 * (upper_A - lower_A).cwiseAbs();
          }
          if (UnitMayOverlapB(center_A, half_width_A)) {
            voxel_candidates_.emplace_back(i, j, k);
          }
        }
      }
    }
  }

  std::unique_ptr<ContactSurface<double>> CalcProductionSurface(
      Method method) const {
    if (method == Method::kTet) {
      return ComputeContactSurfaceFromCompliantVolumes(
          id_A_, *tet_A_, X_WA_, id_B_, *tet_B_, X_WB_,
          HydroelasticContactRepresentation::kPolygon);
    }
    if (method == Method::kAffine) {
      return CalcVoxelSdfPolygonContact(*voxel_A_, X_WA_, id_A_, *voxel_B_,
                                        X_WB_, id_B_);
    }
    return CalcVoxelSdfMarchingCubesContact(*voxel_A_, X_WA_, id_A_, *voxel_B_,
                                            X_WB_, id_B_);
  }

  std::unique_ptr<ContactSurface<double>> Extract(Method method) const {
    switch (method) {
      case Method::kTet:
        return ExtractTet();
      case Method::kAffine:
        return ExtractAffine();
      case Method::kMarchingCubes:
        return ExtractMarchingCubes();
    }
    DRAKE_UNREACHABLE();
  }

  std::unique_ptr<ContactSurface<double>> ExtractTet() const {
    const auto& field_A = tet_A_->pressure();
    const auto& field_B = tet_B_->pressure();
    PolyMeshBuilder<double> builder_A;
    std::vector<Vector3d> grad_p_A_A_per_face;
    std::vector<Vector3d> grad_p_B_A_per_face;
    const math::RotationMatrixd R_BA = X_AB_.rotation().inverse();
    for (const auto& [tet_A, tet_B] : tet_candidates_) {
      Plane<double> equilibrium_plane_A(Vector3d::UnitZ(), Vector3d::Zero());
      if (!CalcEquilibriumPlane(tet_A, field_A, tet_B, field_B, X_AB_,
                                &equilibrium_plane_A)) {
        continue;
      }
      const Vector3d nhat_BA_A = equilibrium_plane_A.unit_normal();
      if (!IsPlaneNormalAlongPressureGradient(nhat_BA_A, tet_A, field_A) ||
          !IsPlaneNormalAlongPressureGradient(R_BA * -nhat_BA_A, tet_B,
                                              field_B)) {
        continue;
      }
      const auto [vertices_A, faces] =
          IntersectTetrahedra(tet_A, field_A.mesh(), tet_B, field_B.mesh(),
                              X_AB_, equilibrium_plane_A);
      static_cast<void>(faces);
      if (vertices_A.size() < 3) continue;
      std::vector<int> vertex_indices;
      vertex_indices.reserve(vertices_A.size());
      for (const Vector3d& vertex_A : vertices_A) {
        vertex_indices.push_back(builder_A.AddVertex(
            vertex_A, field_A.EvaluateCartesian(tet_A, vertex_A)));
      }
      const int faces_added = builder_A.AddPolygon(
          vertex_indices, nhat_BA_A, field_A.EvaluateGradient(tet_A));
      const Vector3d grad_p_A_A = field_A.EvaluateGradient(tet_A);
      const Vector3d grad_p_B_A =
          X_AB_.rotation() * field_B.EvaluateGradient(tet_B);
      for (int face = 0; face < faces_added; ++face) {
        grad_p_A_A_per_face.push_back(grad_p_A_A);
        grad_p_B_A_per_face.push_back(grad_p_B_A);
      }
    }
    return FinalizeContactSurface<PolyMeshBuilder<double>>(
        std::move(builder_A), std::move(grad_p_A_A_per_face),
        std::move(grad_p_B_A_per_face), X_WA_, id_A_, id_B_);
  }

  std::unique_ptr<ContactSurface<double>> ExtractAffine() const {
    PolyMeshBuilder<double> builder_A;
    std::vector<Vector3d> grad_p_A_A_per_face;
    std::vector<Vector3d> grad_p_B_A_per_face;
    for (const Vector3<int>& index : voxel_candidates_) {
      const int i = index[0];
      const int j = index[1];
      const int k = index[2];
      const Vector3d center_A = voxel_A_->cell_center(i, j, k);
      const std::vector<VoxelSdfGeometry::SdfBranch> branches_A =
          voxel_A_->CalcCellSdfBranches(i, j, k);
      const std::vector<VoxelSdfGeometry::SdfBranch> branches_B =
          voxel_B_->EvaluateSdfBranches(X_BA_ * center_A);
      DRAKE_DEMAND(branches_A.size() == 1);
      DRAKE_DEMAND(branches_B.size() == 1);
      VoxelSdfGeometry::SdfSample sample_B_A = branches_B[0].sample;
      sample_B_A.gradient = X_AB_.rotation() * sample_B_A.gradient;
      const std::optional<VoxelSdfContactPolygon> polygon =
          CalcVoxelSdfContactPolygon(
              center_A, voxel_A_->voxel_width(),
              MakePressureField(*voxel_A_, branches_A[0].sample),
              MakePressureField(*voxel_B_, sample_B_A));
      if (!polygon.has_value()) continue;
      std::vector<int> vertex_indices;
      vertex_indices.reserve(polygon->vertices_A.size());
      for (int v = 0; v < static_cast<int>(polygon->vertices_A.size()); ++v) {
        vertex_indices.push_back(
            builder_A.AddVertex(polygon->vertices_A[v], polygon->pressures[v]));
      }
      DRAKE_DEMAND(builder_A.AddPolygon(vertex_indices, polygon->nhat_BA_A,
                                        polygon->grad_p_A) == 1);
      grad_p_A_A_per_face.push_back(polygon->grad_p_A);
      grad_p_B_A_per_face.push_back(polygon->grad_p_B_A);
    }
    return FinalizeContactSurface<PolyMeshBuilder<double>>(
        std::move(builder_A), std::move(grad_p_A_A_per_face),
        std::move(grad_p_B_A_per_face), X_WA_, id_A_, id_B_);
  }

  std::unique_ptr<ContactSurface<double>> ExtractMarchingCubes() const {
    MarchingCubesContactBuilder builder(voxel_A_->voxel_width());
    for (const Vector3<int>& cube_index : voxel_candidates_) {
      std::array<MarchingCubesNode, 8> nodes_A;
      for (int corner = 0; corner < 8; ++corner) {
        const auto& offset = kMcCornerOffsets[corner];
        const int node_i = cube_index[0] + offset[0];
        const int node_j = cube_index[1] + offset[1];
        const int node_k = cube_index[2] + offset[2];
        const Vector3d p_AN_A =
            voxel_A_->mc_node_position(node_i, node_j, node_k);
        const double phi_A = voxel_A_->mc_node_value(node_i, node_j, node_k);
        const double phi_B = voxel_B_->EvaluateSdf(X_BA_ * p_AN_A).value;
        nodes_A[corner] =
            MarchingCubesNode{p_AN_A, -voxel_A_->pressure_scale() * phi_A,
                              -voxel_B_->pressure_scale() * phi_B};
      }
      builder.AddCube(cube_index, nodes_A);
    }
    MarchingCubesMeshData mesh_data = std::move(builder).TakeMeshData();
    std::vector<Vector3d> grad_p_A_A_per_face;
    std::vector<Vector3d> grad_p_B_A_per_face;
    grad_p_A_A_per_face.reserve(mesh_data.face_centroids_A.size());
    grad_p_B_A_per_face.reserve(mesh_data.face_centroids_A.size());
    for (const Vector3d& centroid_A : mesh_data.face_centroids_A) {
      grad_p_A_A_per_face.push_back(-voxel_A_->pressure_scale() *
                                    voxel_A_->EvaluateSdf(centroid_A).gradient);
      grad_p_B_A_per_face.push_back(
          X_AB_.rotation() *
          (-voxel_B_->pressure_scale() *
           voxel_B_->EvaluateSdf(X_BA_ * centroid_A).gradient));
    }
    return FinalizeContactSurface<TriMeshBuilder<double>>(
        std::move(mesh_data.builder_A), std::move(grad_p_A_A_per_face),
        std::move(grad_p_B_A_per_face), X_WA_, id_A_, id_B_);
  }

  int64_t CandidateCount(Method method) const {
    return method == Method::kTet ? tet_candidates_.size()
                                  : voxel_candidates_.size();
  }

  const GeometryId id_A_{GeometryId::get_new_id()};
  const GeometryId id_B_{GeometryId::get_new_id()};
  const RigidTransformd X_WA_{};
  RigidTransformd X_WB_{};
  RigidTransformd X_AB_{};
  RigidTransformd X_BA_{};
  std::unique_ptr<CompliantMesh> tet_A_;
  std::unique_ptr<CompliantMesh> tet_B_;
  std::unique_ptr<VoxelSdfGeometry> voxel_A_;
  std::unique_ptr<VoxelSdfGeometry> voxel_B_;
  std::vector<std::pair<int, int>> tet_candidates_;
  std::vector<Vector3<int>> voxel_candidates_;
  SurfaceSummary summary_;
};

BENCHMARK_DEFINE_F(ExtractionTimingBenchmark, TetPostBroadphase)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  RunBenchmark(&state, Method::kTet);
}

BENCHMARK_DEFINE_F(ExtractionTimingBenchmark, AffinePostBroadphase)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  RunBenchmark(&state, Method::kAffine);
}

BENCHMARK_DEFINE_F(ExtractionTimingBenchmark, McPostBroadphase)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  RunBenchmark(&state, Method::kMarchingCubes);
}

#define REGISTER_EXTRACTION_BENCHMARK(name)             \
  BENCHMARK_REGISTER_F(ExtractionTimingBenchmark, name) \
      ->ArgsProduct({{8, 16, 32}, {0, 1, 2}})           \
      ->ArgNames({"cells_per_axis", "pose"})            \
      ->Unit(benchmark::kMicrosecond)                   \
      ->MinTime(0.1)                                    \
      ->MinWarmUpTime(0.02)

REGISTER_EXTRACTION_BENCHMARK(TetPostBroadphase);
REGISTER_EXTRACTION_BENCHMARK(AffinePostBroadphase);
REGISTER_EXTRACTION_BENCHMARK(McPostBroadphase);

#undef REGISTER_EXTRACTION_BENCHMARK

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
