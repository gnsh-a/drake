#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/common/never_destroyed.h"
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
constexpr int kCellsPerAxis = 16;
constexpr double kSeparation = 1.7;

enum class CandidateKind {
  kReject = 0,
  kInterior = 1,
  kRim = 2,
};

CandidateKind ReadKind(const benchmark::State& state) {
  DRAKE_DEMAND(state.range(0) >= 0 && state.range(0) <= 2);
  return static_cast<CandidateKind>(state.range(0));
}

std::string_view KindName(CandidateKind kind) {
  switch (kind) {
    case CandidateKind::kReject:
      return "reject";
    case CandidateKind::kInterior:
      return "interior";
    case CandidateKind::kRim:
      return "rim";
  }
  DRAKE_UNREACHABLE();
}

int KindIndex(CandidateKind kind) {
  return static_cast<int>(kind);
}

struct TetCase {
  std::pair<int, int> pair{};
  Plane<double> plane_A{Vector3d::UnitZ(), Vector3d::Zero()};
  bool plane_ready{};
  std::vector<Vector3d> vertices_A;
  std::vector<double> pressures;
  Vector3d nhat_BA_A{};
  Vector3d grad_p_A_A{};
  Vector3d grad_p_B_A{};

  bool emits() const { return vertices_A.size() >= 3; }
};

struct AffineHostData {
  Vector3d center_A{};
  PressureFieldSample sdf_A;
};

struct AffineInput {
  Vector3d center_A{};
  PressureFieldSample sdf_A;
  PressureFieldSample sdf_B_A;
};

struct AffineCase {
  Vector3<int> index{};
  AffineInput input;
  std::optional<VoxelSdfContactPolygon> polygon;
};

struct McHostNodes {
  std::array<Vector3d, 8> positions_A;
  std::array<double, 8> pressures_A;
};

struct McCase {
  Vector3<int> index{};
  std::array<MarchingCubesNode, 8> nodes_A;
  std::vector<Vector3d> centroids_A;
  int case_index{};
  int faces{};
  int vertices{};
};

class LocalKernelData final {
 public:
  LocalKernelData() {
    const double resolution = 2.0 * kRadius / kCellsPerAxis;
    const Vector3d direction = Vector3d(1.0, 0.37, 0.19).normalized();
    X_AB_ = RigidTransformd(kSeparation * direction);
    X_BA_ = X_AB_.inverse();
    BuildTetRepresentations(resolution);
    voxel_affine_A_ = std::make_unique<VoxelSdfGeometry>(
        Sphere(kRadius), resolution, kHydroelasticModulus,
        VoxelSdfEvaluationMode::kPrimitiveSdf,
        VoxelSdfExtractionMethod::kPlaneClip);
    voxel_affine_B_ = std::make_unique<VoxelSdfGeometry>(
        Sphere(kRadius), resolution, kHydroelasticModulus,
        VoxelSdfEvaluationMode::kPrimitiveSdf,
        VoxelSdfExtractionMethod::kPlaneClip);
    voxel_mc_A_ = std::make_unique<VoxelSdfGeometry>(
        Sphere(kRadius), resolution, kHydroelasticModulus,
        VoxelSdfEvaluationMode::kPrimitiveSdf,
        VoxelSdfExtractionMethod::kMarchingCubes);
    voxel_mc_B_ = std::make_unique<VoxelSdfGeometry>(
        Sphere(kRadius), resolution, kHydroelasticModulus,
        VoxelSdfEvaluationMode::kPrimitiveSdf,
        VoxelSdfExtractionMethod::kMarchingCubes);
    SelectTetCases();
    SelectAffineCases();
    SelectMcCases();
  }

  static const LocalKernelData& get() {
    static const never_destroyed<LocalKernelData> data;
    return data.access();
  }

  const TetCase& tet_case(CandidateKind kind) const {
    return tet_cases_[KindIndex(kind)];
  }

  const AffineCase& affine_case(CandidateKind kind) const {
    return affine_cases_[KindIndex(kind)];
  }

  const McCase& mc_case(CandidateKind kind) const {
    return mc_cases_[KindIndex(kind)];
  }

  bool CalcTetFieldAndPlane(const std::pair<int, int>& pair,
                            Plane<double>* plane_A) const {
    DRAKE_DEMAND(plane_A != nullptr);
    const auto& field_A = tet_A_->pressure();
    const auto& field_B = tet_B_->pressure();
    if (!CalcEquilibriumPlane(pair.first, field_A, pair.second, field_B, X_AB_,
                              plane_A)) {
      return false;
    }
    const Vector3d nhat_BA_A = plane_A->unit_normal();
    return IsPlaneNormalAlongPressureGradient(nhat_BA_A, pair.first, field_A) &&
           IsPlaneNormalAlongPressureGradient(
               X_AB_.rotation().inverse() * -nhat_BA_A, pair.second, field_B);
  }

  std::pair<std::vector<Vector3d>, std::vector<int>> IntersectTetPair(
      const std::pair<int, int>& pair, const Plane<double>& plane_A) const {
    return IntersectTetrahedra(pair.first, tet_A_->mesh(), pair.second,
                               tet_B_->mesh(), X_AB_, plane_A);
  }

  TetCase EvaluateTetPair(const std::pair<int, int>& pair) const {
    TetCase result;
    result.pair = pair;
    result.plane_ready = CalcTetFieldAndPlane(pair, &result.plane_A);
    if (!result.plane_ready) return result;
    result.nhat_BA_A = result.plane_A.unit_normal();
    auto [vertices_A, faces] = IntersectTetPair(pair, result.plane_A);
    static_cast<void>(faces);
    if (vertices_A.size() < 3) return result;
    result.vertices_A = std::move(vertices_A);
    result.grad_p_A_A = tet_A_->pressure().EvaluateGradient(pair.first);
    result.grad_p_B_A =
        X_AB_.rotation() * tet_B_->pressure().EvaluateGradient(pair.second);
    result.pressures.reserve(result.vertices_A.size());
    for (const Vector3d& vertex_A : result.vertices_A) {
      result.pressures.push_back(
          tet_A_->pressure().EvaluateCartesian(pair.first, vertex_A));
    }
    return result;
  }

  int EmitTet(const TetCase& candidate) const {
    PolyMeshBuilder<double> builder;
    if (!candidate.emits()) return builder.num_faces();
    std::vector<int> indices;
    indices.reserve(candidate.vertices_A.size());
    for (int v = 0; v < static_cast<int>(candidate.vertices_A.size()); ++v) {
      indices.push_back(
          builder.AddVertex(candidate.vertices_A[v], candidate.pressures[v]));
    }
    builder.AddPolygon(indices, candidate.nhat_BA_A, candidate.grad_p_A_A);
    return builder.num_faces();
  }

  AffineHostData LoadAffineA(const Vector3<int>& index) const {
    AffineHostData result;
    result.center_A =
        voxel_affine_A_->cell_center(index[0], index[1], index[2]);
    const auto branches =
        voxel_affine_A_->CalcCellSdfBranches(index[0], index[1], index[2]);
    DRAKE_DEMAND(branches.size() == 1);
    result.sdf_A = MakePressureField(*voxel_affine_A_, branches[0].sample);
    return result;
  }

  AffineInput QueryAffineB(const AffineHostData& host) const {
    AffineInput result;
    result.center_A = host.center_A;
    result.sdf_A = host.sdf_A;
    const auto branches_B =
        voxel_affine_B_->EvaluateSdfBranches(X_BA_ * host.center_A);
    DRAKE_DEMAND(branches_B.size() == 1);
    VoxelSdfGeometry::SdfSample sample_B_A = branches_B[0].sample;
    sample_B_A.gradient = X_AB_.rotation() * sample_B_A.gradient;
    result.sdf_B_A = MakePressureField(*voxel_affine_B_, sample_B_A);
    return result;
  }

  AffineInput SampleAffine(const Vector3<int>& index) const {
    return QueryAffineB(LoadAffineA(index));
  }

  std::optional<VoxelSdfContactPolygon> ExtractAffine(
      const AffineInput& input) const {
    return CalcVoxelSdfContactPolygon(input.center_A,
                                      voxel_affine_A_->voxel_width(),
                                      input.sdf_A, input.sdf_B_A);
  }

  bool AffinePlaneIntersects(const AffineInput& input) const {
    const double p_A0 = -input.sdf_A.pressure_scale * input.sdf_A.value;
    const double p_B0 = -input.sdf_B_A.pressure_scale * input.sdf_B_A.value;
    const Vector3d grad_p_A =
        -input.sdf_A.pressure_scale * input.sdf_A.gradient;
    const Vector3d grad_p_B_A =
        -input.sdf_B_A.pressure_scale * input.sdf_B_A.gradient;
    const Vector3d grad_F = grad_p_A - grad_p_B_A;
    const double grad_F_norm = grad_F.norm();
    constexpr double kToleranceScale = 64.0;
    const double gradient_tolerance =
        kToleranceScale * std::numeric_limits<double>::epsilon() *
        std::max(grad_p_A.norm(), grad_p_B_A.norm());
    if (grad_F_norm <= gradient_tolerance) return false;
    const double radius = 0.5 * voxel_affine_A_->voxel_width();
    const double pressure_tolerance =
        kToleranceScale * std::numeric_limits<double>::epsilon() *
        std::max({std::abs(p_A0), std::abs(p_B0), radius * grad_p_A.norm(),
                  radius * grad_p_B_A.norm()});
    const double maximum_variation = radius * grad_F.cwiseAbs().sum();
    return std::abs(p_A0 - p_B0) <= maximum_variation + pressure_tolerance;
  }

  int EmitAffine(const std::optional<VoxelSdfContactPolygon>& polygon) const {
    PolyMeshBuilder<double> builder;
    if (!polygon.has_value()) return builder.num_faces();
    std::vector<int> indices;
    indices.reserve(polygon->vertices_A.size());
    for (int v = 0; v < static_cast<int>(polygon->vertices_A.size()); ++v) {
      indices.push_back(
          builder.AddVertex(polygon->vertices_A[v], polygon->pressures[v]));
    }
    builder.AddPolygon(indices, polygon->nhat_BA_A, polygon->grad_p_A);
    return builder.num_faces();
  }

  McHostNodes LoadMcA(const Vector3<int>& index) const {
    McHostNodes result;
    for (int corner = 0; corner < 8; ++corner) {
      const auto& offset = kMcCornerOffsets[corner];
      const int i = index[0] + offset[0];
      const int j = index[1] + offset[1];
      const int k = index[2] + offset[2];
      result.positions_A[corner] = voxel_mc_A_->mc_node_position(i, j, k);
      result.pressures_A[corner] =
          -voxel_mc_A_->pressure_scale() * voxel_mc_A_->mc_node_value(i, j, k);
    }
    return result;
  }

  std::array<MarchingCubesNode, 8> QueryMcB(const McHostNodes& host) const {
    std::array<MarchingCubesNode, 8> result;
    for (int corner = 0; corner < 8; ++corner) {
      const double phi_B =
          voxel_mc_B_->EvaluateSdf(X_BA_ * host.positions_A[corner]).value;
      result[corner] =
          MarchingCubesNode{host.positions_A[corner], host.pressures_A[corner],
                            -voxel_mc_B_->pressure_scale() * phi_B};
    }
    return result;
  }

  std::array<MarchingCubesNode, 8> SampleMc(const Vector3<int>& index) const {
    return QueryMcB(LoadMcA(index));
  }

  int CalcMcCase(const std::array<MarchingCubesNode, 8>& nodes) const {
    int result = 0;
    for (int corner = 0; corner < 8; ++corner) {
      if (nodes[corner].pressure_A - nodes[corner].pressure_B < 0.0) {
        result |= 1 << corner;
      }
    }
    return result;
  }

  int CountMcRawFaces(const std::array<MarchingCubesNode, 8>& nodes) const {
    const std::span<const int> triangles = McTriangles(CalcMcCase(nodes));
    int result = 0;
    for (int offset = 0; triangles[offset] != -1; offset += 3) ++result;
    return result;
  }

  McCase EvaluateMcCube(const Vector3<int>& index) const {
    McCase result;
    result.index = index;
    result.nodes_A = SampleMc(index);
    result.case_index = CalcMcCase(result.nodes_A);
    MarchingCubesContactBuilder builder(voxel_mc_A_->voxel_width());
    builder.AddCube(index, result.nodes_A);
    MarchingCubesMeshData mesh_data = std::move(builder).TakeMeshData();
    result.faces = mesh_data.builder_A.num_faces();
    result.vertices = mesh_data.builder_A.num_vertices();
    result.centroids_A = std::move(mesh_data.face_centroids_A);
    return result;
  }

  int ExtractMc(const McCase& candidate) const {
    MarchingCubesContactBuilder builder(voxel_mc_A_->voxel_width());
    builder.AddCube(candidate.index, candidate.nodes_A);
    return std::move(builder).TakeMeshData().builder_A.num_faces();
  }

  double QueryMcCentroidGradients(const McCase& candidate) const {
    double checksum = 0.0;
    for (const Vector3d& centroid_A : candidate.centroids_A) {
      checksum += (-voxel_mc_A_->pressure_scale() *
                   voxel_mc_A_->EvaluateSdf(centroid_A).gradient)
                      .sum();
      checksum += (X_AB_.rotation() *
                   (-voxel_mc_B_->pressure_scale() *
                    voxel_mc_B_->EvaluateSdf(X_BA_ * centroid_A).gradient))
                      .sum();
    }
    return checksum;
  }

  int FullMc(const Vector3<int>& index, double* gradient_checksum) const {
    McCase candidate = EvaluateMcCube(index);
    *gradient_checksum = QueryMcCentroidGradients(candidate);
    return candidate.faces;
  }

 private:
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

  bool UnitMayOverlapB(const Vector3d& center_A, const Vector3d& half_width_A,
                       const VoxelSdfGeometry& B) const {
    const Vector3d half_width_B = -B.lower_cell_boundary();
    const Vector3d extent_B_A =
        X_AB_.rotation().matrix().cwiseAbs() * half_width_B;
    return ((center_A - X_AB_.translation()).cwiseAbs().array() <=
            (extent_B_A + half_width_A).array())
        .all();
  }

  void SelectTetCases() {
    std::array<bool, 3> found{};
    auto callback = [this, &found](int tet_A, int tet_B) {
      const TetCase candidate = EvaluateTetPair({tet_A, tet_B});
      CandidateKind kind{};
      if (candidate.plane_ready && !candidate.emits()) {
        kind = CandidateKind::kReject;
      } else if (candidate.emits()) {
        const bool rim =
            std::any_of(candidate.pressures.begin(), candidate.pressures.end(),
                        [](double pressure) {
                          return std::abs(pressure) <= 1.0e-10;
                        });
        kind = rim ? CandidateKind::kRim : CandidateKind::kInterior;
      } else {
        return BvttCallbackResult::Continue;
      }
      const int slot = KindIndex(kind);
      if (!found[slot]) {
        tet_cases_[slot] = candidate;
        found[slot] = true;
      }
      return std::all_of(found.begin(), found.end(),
                         [](bool value) {
                           return value;
                         })
                 ? BvttCallbackResult::Terminate
                 : BvttCallbackResult::Continue;
    };
    tet_A_->bvh().Collide(tet_B_->bvh(), X_AB_, callback);
    DRAKE_DEMAND(std::all_of(found.begin(), found.end(), [](bool value) {
      return value;
    }));
  }

  void SelectAffineCases() {
    std::array<bool, 3> found{};
    const Vector3<int>& counts = voxel_affine_A_->cell_counts();
    const Vector3d half_width =
        Vector3d::Constant(0.5 * voxel_affine_A_->voxel_width());
    for (int k = 0; k < counts[2]; ++k) {
      for (int j = 0; j < counts[1]; ++j) {
        for (int i = 0; i < counts[0]; ++i) {
          const Vector3<int> index(i, j, k);
          const Vector3d center_A = voxel_affine_A_->cell_center(i, j, k);
          if (!UnitMayOverlapB(center_A, half_width, *voxel_affine_B_))
            continue;
          AffineCase candidate;
          candidate.index = index;
          candidate.input = SampleAffine(index);
          candidate.polygon = ExtractAffine(candidate.input);
          CandidateKind kind{};
          if (!candidate.polygon.has_value()) {
            kind = CandidateKind::kReject;
          } else {
            const bool rim = std::any_of(candidate.polygon->pressures.begin(),
                                         candidate.polygon->pressures.end(),
                                         [](double pressure) {
                                           return pressure == 0.0;
                                         });
            kind = rim ? CandidateKind::kRim : CandidateKind::kInterior;
          }
          const int slot = KindIndex(kind);
          if (!found[slot]) {
            affine_cases_[slot] = std::move(candidate);
            found[slot] = true;
          }
        }
      }
    }
    DRAKE_DEMAND(std::all_of(found.begin(), found.end(), [](bool value) {
      return value;
    }));
  }

  void SelectMcCases() {
    std::array<bool, 3> found{};
    const Vector3<int> counts = voxel_mc_A_->mc_cube_counts();
    for (int k = 0; k < counts[2]; ++k) {
      for (int j = 0; j < counts[1]; ++j) {
        for (int i = 0; i < counts[0]; ++i) {
          const Vector3<int> index(i, j, k);
          const Vector3d lower_A = voxel_mc_A_->mc_node_position(i, j, k);
          const Vector3d upper_A =
              voxel_mc_A_->mc_node_position(i + 1, j + 1, k + 1);
          const Vector3d center_A = 0.5 * (lower_A + upper_A);
          const Vector3d half_width = 0.5 * (upper_A - lower_A).cwiseAbs();
          if (!UnitMayOverlapB(center_A, half_width, *voxel_mc_B_)) continue;
          McCase candidate = EvaluateMcCube(index);
          CandidateKind kind{};
          if (candidate.faces == 0) {
            kind = CandidateKind::kReject;
          } else {
            MarchingCubesContactBuilder builder(voxel_mc_A_->voxel_width());
            builder.AddCube(index, candidate.nodes_A);
            MarchingCubesMeshData mesh_data = std::move(builder).TakeMeshData();
            auto [mesh, field] = mesh_data.builder_A.MakeMeshAndField();
            static_cast<void>(mesh);
            const bool rim =
                std::any_of(field->values().begin(), field->values().end(),
                            [](double pressure) {
                              return pressure == 0.0;
                            });
            kind = rim ? CandidateKind::kRim : CandidateKind::kInterior;
          }
          const int slot = KindIndex(kind);
          if (!found[slot]) {
            mc_cases_[slot] = std::move(candidate);
            found[slot] = true;
          }
        }
      }
    }
    DRAKE_DEMAND(std::all_of(found.begin(), found.end(), [](bool value) {
      return value;
    }));
  }

  RigidTransformd X_AB_{};
  RigidTransformd X_BA_{};
  std::unique_ptr<CompliantMesh> tet_A_;
  std::unique_ptr<CompliantMesh> tet_B_;
  std::unique_ptr<VoxelSdfGeometry> voxel_affine_A_;
  std::unique_ptr<VoxelSdfGeometry> voxel_affine_B_;
  std::unique_ptr<VoxelSdfGeometry> voxel_mc_A_;
  std::unique_ptr<VoxelSdfGeometry> voxel_mc_B_;
  std::array<TetCase, 3> tet_cases_;
  std::array<AffineCase, 3> affine_cases_;
  std::array<McCase, 3> mc_cases_;
};

class LocalKernelBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State&) final {
    static_cast<void>(LocalKernelData::get());
  }

  void Record(benchmark::State* state, CandidateKind kind, int faces,
              int vertices) const {
    state->SetItemsProcessed(state->iterations());
    state->SetLabel(std::string(KindName(kind)));
    state->counters["faces"] = faces;
    state->counters["vertices"] = vertices;
  }
};

BENCHMARK_DEFINE_F(LocalKernelBenchmark, TetFieldAndPlane)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const TetCase& candidate = data.tet_case(kind);
  for (auto _ : state) {
    Plane<double> plane(Vector3d::UnitZ(), Vector3d::Zero());
    bool accepted = data.CalcTetFieldAndPlane(candidate.pair, &plane);
    benchmark::DoNotOptimize(accepted);
    benchmark::DoNotOptimize(plane);
  }
  Record(&state, kind, candidate.emits() ? 1 : 0, candidate.vertices_A.size());
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, TetIntersectClip)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const TetCase& candidate = data.tet_case(kind);
  for (auto _ : state) {
    auto result = data.IntersectTetPair(candidate.pair, candidate.plane_A);
    benchmark::DoNotOptimize(result);
  }
  Record(&state, kind, candidate.emits() ? 1 : 0, candidate.vertices_A.size());
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, TetEmit)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const TetCase& candidate = data.tet_case(kind);
  for (auto _ : state) benchmark::DoNotOptimize(data.EmitTet(candidate));
  Record(&state, kind, candidate.emits() ? 1 : 0, candidate.vertices_A.size());
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, TetFullCandidate)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const TetCase& candidate = data.tet_case(kind);
  for (auto _ : state) {
    const TetCase result = data.EvaluateTetPair(candidate.pair);
    benchmark::DoNotOptimize(data.EmitTet(result));
  }
  Record(&state, kind, candidate.emits() ? 1 : 0, candidate.vertices_A.size());
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineLoadA)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  for (auto _ : state)
    benchmark::DoNotOptimize(data.LoadAffineA(candidate.index));
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineQueryB)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  const AffineHostData host = data.LoadAffineA(candidate.index);
  for (auto _ : state) benchmark::DoNotOptimize(data.QueryAffineB(host));
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineClassify)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  for (auto _ : state) {
    benchmark::DoNotOptimize(data.AffinePlaneIntersects(candidate.input));
  }
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineExtractClip)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  for (auto _ : state)
    benchmark::DoNotOptimize(data.ExtractAffine(candidate.input));
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineEmit)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  for (auto _ : state)
    benchmark::DoNotOptimize(data.EmitAffine(candidate.polygon));
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, AffineFullCandidate)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const AffineCase& candidate = data.affine_case(kind);
  for (auto _ : state) {
    const AffineInput input = data.SampleAffine(candidate.index);
    benchmark::DoNotOptimize(data.EmitAffine(data.ExtractAffine(input)));
  }
  Record(
      &state, kind, candidate.polygon.has_value() ? 1 : 0,
      candidate.polygon.has_value() ? candidate.polygon->vertices_A.size() : 0);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McLoadA)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  for (auto _ : state) benchmark::DoNotOptimize(data.LoadMcA(candidate.index));
  Record(&state, kind, candidate.faces, candidate.vertices);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McQueryB)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  const McHostNodes host = data.LoadMcA(candidate.index);
  for (auto _ : state) benchmark::DoNotOptimize(data.QueryMcB(host));
  Record(&state, kind, candidate.faces, candidate.vertices);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McLookup)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  for (auto _ : state)
    benchmark::DoNotOptimize(data.CountMcRawFaces(candidate.nodes_A));
  Record(&state, kind, candidate.faces, candidate.vertices);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McExtractClipEmit)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  for (auto _ : state) benchmark::DoNotOptimize(data.ExtractMc(candidate));
  Record(&state, kind, candidate.faces, candidate.vertices);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McCentroidGradients)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  for (auto _ : state)
    benchmark::DoNotOptimize(data.QueryMcCentroidGradients(candidate));
  Record(&state, kind, candidate.faces, candidate.vertices);
}

BENCHMARK_DEFINE_F(LocalKernelBenchmark, McFullCandidate)
// NOLINTNEXTLINE(runtime/references)
(benchmark::State& state) {
  const CandidateKind kind = ReadKind(state);
  const auto& data = LocalKernelData::get();
  const McCase& candidate = data.mc_case(kind);
  for (auto _ : state) {
    double gradient_checksum{};
    benchmark::DoNotOptimize(data.FullMc(candidate.index, &gradient_checksum));
    benchmark::DoNotOptimize(gradient_checksum);
  }
  Record(&state, kind, candidate.faces, candidate.vertices);
}

#define REGISTER_LOCAL_KERNEL_BENCHMARK(name)      \
  BENCHMARK_REGISTER_F(LocalKernelBenchmark, name) \
      ->DenseRange(0, 2)                           \
      ->ArgName("kind")                            \
      ->Unit(benchmark::kNanosecond)               \
      ->MinTime(0.1)                               \
      ->MinWarmUpTime(0.02)

REGISTER_LOCAL_KERNEL_BENCHMARK(TetFieldAndPlane);
REGISTER_LOCAL_KERNEL_BENCHMARK(TetIntersectClip);
REGISTER_LOCAL_KERNEL_BENCHMARK(TetEmit);
REGISTER_LOCAL_KERNEL_BENCHMARK(TetFullCandidate);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineLoadA);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineQueryB);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineClassify);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineExtractClip);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineEmit);
REGISTER_LOCAL_KERNEL_BENCHMARK(AffineFullCandidate);
REGISTER_LOCAL_KERNEL_BENCHMARK(McLoadA);
REGISTER_LOCAL_KERNEL_BENCHMARK(McQueryB);
REGISTER_LOCAL_KERNEL_BENCHMARK(McLookup);
REGISTER_LOCAL_KERNEL_BENCHMARK(McExtractClipEmit);
REGISTER_LOCAL_KERNEL_BENCHMARK(McCentroidGradients);
REGISTER_LOCAL_KERNEL_BENCHMARK(McFullCandidate);

#undef REGISTER_LOCAL_KERNEL_BENCHMARK

}  // namespace
}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
