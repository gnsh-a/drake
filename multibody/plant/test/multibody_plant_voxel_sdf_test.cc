#include <cmath>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace {

using Eigen::Vector3d;
using geometry::ContactSurface;
using geometry::GeometryId;
using geometry::HydroelasticContactRepresentation;
using geometry::ProximityProperties;
using math::RigidTransformd;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

class VoxelSdfSapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DiagramBuilder<double> builder;
    plant_ = &AddMultibodyPlantSceneGraph(&builder, kTimeStep).plant;

    plant_->set_contact_model(ContactModel::kHydroelastic);
    plant_->set_discrete_contact_approximation(
        DiscreteContactApproximation::kSap);
    plant_->set_contact_surface_representation(
        HydroelasticContactRepresentation::kPolygon);
    // These tests inspect the response at the current pose before advancing.
    plant_->SetUseSampledOutputPorts(false);
    plant_->mutable_gravity_field().set_gravity_vector(Vector3d::Zero());

    const geometry::Box box = geometry::Box::MakeCube(kBoxWidth);
    const ProximityProperties properties = MakeProperties();
    anchored_geometry_id_ = plant_->RegisterCollisionGeometry(
        plant_->world_body(), RigidTransformd(), box, "anchored_box",
        properties);

    const SpatialInertia<double> M_BBcm =
        SpatialInertia<double>::SolidBoxWithMass(kMass, kBoxWidth, kBoxWidth,
                                                 kBoxWidth);
    dynamic_body_ = &plant_->AddRigidBody("dynamic_box", M_BBcm);
    dynamic_geometry_id_ = plant_->RegisterCollisionGeometry(
        *dynamic_body_, RigidTransformd(), box, "dynamic_box", properties);
    ASSERT_LT(anchored_geometry_id_, dynamic_geometry_id_);

    plant_->Finalize();
    diagram_ = builder.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get());
    SetDynamicPose(kOverlappingZ);
  }

  ProximityProperties MakeProperties() const {
    ProximityProperties properties;
    geometry::AddCompliantHydroelasticVoxelSdfProperties(
        kVoxelWidth, kHydroelasticModulus, &properties);
    geometry::AddContactMaterial(
        /* dissipation = */ 0.0, /* point_stiffness = */ {},
        CoulombFriction<double>(kFriction, kFriction), &properties);
    return properties;
  }

  void SetDynamicPose(double z) {
    plant_->SetFreeBodyPose(plant_context_, *dynamic_body_,
                            RigidTransformd(Vector3d(0.0, 0.0, z)));
    plant_->SetFreeBodySpatialVelocity(plant_context_, *dynamic_body_,
                                       SpatialVelocity<double>::Zero());
  }

  const ContactResults<double>& EvalContactResults(
      const Context<double>& plant_context) const {
    return plant_->get_contact_results_output_port()
        .Eval<ContactResults<double>>(plant_context);
  }

  void ExpectValidVoxelSurface(const ContactSurface<double>& surface) const {
    EXPECT_EQ(surface.id_M(), anchored_geometry_id_);
    EXPECT_EQ(surface.id_N(), dynamic_geometry_id_);
    EXPECT_FALSE(surface.is_triangle());
    ASSERT_GT(surface.num_faces(), 0);
    ASSERT_GT(surface.num_vertices(), 0);
    EXPECT_TRUE(std::isfinite(surface.total_area()));
    EXPECT_GT(surface.total_area(), 0.0);
    EXPECT_TRUE(surface.centroid().allFinite());
    ASSERT_TRUE(surface.HasGradE_M());
    ASSERT_TRUE(surface.HasGradE_N());

    bool has_positive_pressure = false;
    for (int face = 0; face < surface.num_faces(); ++face) {
      const double area = surface.area(face);
      const Vector3d& normal = surface.face_normal(face);
      const Vector3d centroid = surface.centroid(face);
      EXPECT_TRUE(std::isfinite(area));
      EXPECT_GT(area, 0.0);
      EXPECT_TRUE(normal.allFinite());
      EXPECT_NEAR(normal.norm(), 1.0, 1e-14);
      EXPECT_TRUE(centroid.allFinite());
      EXPECT_TRUE(surface.EvaluateGradE_M_W(face).allFinite());
      EXPECT_TRUE(surface.EvaluateGradE_N_W(face).allFinite());
      const double pressure =
          surface.poly_e_MN().EvaluateCartesian(face, centroid);
      EXPECT_TRUE(std::isfinite(pressure));
      EXPECT_GE(pressure, 0.0);
      has_positive_pressure = has_positive_pressure || pressure > 0.0;
    }
    EXPECT_TRUE(has_positive_pressure);
  }

  static constexpr double kTimeStep = 1e-3;
  static constexpr double kBoxWidth = 1.0;
  static constexpr double kVoxelWidth = 0.25;
  static constexpr double kHydroelasticModulus = 1e5;
  static constexpr double kFriction = 0.5;
  static constexpr double kMass = 10.0;
  static constexpr double kOverlappingZ = 0.9;
  static constexpr double kSeparatedZ = 2.0;

  MultibodyPlant<double>* plant_{nullptr};
  const RigidBody<double>* dynamic_body_{nullptr};
  GeometryId anchored_geometry_id_;
  GeometryId dynamic_geometry_id_;
  std::unique_ptr<Diagram<double>> diagram_;
  std::unique_ptr<Context<double>> diagram_context_;
  Context<double>* plant_context_{nullptr};
};

// TODO(gnsh-a): Add an aligned-Box analytical oracle for interface
// position, area, pressure, integrated force and torque, and one-step SAP
// response. This test currently verifies integration, finiteness, and signs.
TEST_F(VoxelSdfSapTest, SapConsumesVoxelSurfaceAndAdvances) {
  const ContactResults<double>& initial_results =
      EvalContactResults(*plant_context_);
  EXPECT_EQ(initial_results.num_point_pair_contacts(), 0);
  ASSERT_EQ(initial_results.num_hydroelastic_contacts(), 1);
  const HydroelasticContactInfo<double>& initial_contact =
      initial_results.hydroelastic_contact_info(0);
  ExpectValidVoxelSurface(initial_contact.contact_surface());

  const SpatialForce<double>& F_Ac_W = initial_contact.F_Ac_W();
  EXPECT_TRUE(F_Ac_W.get_coeffs().allFinite());
  // Geometry M is the anchored lower Box, so the repulsive force on M points
  // downward and the equal-and-opposite force on the dynamic Box points up.
  EXPECT_LT(F_Ac_W.translational().z(), 0.0);

  Simulator<double> simulator(*diagram_, std::move(diagram_context_));
  EXPECT_NO_THROW(simulator.AdvanceTo(kTimeStep));
  const Context<double>& updated_plant_context =
      plant_->GetMyContextFromRoot(simulator.get_context());
  EXPECT_TRUE(
      plant_->GetPositionsAndVelocities(updated_plant_context).allFinite());
  const SpatialVelocity<double> V_WB =
      dynamic_body_->EvalSpatialVelocityInWorld(updated_plant_context);
  EXPECT_TRUE(V_WB.get_coeffs().allFinite());
  EXPECT_GT(V_WB.translational().z(), 0.0);

  const ContactResults<double>& updated_results =
      EvalContactResults(updated_plant_context);
  for (int i = 0; i < updated_results.num_hydroelastic_contacts(); ++i) {
    EXPECT_TRUE(updated_results.hydroelastic_contact_info(i)
                    .F_Ac_W()
                    .get_coeffs()
                    .allFinite());
  }
}

TEST_F(VoxelSdfSapTest, SeparatedPoseHasNoContact) {
  SetDynamicPose(kSeparatedZ);
  const ContactResults<double>& results = EvalContactResults(*plant_context_);
  EXPECT_EQ(results.num_point_pair_contacts(), 0);
  EXPECT_EQ(results.num_hydroelastic_contacts(), 0);
}

GTEST_TEST(VoxelSdfMarchingCubesSapTest,
           SapConsumesTriangleSurfaceAndAdvances) {
  constexpr double kTimeStep = 1e-3;
  constexpr double kRadius = 1.0;
  constexpr double kSeparation = 1.85;
  constexpr double kVoxelWidth = 0.25;
  constexpr double kHydroelasticModulus = 1e5;
  constexpr double kFriction = 0.5;
  constexpr double kMass = 10.0;

  DiagramBuilder<double> builder;
  MultibodyPlant<double>& plant =
      AddMultibodyPlantSceneGraph(&builder, kTimeStep).plant;
  plant.set_contact_model(ContactModel::kHydroelastic);
  plant.set_discrete_contact_approximation(DiscreteContactApproximation::kSap);
  plant.set_contact_surface_representation(
      HydroelasticContactRepresentation::kTriangle);
  plant.SetUseSampledOutputPorts(false);
  plant.mutable_gravity_field().set_gravity_vector(Vector3d::Zero());

  ProximityProperties properties;
  geometry::AddCompliantHydroelasticVoxelSdfProperties(
      kVoxelWidth, kHydroelasticModulus,
      geometry::VoxelSdfEvaluationMode::kPrimitiveSdf,
      geometry::VoxelSdfExtractionMethod::kMarchingCubes, &properties);
  geometry::AddContactMaterial(
      /* dissipation = */ 0.0, /* point_stiffness = */ {},
      CoulombFriction<double>(kFriction, kFriction), &properties);

  const geometry::Sphere sphere(kRadius);
  const GeometryId anchored_geometry_id =
      plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(),
                                      sphere, "anchored_sphere", properties);
  const SpatialInertia<double> M_BBcm =
      SpatialInertia<double>::SolidSphereWithMass(kMass, kRadius);
  const RigidBody<double>& dynamic_body =
      plant.AddRigidBody("dynamic_sphere", M_BBcm);
  const GeometryId dynamic_geometry_id = plant.RegisterCollisionGeometry(
      dynamic_body, RigidTransformd(), sphere, "dynamic_sphere", properties);
  ASSERT_LT(anchored_geometry_id, dynamic_geometry_id);

  plant.Finalize();
  std::unique_ptr<Diagram<double>> diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetFreeBodyPose(&plant_context, dynamic_body,
                        RigidTransformd(Vector3d(0.0, 0.0, kSeparation)));
  plant.SetFreeBodySpatialVelocity(&plant_context, dynamic_body,
                                   SpatialVelocity<double>::Zero());

  const ContactResults<double>& initial_results =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          plant_context);
  EXPECT_EQ(initial_results.num_point_pair_contacts(), 0);
  ASSERT_EQ(initial_results.num_hydroelastic_contacts(), 1);
  const HydroelasticContactInfo<double>& initial_contact =
      initial_results.hydroelastic_contact_info(0);
  const ContactSurface<double>& surface = initial_contact.contact_surface();
  EXPECT_EQ(surface.id_M(), anchored_geometry_id);
  EXPECT_EQ(surface.id_N(), dynamic_geometry_id);
  ASSERT_TRUE(surface.is_triangle());
  ASSERT_GT(surface.num_faces(), 0);
  ASSERT_GT(surface.num_vertices(), 0);
  EXPECT_TRUE(std::isfinite(surface.total_area()));
  EXPECT_GT(surface.total_area(), 0.0);
  ASSERT_TRUE(surface.HasGradE_M());
  ASSERT_TRUE(surface.HasGradE_N());

  bool has_positive_pressure = false;
  for (int vertex = 0; vertex < surface.num_vertices(); ++vertex) {
    EXPECT_TRUE(surface.tri_mesh_W().vertex(vertex).allFinite());
    const double pressure = surface.tri_e_MN().EvaluateAtVertex(vertex);
    EXPECT_TRUE(std::isfinite(pressure));
    EXPECT_GE(pressure, 0.0);
    has_positive_pressure = has_positive_pressure || pressure > 0.0;
  }
  EXPECT_TRUE(has_positive_pressure);
  for (int face = 0; face < surface.num_faces(); ++face) {
    EXPECT_TRUE(surface.EvaluateGradE_M_W(face).allFinite());
    EXPECT_TRUE(surface.EvaluateGradE_N_W(face).allFinite());
  }

  const SpatialForce<double>& F_Ac_W = initial_contact.F_Ac_W();
  EXPECT_TRUE(F_Ac_W.get_coeffs().allFinite());
  EXPECT_LT(F_Ac_W.translational().z(), 0.0);

  Simulator<double> simulator(*diagram, std::move(diagram_context));
  EXPECT_NO_THROW(simulator.AdvanceTo(kTimeStep));
  const Context<double>& updated_plant_context =
      plant.GetMyContextFromRoot(simulator.get_context());
  EXPECT_TRUE(
      plant.GetPositionsAndVelocities(updated_plant_context).allFinite());
  const SpatialVelocity<double> V_WB =
      dynamic_body.EvalSpatialVelocityInWorld(updated_plant_context);
  EXPECT_TRUE(V_WB.get_coeffs().allFinite());
  EXPECT_GT(V_WB.translational().z(), 0.0);

  const ContactResults<double>& updated_results =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          updated_plant_context);
  for (int i = 0; i < updated_results.num_hydroelastic_contacts(); ++i) {
    EXPECT_TRUE(updated_results.hydroelastic_contact_info(i)
                    .F_Ac_W()
                    .get_coeffs()
                    .allFinite());
  }
}

}  // namespace
}  // namespace multibody
}  // namespace drake
