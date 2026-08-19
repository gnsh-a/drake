/* Tests MultibodyPlant::EvalSapSolverStatistics.

 The statistics travel a long way to reach a caller: SapSolver fills them in,
 SapDriver copies them into the contact-solver results, the plant caches those
 results, and the accessor reads them back out under whichever output-port mode
 the plant is in. Every link in that chain fails the same way -- a
 default-constructed SapStatistics, whose counters are all zero and which is
 indistinguishable from a solve that converged immediately. These tests
 therefore check that the statistics are populated, not merely present. */

#include <memory>
#include <optional>

#include <gtest/gtest.h>

#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace {

using contact_solvers::internal::SapStatistics;
using geometry::HalfSpace;
using geometry::SceneGraph;
using geometry::Sphere;
using math::RigidTransformd;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::Simulator;

constexpr double kRadius = 0.05;
constexpr double kPenetration = 1.0e-3;

/* A single sphere resting on a half space, started already in contact so the
 very first discrete update has a contact problem to solve. */
struct Scene {
  std::unique_ptr<Diagram<double>> diagram;
  MultibodyPlant<double>* plant{};
};

Scene MakeScene(double time_step,
                bool use_sampled_output_ports = true) {
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, time_step);
  if (time_step > 0) {
    plant.set_discrete_contact_approximation(
        DiscreteContactApproximation::kSap);
  }
  const CoulombFriction<double> friction(1.0, 1.0);
  plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(),
                                  HalfSpace(), "ground", friction);
  const RigidBody<double>& ball = plant.AddRigidBody(
      "ball", SpatialInertia<double>::SolidSphereWithMass(1.0, kRadius));
  plant.RegisterCollisionGeometry(ball, RigidTransformd(), Sphere(kRadius),
                                  "ball", friction);
  /* The output-port mode is a pre-finalize setting. */
  if (time_step > 0) {
    plant.SetUseSampledOutputPorts(use_sampled_output_ports);
  }
  plant.Finalize();
  Scene scene;
  scene.plant = &plant;
  scene.diagram = builder.Build();
  return scene;
}

void PlaceBallInContact(const Scene& scene, Context<double>* root_context) {
  Context<double>& plant_context =
      scene.plant->GetMyMutableContextFromRoot(root_context);
  scene.plant->SetFreeBodyPose(
      &plant_context, scene.plant->GetBodyByName("ball"),
      RigidTransformd(Vector3<double>(0.0, 0.0, kRadius - kPenetration)));
}

/* With sampled output ports -- the default for a discrete plant -- the
 statistics describe the most recent completed update, so there are none before
 the first one. */
GTEST_TEST(SapSolverStatisticsTest, SampledModeReportsTheCompletedStep) {
  const Scene scene = MakeScene(1.0e-3);
  Simulator<double> simulator(*scene.diagram);
  PlaceBallInContact(scene, &simulator.get_mutable_context());
  const Context<double>& plant_context =
      scene.plant->GetMyContextFromRoot(simulator.get_context());

  EXPECT_FALSE(
      scene.plant->EvalSapSolverStatistics(plant_context).has_value());

  simulator.AdvanceTo(0.01);
  const std::optional<SapStatistics> statistics =
      scene.plant->EvalSapSolverStatistics(plant_context);
  ASSERT_TRUE(statistics.has_value());
  /* A live solve takes at least one Newton iteration and stops on one of the
   two criteria. Zeros here mean the statistics were default-constructed
   somewhere along the chain rather than filled in by the solver. */
  EXPECT_GT(statistics->num_iters, 0);
  EXPECT_TRUE(statistics->optimality_criterion_reached ||
              statistics->cost_criterion_reached);
  /* The per-iteration histories are documented as size num_iters + 1. */
  EXPECT_EQ(std::ssize(statistics->momentum_residual),
            statistics->num_iters + 1);
  EXPECT_EQ(std::ssize(statistics->momentum_scale), statistics->num_iters + 1);
}

/* Live output ports keep no step memory, so the accessor falls back to the
 contact-solver results cached at the context. Those describe the update that
 advances *from* this state, which is why they are available before any step
 has been taken -- the opposite of the sampled convention above. */
GTEST_TEST(SapSolverStatisticsTest, LiveModeReportsTheUpcomingStep) {
  const Scene scene = MakeScene(1.0e-3, /* use_sampled_output_ports = */ false);
  Simulator<double> simulator(*scene.diagram);
  PlaceBallInContact(scene, &simulator.get_mutable_context());
  const Context<double>& plant_context =
      scene.plant->GetMyContextFromRoot(simulator.get_context());

  const std::optional<SapStatistics> statistics =
      scene.plant->EvalSapSolverStatistics(plant_context);
  ASSERT_TRUE(statistics.has_value());
  EXPECT_GT(statistics->num_iters, 0);
  EXPECT_TRUE(statistics->optimality_criterion_reached ||
              statistics->cost_criterion_reached);
}

/* A continuous plant runs no discrete contact solve at all. */
GTEST_TEST(SapSolverStatisticsTest, ContinuousPlantHasNoStatistics) {
  const Scene scene = MakeScene(0.0);
  auto context = scene.diagram->CreateDefaultContext();
  PlaceBallInContact(scene, context.get());
  EXPECT_FALSE(scene.plant
                   ->EvalSapSolverStatistics(
                       scene.plant->GetMyContextFromRoot(*context))
                   .has_value());
}

}  // namespace
}  // namespace multibody
}  // namespace drake
