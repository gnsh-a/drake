"""
Side-by-side validation of standalone implementation against Drake.

Requires: conda activate drake (pydrake installed)
Run from: test_scripts/

Author: Ganesh Arivoli <arivoli@wisc.edu>
"""

import sys
import numpy as np

# Standalone modules
import dynamics
import contact
import solver
from main import (
    MASS, RADIUS, GRAVITY, INCLINE_DEG, MU_STATIC,
    DT, PENETRATION_ALLOWANCE,
)

# pydrake
from pydrake.multibody.plant import (
    MultibodyPlant, AddMultibodyPlantSceneGraph,
    CoulombFriction, MultibodyPlantConfig, AddMultibodyPlant,
    DiscreteContactApproximation,
)
from pydrake.multibody.tree import (
    SpatialInertia, UnitInertia, JacobianWrtVariable,
)
from pydrake.geometry import HalfSpace, Sphere, ProximityProperties
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.math import RigidTransform, RotationMatrix

THETA = np.radians(INCLINE_DEG)


def check(name, mine, drake, atol=1e-12):
    """Assert two values are close, print result."""
    mine = np.asarray(mine, dtype=float)
    drake = np.asarray(drake, dtype=float)
    diff = np.max(np.abs(mine - drake))
    ok = diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max diff = {diff:.2e} (tol={atol:.0e})")
    if not ok:
        print(f"    mine:  {mine}")
        print(f"    drake: {drake}")
    return ok


def setup_drake():
    """Create Drake plant + scene_graph matching the standalone demo."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=DT)

    # Inclined plane (half-space on world body)
    R_WH = RotationMatrix.MakeYRotation(THETA)
    X_WH = RigidTransform(R_WH, [0, 0, 0])
    friction_plane = CoulombFriction(MU_STATIC, MU_STATIC)
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WH, HalfSpace(),
        "InclinedPlaneCollisionGeometry", friction_plane)

    # Sphere (free body)
    I_scalar = 2.0 / 5.0 * MASS * RADIUS**2
    M_BBo = SpatialInertia(
        MASS, [0, 0, 0],
        UnitInertia(I_scalar / MASS, I_scalar / MASS, I_scalar / MASS))
    body = plant.AddRigidBody("BodyB", M_BBo)
    friction_sphere = CoulombFriction(MU_STATIC, MU_STATIC)
    plant.RegisterCollisionGeometry(
        body, RigidTransform(), Sphere(RADIUS),
        "SphereB_CollisionGeometry", friction_sphere)

    # Gravity
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -GRAVITY])

    # Contact settings
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)

    plant.Finalize()
    plant.set_penetration_allowance(PENETRATION_ALLOWANCE)

    diagram = builder.Build()
    return diagram, plant, scene_graph


def validate_dynamics(plant, plant_context, q, v):
    """Phase 1: Validate mass matrix, bias, gravity, v_star."""
    print("\nPhase 1: Dynamics")
    plant.SetPositions(plant_context, q)
    plant.SetVelocities(plant_context, v)

    I_B = (2.0 / 5.0 * MASS * RADIUS**2) * np.eye(3)

    M_mine = dynamics.mass_matrix(q, I_B, MASS)
    M_drake = plant.CalcMassMatrix(plant_context)
    ok = check("Mass matrix", M_mine, M_drake)

    k_mine = dynamics.bias_forces(q, v, I_B)
    k_drake = plant.CalcBiasTerm(plant_context)
    ok &= check("Bias term", k_mine, k_drake)

    tau_g_mine = dynamics.gravity_forces(MASS, GRAVITY)
    tau_g_drake = plant.CalcGravityGeneralizedForces(plant_context)
    ok &= check("Gravity forces", tau_g_mine, tau_g_drake)

    # v_star (not directly exposed, compute from Drake quantities)
    v_star_mine = dynamics.free_motion_velocity(M_mine, v, tau_g_mine, k_mine, DT)
    v_star_ref = v + DT * np.linalg.solve(M_drake, tau_g_drake - k_drake)
    ok &= check("Free-motion velocity", v_star_mine, v_star_ref)

    return ok


def validate_collision(plant, plant_context, scene_graph, diagram_context, q):
    """Phase 2: Validate collision detection."""
    print("\nPhase 2: Collision Detection")
    plant.SetPositions(plant_context, q)

    R_WH = np.array([
        [ np.cos(THETA), 0.0, np.sin(THETA)],
        [           0.0, 1.0,           0.0],
        [-np.sin(THETA), 0.0, np.cos(THETA)],
    ])
    p_WH = np.zeros(3)

    result_mine = contact.sphere_halfspace(q, RADIUS, R_WH, p_WH)

    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    query_object = scene_graph.get_query_output_port().Eval(sg_context)
    pairs = query_object.ComputePointPairPenetration()

    if result_mine is None and len(pairs) == 0:
        print("  [PASS] No contact (both agree)")
        return True

    if result_mine is None or len(pairs) == 0:
        print("  [FAIL] Contact detection disagreement")
        return False

    pair = pairs[0]
    ok = check("Depth", result_mine["depth"], pair.depth)
    ok &= check("p_WCa", result_mine["p_WCa"], pair.p_WCa)
    ok &= check("p_WCb", result_mine["p_WCb"], pair.p_WCb)
    ok &= check("nhat_BA_W", result_mine["nhat_BA_W"], pair.nhat_BA_W)
    return ok


def validate_jacobian(plant, plant_context, q, v):
    """Phase 3: Validate contact Jacobian."""
    print("\nPhase 3: Contact Jacobian")
    plant.SetPositions(plant_context, q)
    plant.SetVelocities(plant_context, v)

    R_WH = np.array([
        [ np.cos(THETA), 0.0, np.sin(THETA)],
        [           0.0, 1.0,           0.0],
        [-np.sin(THETA), 0.0, np.cos(THETA)],
    ])
    p_WH = np.zeros(3)

    result = contact.sphere_halfspace(q, RADIUS, R_WH, p_WH)
    if result is None:
        print("  [SKIP] No contact at this state")
        return True

    nhat_BA_W = result["nhat_BA_W"]
    geometry_k, _ = contact.estimate_contact_parameters(
        MASS, GRAVITY, PENETRATION_ALLOWANCE)
    p_WC = contact.hertz_contact_point(
        result["p_WCa"], result["p_WCb"], geometry_k, geometry_k)
    R_WC = contact.contact_frame(nhat_BA_W)

    J_mine = contact.contact_jacobian(q, p_WC, R_WC)

    # Drake Jacobian
    # p_BoBi_B must be in the body frame, not world frame
    body = plant.GetBodyByName("BodyB")
    X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
    p_WB = X_WB.translation()
    R_WB = X_WB.rotation().matrix()
    p_BC_B = R_WB.T @ (p_WC - p_WB)

    Jv_WBc_drake = plant.CalcJacobianTranslationalVelocity(
        plant_context, JacobianWrtVariable.kV,
        body.body_frame(),
        p_BC_B.reshape(3, 1),
        plant.world_frame(), plant.world_frame())
    J_drake = R_WC.T @ Jv_WBc_drake

    return check("Contact Jacobian", J_mine, J_drake)


def validate_single_step(diagram, plant, scene_graph, q0, v0):
    """Phase 4: Single time step end-to-end comparison."""
    print("\nPhase 4: Single Step End-to-End")

    # Drake step
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT)
    q_drake = plant.GetPositions(plant_context)
    v_drake = plant.GetVelocities(plant_context)

    # Standalone step
    I_B = (2.0 / 5.0 * MASS * RADIUS**2) * np.eye(3)
    R_WH = np.array([
        [ np.cos(THETA), 0.0, np.sin(THETA)],
        [           0.0, 1.0,           0.0],
        [-np.sin(THETA), 0.0, np.cos(THETA)],
    ])
    p_WH = np.zeros(3)
    tau_g = dynamics.gravity_forces(MASS, GRAVITY)
    geometry_k, dissipation = contact.estimate_contact_parameters(
        MASS, GRAVITY, PENETRATION_ALLOWANCE)
    k_combined, _, mu_combined, tau_combined = contact.combine_contact_properties(
        geometry_k, geometry_k, dissipation, dissipation, MU_STATIC, MU_STATIC)

    q, v = q0.copy(), v0.copy()
    M = dynamics.mass_matrix(q, I_B, MASS)
    k_bias = dynamics.bias_forces(q, v, I_B)
    v_star = dynamics.free_motion_velocity(M, v, tau_g, k_bias, DT)

    contact_data = contact.sphere_halfspace(q, RADIUS, R_WH, p_WH)
    if contact_data is not None:
        p_WC = contact.hertz_contact_point(
            contact_data["p_WCa"], contact_data["p_WCb"], geometry_k, geometry_k)
        R_WC = contact.contact_frame(contact_data["nhat_BA_W"])
        J = contact.contact_jacobian(q, p_WC, R_WC)
        v_next = solver.sap_solve(M, v_star, J, contact_data["phi0"],
                                   k_combined, tau_combined, mu_combined, v, DT)
    else:
        v_next = v_star

    q_mine = dynamics.integrate_state(q, v_next, DT)

    ok = check("v_next", v_next, v_drake, atol=1e-8)
    ok &= check("q_next", q_mine, q_drake, atol=1e-8)
    return ok


def validate_no_contact(diagram, plant):
    """Phase 5: No-contact case — v_next should equal v_star."""
    print("\nPhase 5: No-Contact Free Fall")

    q_high = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    v0 = np.array([0.1, 0.2, 0.3, 0.5, -0.3, 0.0])

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, q_high)
    plant.SetVelocities(plant_context, v0)
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT)
    v_drake = plant.GetVelocities(plant_context)

    I_B = (2.0 / 5.0 * MASS * RADIUS**2) * np.eye(3)
    M = dynamics.mass_matrix(q_high, I_B, MASS)
    k_bias = dynamics.bias_forces(q_high, v0, I_B)
    tau_g = dynamics.gravity_forces(MASS, GRAVITY)
    v_star = dynamics.free_motion_velocity(M, v0, tau_g, k_bias, DT)

    return check("No-contact v_next = v_star", v_star, v_drake, atol=1e-10)


def validate_multi_step(diagram, plant, scene_graph, num_steps=200):
    """Phase 6: Multi-step trajectory comparison."""
    print(f"\nPhase 6: Multi-Step Trajectory ({num_steps} steps)")

    q0 = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.2])
    v0 = np.zeros(6)

    # Drake simulation
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
    sim = Simulator(diagram, context)
    sim.Initialize()

    # Standalone
    I_B = (2.0 / 5.0 * MASS * RADIUS**2) * np.eye(3)
    R_WH = np.array([
        [ np.cos(THETA), 0.0, np.sin(THETA)],
        [           0.0, 1.0,           0.0],
        [-np.sin(THETA), 0.0, np.cos(THETA)],
    ])
    p_WH = np.zeros(3)
    tau_g = dynamics.gravity_forces(MASS, GRAVITY)
    geometry_k, dissipation = contact.estimate_contact_parameters(
        MASS, GRAVITY, PENETRATION_ALLOWANCE)
    k_combined, _, mu_combined, tau_combined = contact.combine_contact_properties(
        geometry_k, geometry_k, dissipation, dissipation, MU_STATIC, MU_STATIC)

    q, v = q0.copy(), v0.copy()
    max_q_diff = 0.0
    max_v_diff = 0.0

    for step in range(num_steps):
        # Drake step
        sim.AdvanceTo((step + 1) * DT)
        q_drake = plant.GetPositions(plant_context)
        v_drake = plant.GetVelocities(plant_context)

        # Standalone step
        M = dynamics.mass_matrix(q, I_B, MASS)
        k_bias = dynamics.bias_forces(q, v, I_B)
        v_star = dynamics.free_motion_velocity(M, v, tau_g, k_bias, DT)

        contact_data = contact.sphere_halfspace(q, RADIUS, R_WH, p_WH)
        if contact_data is not None:
            p_WC = contact.hertz_contact_point(
                contact_data["p_WCa"], contact_data["p_WCb"],
                geometry_k, geometry_k)
            R_WC = contact.contact_frame(contact_data["nhat_BA_W"])
            J = contact.contact_jacobian(q, p_WC, R_WC)
            v_next = solver.sap_solve(M, v_star, J, contact_data["phi0"],
                                       k_combined, tau_combined, mu_combined,
                                       v, DT)
        else:
            v_next = v_star

        q = dynamics.integrate_state(q, v_next, DT)
        v = v_next

        q_diff = np.max(np.abs(q - q_drake))
        v_diff = np.max(np.abs(v - v_drake))
        max_q_diff = max(max_q_diff, q_diff)
        max_v_diff = max(max_v_diff, v_diff)

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d}: q_diff={q_diff:.2e}, v_diff={v_diff:.2e}")
            print(f"    q_mine=[{q[4]:.6f},{q[5]:.6f},{q[6]:.6f}] "
                  f"q_drake=[{q_drake[4]:.6f},{q_drake[5]:.6f},{q_drake[6]:.6f}]")
            print(f"    v_mine=[{v[3]:.6f},{v[4]:.6f},{v[5]:.6f}] "
                  f"v_drake=[{v_drake[3]:.6f},{v_drake[4]:.6f},{v_drake[5]:.6f}]")

    ok = max_q_diff < 1e-4 and max_v_diff < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Max q drift: {max_q_diff:.2e}, max v drift: {max_v_diff:.2e}")
    return ok


def main():
    print("=" * 60)
    print("Standalone SAP Validation Against Drake")
    print("=" * 60)

    diagram, plant, scene_graph = setup_drake()

    # Shared context for phases 1-3
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Test states
    q_contact = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, RADIUS])  # on plane
    v_contact = np.array([0.0, 0.0, 0.0, 0.5, 0.0, -0.1])
    q_above = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.2])  # above plane
    v_zero = np.zeros(6)

    all_ok = True

    # Phase 1: Dynamics (no contact needed)
    all_ok &= validate_dynamics(plant, plant_context, q_above, v_contact)

    # Phase 2: Collision detection
    all_ok &= validate_collision(
        plant, plant_context, scene_graph, context, q_contact)

    # Phase 3: Contact Jacobian
    all_ok &= validate_jacobian(plant, plant_context, q_contact, v_contact)

    # Phase 4: Single step (with contact)
    all_ok &= validate_single_step(diagram, plant, scene_graph,
                                    q_contact, v_contact)

    # Phase 5: No-contact passthrough
    all_ok &= validate_no_contact(diagram, plant)

    # Phase 6: Multi-step trajectory
    all_ok &= validate_multi_step(diagram, plant, scene_graph, num_steps=1000)

    print()
    print("=" * 60)
    if all_ok:
        print("ALL PHASES PASSED")
    else:
        print("SOME PHASES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
