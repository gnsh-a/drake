"""Drop a compliant hydroelastic sphere onto a welded compliant hydroelastic
sphere using Drake's discrete-time ICF (kLagged) contact path.

Modes:
  --mode=impact  Top sphere at z=0.20 m (5 cm clear gap). Free-falls onto
                 the bottom sphere, impacts at ~1 m/s, bounces, settles.
  --mode=quasi   Top sphere at z=0.15 m (surfaces touching). Gently settles
                 into the bottom sphere under gravity.
"""

import argparse
import csv
import math
import os

import numpy as np

from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    ProximityProperties,
    Sphere,
)
from pydrake.math import RigidTransform
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    CoulombFriction,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import SpatialInertia
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    PrintSimulatorStatistics,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

SPHERE_RADIUS = 0.05
SPHERE_MASS = 0.1
HYDRO_RESOLUTION_HINT = 0.015
HYDRO_MODULUS = 5.0e5
HUNT_CROSSLEY_DISSIPATION = 0.5
MU = 0.3

BOTTOM_NAME = "bottom_sphere"
TOP_NAME = "top_sphere"
BOTTOM_COLOR = [0.2, 0.5, 0.9, 0.5]
TOP_COLOR = [0.9, 0.5, 0.2, 0.5]

TIME_STEP = 0.001
SIMULATION_TIME = 1.0
TARGET_REALTIME_RATE = 0.2
LOG_PERIOD = 0.01

BOTTOM_Z = SPHERE_RADIUS
TOP_Z_BY_MODE = {"impact": 0.20, "quasi": 0.15}

OUTPUT_DIR = "output"
DEMO_NAME = "sphere_on_sphere"

TRACE_FIELDS = [
    "time_s",
    "z_m",
    "vz_m_per_s",
    "depth_geom_m",
    "num_pairs",
    "num_faces",
    "contact_area_m2",
    "force_total_N",
]


def _make_proximity_props():
    props = ProximityProperties()
    AddContactMaterial(
        properties=props,
        dissipation=HUNT_CROSSLEY_DISSIPATION,
        friction=CoulombFriction(static_friction=MU, dynamic_friction=MU),
    )
    AddCompliantHydroelasticProperties(
        resolution_hint=HYDRO_RESOLUTION_HINT,
        hydroelastic_modulus=HYDRO_MODULUS,
        properties=props,
    )
    return props


def _add_sphere_body(plant, name, color):
    inertia = SpatialInertia.SolidSphereWithMass(
        mass=SPHERE_MASS, radius=SPHERE_RADIUS
    )
    body = plant.AddRigidBody(name=name, M_BBo_B=inertia)
    plant.RegisterVisualGeometry(
        body=body,
        X_BG=RigidTransform(),
        shape=Sphere(SPHERE_RADIUS),
        name=f"{name}_visual",
        diffuse_color=color,
    )
    plant.RegisterCollisionGeometry(
        body=body,
        X_BG=RigidTransform(),
        shape=Sphere(SPHERE_RADIUS),
        name=f"{name}_collision",
        properties=_make_proximity_props(),
    )
    return body


def make_sphere_on_sphere(contact_surface_representation):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(
            time_step=TIME_STEP,
            contact_model="hydroelastic",
            discrete_contact_approximation="lagged",
            contact_surface_representation=contact_surface_representation,
        ),
        builder,
    )

    bottom = _add_sphere_body(plant, name=BOTTOM_NAME, color=BOTTOM_COLOR)
    _add_sphere_body(plant, name=TOP_NAME, color=TOP_COLOR)

    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=bottom.body_frame(),
        X_FM=RigidTransform([0.0, 0.0, BOTTOM_Z]),
    )

    plant.Finalize()

    AddDefaultVisualization(builder=builder)

    return builder.Build(), plant


def _trace_row(plant, plant_context, top_body, time_s):
    pose = plant.GetFreeBodyPose(context=plant_context, body=top_body)
    velocity = plant.EvalBodySpatialVelocityInWorld(
        context=plant_context, body=top_body
    )
    z = float(pose.translation()[2])
    vz = float(velocity.translational()[2])
    depth_geom = max(0.0, 2.0 * SPHERE_RADIUS - abs(z - BOTTOM_Z))

    contact_results = plant.get_contact_results_output_port().Eval(
        plant_context
    )
    num_pairs = contact_results.num_hydroelastic_contacts()
    num_faces = 0
    contact_area = 0.0
    total_force = np.zeros(3)
    for i in range(num_pairs):
        info = contact_results.hydroelastic_contact_info(i)
        surface = info.contact_surface()
        num_faces += surface.num_faces()
        contact_area += surface.total_area()
        total_force += info.F_Ac_W().translational()

    return {
        "time_s": time_s,
        "z_m": z,
        "vz_m_per_s": vz,
        "depth_geom_m": depth_geom,
        "num_pairs": num_pairs,
        "num_faces": num_faces,
        "contact_area_m2": contact_area,
        "force_total_N": float(np.linalg.norm(total_force)),
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} samples)")


def _print_trace_summary(rows):
    if not rows:
        return
    max_depth = max(row["depth_geom_m"] for row in rows)
    max_pairs = max(row["num_pairs"] for row in rows)
    max_faces = max(row["num_faces"] for row in rows)
    max_force = max(row["force_total_N"] for row in rows)
    print(
        "trace summary: "
        f"max_depth={max_depth:.6g} m, "
        f"max_pairs={max_pairs}, "
        f"max_faces={max_faces}, "
        f"max_force={max_force:.6g} N"
    )


def run(diagram, plant, top_initial_z, output_csv):
    simulator = Simulator(diagram)
    ApplySimulatorConfig(
        SimulatorConfig(
            target_realtime_rate=TARGET_REALTIME_RATE,
            publish_every_time_step=True,
        ),
        simulator,
    )

    plant_context = plant.GetMyMutableContextFromRoot(
        simulator.get_mutable_context()
    )
    top_body = plant.GetBodyByName(TOP_NAME)
    plant.SetFreeBodyPose(
        plant_context,
        top_body,
        RigidTransform([0.0, 0.0, top_initial_z]),
    )

    simulator.Initialize()
    rows = [
        _trace_row(
            plant=plant,
            plant_context=plant_context,
            top_body=top_body,
            time_s=0.0,
        )
    ]

    num_samples = math.ceil(SIMULATION_TIME / LOG_PERIOD)
    for i in range(1, num_samples + 1):
        sample_time = min(i * LOG_PERIOD, SIMULATION_TIME)
        simulator.AdvanceTo(sample_time)
        rows.append(
            _trace_row(
                plant=plant,
                plant_context=plant_context,
                top_body=top_body,
                time_s=sample_time,
            )
        )

    PrintSimulatorStatistics(simulator)
    _print_trace_summary(rows)
    _write_csv(output_csv, rows)
    return rows


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "--mode",
        choices=("impact", "quasi"),
        default="impact",
        help="impact: top at z=0.20 m (gap=0.05 m), free fall. "
        "quasi: top at z=0.15 m (touching), gentle settle.",
    )
    arg_parser.add_argument(
        "--top_initial_z",
        type=float,
        default=None,
        help="Initial Z of the top sphere's center, in meters. "
        f"Defaults to {TOP_Z_BY_MODE['impact']} (impact) or "
        f"{TOP_Z_BY_MODE['quasi']} (quasi).",
    )
    arg_parser.add_argument(
        "--contact_surface_representation",
        type=str,
        default="polygon",
        help="'triangle' or 'polygon'.",
    )
    args = arg_parser.parse_args()

    top_initial_z = (
        args.top_initial_z
        if args.top_initial_z is not None
        else TOP_Z_BY_MODE[args.mode]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{DEMO_NAME}_{args.mode}.csv")

    diagram, plant = make_sphere_on_sphere(
        contact_surface_representation=args.contact_surface_representation,
    )
    print(f"mode={args.mode}, top_initial_z={top_initial_z:.4f} m")
    run(diagram, plant, top_initial_z=top_initial_z, output_csv=csv_path)
