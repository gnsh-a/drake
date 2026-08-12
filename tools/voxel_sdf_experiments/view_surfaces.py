#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyvista", "pyvistaqt", "PySide6"]
# ///
"""Views frozen contact surfaces for all three representations side by side.

Pick a scene, a resolution rung, and a scalar from the drop-downs; the three
panes reload in place and share one camera, so a viewpoint held while stepping
through the ladder shows convergence directly.

Colour by component_id to see fragmentation. Gaps in the affine surface are
microns wide on a patch tens of millimetres across, so they are invisible as
geometry at this scale; the component colouring is what makes them legible.

Usage:
  uv run tools/voxel_sdf_experiments/view_surfaces.py
  uv run tools/voxel_sdf_experiments/view_surfaces.py --mesh_dir <dir>
"""

import argparse
import csv
import pathlib
import re
import sys

import numpy as np
from pyvistaqt import MainWindow
from pyvistaqt import QtInteractor
import pyvista as pv
from qtpy import QtWidgets

REPRESENTATIONS = ("tet", "plane_clip", "marching_cubes")
SCALARS = ("component_id", "pressure", "area")
STEM_PATTERN = re.compile(r"^(?P<scene>.+)__(?P<rep>.+)__h_(?P<h>.+)mm$")


def _default_mesh_dir() -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[2]
    return root / "tools/voxel_sdf_experiments/out/frozen_surface_ladder/meshes"


def _parse_stem(stem: str):
    match = STEM_PATTERN.match(stem)
    if match is None:
        return None
    return (
        match.group("scene"),
        match.group("rep"),
        float(match.group("h").replace("p", ".")),
    )


def _discover(mesh_dir: pathlib.Path) -> dict:
    """Maps (scene, h_mm, representation) -> mesh path."""
    found = {}
    for path in sorted(mesh_dir.glob("*.vtk")):
        parsed = _parse_stem(path.stem)
        if parsed is None:
            continue
        scene, representation, h_mm = parsed
        found[(scene, h_mm, representation)] = path
    return found


def _read_annotation(path: pathlib.Path) -> str:
    """Summarizes the run from the CSV written beside the mesh directory."""
    csv_path = path.parent.parent / f"{path.stem}.csv"
    if not csv_path.is_file():
        return ""
    with open(csv_path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return ""
    row = rows[0]
    faces = row.get("num_faces", "?")
    error = float(row.get("normal_force_relative_error", "nan"))
    return f"faces={faces}  force err={error:.3e}"


def _shuffled_components(mesh: pv.PolyData) -> np.ndarray:
    """Relabels component ids so neighbouring components contrast.

    The ids are union-find roots, i.e. face indices, so a sequential colour map
    would give adjacent components nearly identical colours.
    """
    ids = np.asarray(mesh.cell_data["component_id"])
    unique = np.unique(ids)
    shuffled = np.arange(len(unique))
    np.random.default_rng(0).shuffle(shuffled)
    lookup = {value: shuffled[i] for i, value in enumerate(unique)}
    return np.array([lookup[value] for value in ids])


class Window(MainWindow):
    def __init__(self, meshes: dict, parent=None):
        super().__init__(parent)
        self._meshes = meshes
        self._actors = []

        scenes = sorted({key[0] for key in meshes})
        rungs = sorted({key[1] for key in meshes}, reverse=True)

        central = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self._scene_box = self._add_combo(controls, "Scene", scenes)
        self._rung_box = self._add_combo(
            controls, "Resolution h (mm)", [f"{h:g}" for h in rungs]
        )
        self._scalar_box = self._add_combo(
            controls, "Colour by", list(SCALARS)
        )
        controls.addStretch(1)
        outer.addLayout(controls)

        self._plotter = QtInteractor(central, shape=(1, 3))
        outer.addWidget(self._plotter.interactor)
        self.signal_close.connect(self._plotter.close)

        self.setCentralWidget(central)
        self.setWindowTitle("Frozen contact surfaces")

        for box in (self._scene_box, self._rung_box, self._scalar_box):
            box.currentTextChanged.connect(self._refresh)

        self._refresh()
        self._plotter.link_views()
        self.fit_camera()

    def fit_camera(self):
        """Looks down the contact plane and fits the patch in every pane."""
        self._plotter.subplot(0, 0)
        self._plotter.view_xy()
        self._plotter.reset_camera()
        self._plotter.render()

    @staticmethod
    def _add_combo(layout, label: str, items: list) -> QtWidgets.QComboBox:
        layout.addWidget(QtWidgets.QLabel(f"{label}:"))
        box = QtWidgets.QComboBox()
        box.addItems(items)
        layout.addWidget(box)
        return box

    def _selection(self):
        return (
            self._scene_box.currentText(),
            float(self._rung_box.currentText()),
            self._scalar_box.currentText(),
        )

    def _load(self, scene: str, h_mm: float, representation: str):
        path = self._meshes.get((scene, h_mm, representation))
        if path is None:
            return None, ""
        mesh = pv.read(path)
        mesh.cell_data["component"] = _shuffled_components(mesh)
        return mesh, _read_annotation(path)

    def _refresh(self):
        scene, h_mm, scalar = self._selection()
        for actor in self._actors:
            self._plotter.remove_actor(actor, render=False)
        self._actors = []

        loaded = [
            (rep, *self._load(scene, h_mm, rep)) for rep in REPRESENTATIONS
        ]

        # Pressure and area share one colour range so the panes stay
        # comparable; component labels are per-mesh by nature.
        clim = None
        if scalar in ("pressure", "area"):
            values = [
                mesh.cell_data[scalar]
                for _, mesh, _ in loaded
                if mesh is not None
            ]
            if values:
                clim = (
                    float(min(v.min() for v in values)),
                    float(max(v.max() for v in values)),
                )

        array = "component" if scalar == "component_id" else scalar
        for column, (representation, mesh, annotation) in enumerate(loaded):
            self._plotter.subplot(0, column)
            if mesh is None:
                self._actors.append(
                    self._plotter.add_text(
                        f"{representation}\n(missing)", font_size=9
                    )
                )
                continue
            self._actors.append(
                self._plotter.add_mesh(
                    mesh,
                    scalars=array,
                    clim=clim,
                    cmap="tab20" if array == "component" else "viridis",
                    show_edges=True,
                    edge_color="black",
                    line_width=1,
                    show_scalar_bar=False,
                )
            )
            self._actors.append(
                self._plotter.add_text(
                    f"{representation}\n{annotation}", font_size=9
                )
            )
        self._plotter.render()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh_dir",
        type=pathlib.Path,
        default=_default_mesh_dir(),
        help="Directory of VTK meshes written by --mesh_output.",
    )
    args = parser.parse_args()

    meshes = _discover(args.mesh_dir)
    if not meshes:
        parser.error(
            f"no meshes found in {args.mesh_dir}; run run_ladder.py --meshes"
        )
    scenes = sorted({key[0] for key in meshes})
    rungs = sorted({key[1] for key in meshes}, reverse=True)
    print(f"{len(meshes)} meshes: scenes={scenes}, rungs={rungs} mm")

    app = QtWidgets.QApplication(sys.argv)
    window = Window(meshes)
    window.resize(1600, 700)
    window.show()
    # Fit the camera only once the interactor has its final size, otherwise the
    # surfaces are framed against the pre-layout viewport and look tiny.
    window.fit_camera()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
