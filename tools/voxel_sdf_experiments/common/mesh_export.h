#pragma once

#include <filesystem>
#include <string_view>

#include "drake/tools/voxel_sdf_experiments/common/surface_view.h"

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {

/* Writes `surface` to `path` as a legacy VTK POLYDATA file, which ParaView and
 PyVista open directly. Polygonal and triangular surfaces are both written as
 POLYGONS, so every representation round-trips without being triangulated.

 Three cell fields accompany the geometry:

   pressure      per-face contact pressure, in Pa.
   area          per-face area, in m^2.
   component_id  connected-component label from CalcFaceComponentIds().

 The component field is what makes fragmentation visible: a conforming mesh is
 a single flat colour, whereas a representation emitting independent per-cell
 faces shows one colour per piece. */
void WriteSurfaceVtk(const std::filesystem::path& path,
                     const SurfaceView& surface, std::string_view title);

}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
