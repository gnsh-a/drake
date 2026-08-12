#include "drake/tools/voxel_sdf_experiments/common/mesh_export.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace tools {
namespace voxel_sdf_experiments {
namespace {

std::filesystem::path ScratchPath(const std::string& name) {
  const char* directory = std::getenv("TEST_TMPDIR");
  return std::filesystem::path(directory != nullptr ? directory : ".") / name;
}

/* A quad and a triangle sharing an edge, so the file exercises mixed face
 sizes rather than triangles only. */
SurfaceView MakeMixedSurface() {
  return SurfaceView{.num_vertices = 5,
                     .vertices_W = {{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {1.0, 1.0, 0.0},
                                    {0.0, 1.0, 0.0},
                                    {2.0, 0.5, 0.0}},
                     .faces = {
                         Face{.centroid_W = {0.5, 0.5, 0.0},
                              .area = 1.0,
                              .pressure = 10.0,
                              .normal_W = Eigen::Vector3d::UnitZ(),
                              .vertex_indices = {0, 1, 2, 3}},
                         Face{.centroid_W = {4.0 / 3.0, 0.5, 0.0},
                              .area = 0.5,
                              .pressure = 20.0,
                              .normal_W = Eigen::Vector3d::UnitZ(),
                              .vertex_indices = {1, 4, 2}},
                     }};
}

std::string ReadAll(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream stream;
  stream << file.rdbuf();
  return stream.str();
}

GTEST_TEST(MeshExportTest, WritesPolydataWithCellFields) {
  const std::filesystem::path path = ScratchPath("mixed.vtk");
  WriteSurfaceVtk(path, MakeMixedSurface(), "mixed surface");
  const std::string contents = ReadAll(path);

  EXPECT_NE(contents.find("# vtk DataFile Version 3.0"), std::string::npos);
  EXPECT_NE(contents.find("mixed surface"), std::string::npos);
  EXPECT_NE(contents.find("DATASET POLYDATA"), std::string::npos);
  EXPECT_NE(contents.find("POINTS 5 double"), std::string::npos);
  // Connectivity size is (1 + 4) + (1 + 3) = 9.
  EXPECT_NE(contents.find("POLYGONS 2 9"), std::string::npos);
  EXPECT_NE(contents.find("4 0 1 2 3"), std::string::npos);
  EXPECT_NE(contents.find("3 1 4 2"), std::string::npos);
  EXPECT_NE(contents.find("CELL_DATA 2"), std::string::npos);
  EXPECT_NE(contents.find("SCALARS pressure double 1"), std::string::npos);
  EXPECT_NE(contents.find("SCALARS area double 1"), std::string::npos);
  EXPECT_NE(contents.find("SCALARS component_id int 1"), std::string::npos);
}

GTEST_TEST(MeshExportTest, PressureRoundTripsExactly) {
  const std::filesystem::path path = ScratchPath("pressure.vtk");
  WriteSurfaceVtk(path, MakeMixedSurface(), "pressure");
  std::istringstream stream(ReadAll(path));
  std::string line;
  std::vector<double> pressures;
  bool in_pressure = false;
  while (std::getline(stream, line)) {
    if (line == "SCALARS pressure double 1") {
      in_pressure = true;
      std::getline(stream, line);  // LOOKUP_TABLE
      continue;
    }
    if (in_pressure) {
      if (line.rfind("SCALARS", 0) == 0) break;
      pressures.push_back(std::stod(line));
    }
  }
  ASSERT_EQ(pressures.size(), 2);
  EXPECT_EQ(pressures[0], 10.0);
  EXPECT_EQ(pressures[1], 20.0);
}

GTEST_TEST(MeshExportTest, CreatesMissingParentDirectory) {
  const std::filesystem::path path =
      ScratchPath("nested") / "deeper" / "surface.vtk";
  WriteSurfaceVtk(path, MakeMixedSurface(), "nested");
  EXPECT_TRUE(std::filesystem::exists(path));
}

GTEST_TEST(MeshExportTest, RejectsBadInput) {
  const SurfaceView empty;
  EXPECT_THROW(WriteSurfaceVtk(ScratchPath("empty.vtk"), empty, "empty"),
               std::logic_error);

  SurfaceView bad_index = MakeMixedSurface();
  bad_index.faces[0].vertex_indices = {0, 1, 99};
  EXPECT_THROW(WriteSurfaceVtk(ScratchPath("bad.vtk"), bad_index, "bad"),
               std::logic_error);

  SurfaceView degenerate = MakeMixedSurface();
  degenerate.faces[0].vertex_indices = {0, 1};
  EXPECT_THROW(WriteSurfaceVtk(ScratchPath("degen.vtk"), degenerate, "degen"),
               std::logic_error);

  EXPECT_THROW(WriteSurfaceVtk(ScratchPath("title.vtk"), MakeMixedSurface(),
                               "two\nlines"),
               std::logic_error);
}

}  // namespace
}  // namespace voxel_sdf_experiments
}  // namespace tools
}  // namespace drake
