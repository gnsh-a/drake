# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drake ("Dragon") is a C++/Python robotics toolbox for model-based design and verification. This is a fork at `gnsh-a/drake` (origin), with upstream at `RobotLocomotion/drake`. Documentation: https://drake.mit.edu

## Build System

Drake uses **Bazel 9.0.0** with C++23 and Python 3.10+.

```bash
# Setup (first time)
setup/install_prereqs --developer

# Build
bazel build //common:...              # Build a package
bazel build //...                     # Build everything

# Test
bazel test common:polynomial_test     # Single test
bazel test //common:...               # All tests in a package
bazel test //...                      # All tests

# Lint
bazel test --config lint //...        # All lint checks
bazel test --config lint //common:... # Lint one package

# Debug/sanitizer builds
bazel test -c dbg common:polynomial_test       # Debug mode
bazel test --config=asan common:polynomial_test # AddressSanitizer
bazel test --config=kcov common:polynomial_test # Coverage
```

Default build mode is optimized (`-c opt`). Tests use `memq://` for LCM isolation.

## Architecture

Core libraries with acyclic dependencies (downstream → upstream):

- **common/** - General-purpose C++ utilities (macros, containers, types)
- **math/** - Mathematical utilities
- **geometry/** - Geometry, collision detection, SceneGraph
- **multibody/** - Rigid body dynamics and kinematics (MultibodyPlant)
- **systems/** - Dynamical systems framework (DiagramBuilder, System, Context)
- **solvers/** - Optimization solvers (MathematicalProgram)
- **planning/** - Motion planning
- **manipulation/** - Robot manipulation utilities
- **perception/** - Perception algorithms
- **visualization/** - Meshcat and other visualizers
- **lcm/** - LCM middleware integration
- **examples/** - Example programs

All C++ code lives in the `drake::` namespace.

## Python Bindings (pydrake)

Located in `bindings/pydrake/`. Uses pybind11 to wrap C++ code. Mirrors the C++ directory structure:

```python
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import SceneGraph
```

Build macros: `drake_pybind_library()`, `drake_py_unittest()`, `drake_py_library()` (in `tools/skylark/`).

**Local install**: pydrake v1.50.0 is installed in the `drake-py` conda environment. Activate with `conda activate drake-py` before running pydrake scripts.

## Code Style

**C++**: Google C++ Style Guide (Drake fork). 80-char lines, `#pragma once`, `snake_case` functions, `CamelCase` classes, `member_name_` for private members. Formatted with clang-format.

**Python**: PEP 8 via Ruff (`.ruff.toml`). 80-char lines, absolute imports only.

**BUILD files**: Formatted with buildifier.

## Key Build Macros

Located in `tools/skylark/`:
- `drake_cc.bzl`: `drake_cc_library()`, `drake_cc_binary()`, `drake_cc_googletest()`
- `drake_py.bzl`: `drake_py_library()`, `drake_py_binary()`, `drake_py_unittest()`
- `pybind.bzl`: `drake_pybind_library()`

## Conventions

- Tests go in `*/test/` subdirectories; C++ tests use gtest, Python tests use unittest
- `*/dev/` directories contain experimental code with relaxed standards
- Drake models are external: `@drake_models`
- Use `FindDrakeResources()` for model/data file discovery
- Deprecation: use `DRAKE_DEPRECATED("removal_date", "message")` macro
- PR size limit: ≤750 lines added/changed (hard cap 1500)
