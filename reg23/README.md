# Grangeat-based 2D/3D image registration

# Package contents

```
Docs/
  > Doxygen stuff for generation of the documentation linked below.

include/
  > Header C++ files

src/
  cpu/
    > C++ source files for CPU implementations of PyTorch functions
  cuda/
    > .cu source files for CUDA implementations of PyTorch functions
  mps/
    > .mm source files for MPS implementations of PyTorch functions
  main.cpp
    > This is where all the Python bindings are declared.

tests/
  > Pytest test functions

autograd.py
  > Implementations of backward passes using `torch.library.register_autograd`.

CMakeLists.txt
  > Cmake is not used for building, but this file allows IDEs to find headers and process files properly for useful autocompletion and syntax checking.

Conventions.md
  > Contains details of Python and C++ coding conventions, regarding style and structure.

mainpage.md
  > The main page used for the documentation webpage.

ops.py
  > Thin wrappers for all PyTorch extension functions

pyproject.toml
  > The project configuration file used by `uv` to setup the building of the package with `setuptools`.

setup.py
  > A script used by `setuptools` to build the package.

structs.py
  > Wrapper classes for C++ structures that were given python bindings.
```

## Build

First, make sure you have activated the virtual environment in the outer directory:

```
uv venv .venv
source .venv/bin/activate
cd reg23
```

Building of this package will automatically by done by `uv` when running any script with `uv run` in the outer
directory, but can be done manually with `uv` like so, in the `reg23/` directory:

```bash
uv pip install .[cpu]
```

or

```bash
uv pip install .[cuda]
```

*Note: In ZSH, square brackets confuse it, so add quotes, e.g.:*

```zsh
uv pip install '.[cpu]'
```

This will use a `uv` temporary build environment, and only output the resulting `.so` into `reg23/src/`.

To perform the build directly in the `reg23/` directory with setuptools:

```bash
python setup.py develop [--verbose] [--debug] [--no-cuda]
```

This will create and fill the directory `reg23/build/`.

## Documentation

The documentation for this extension is uploaded
at https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/index.html.

## Module contents

See the names listed in the `__all__` variables in `ops.py` and `structs.py` for all names that can be imported.

### Functions

The following functions are provided. These map (almost) directly to the C++ functions documented at the link above. The
mappings are given in [ops.py](ops.py). Unless stated otherwise, all functions are implemented for both CPU and CUDA.
Unless stated otherwise, backward passes for functions have not yet been implemented.

- `radon2d`
- `radon2d_v2`
- `d_radon2d_dr`
- `radon3d`
- `radon3d_v2`
- `d_radon3d_dr`
- `d_radon3d_dr_v2`
- `resample_sinogram3d`
- `normalised_cross_correlation` (backward pass also implemented)
- `grid_sample3d`
- `project_drr` (backward pass also implemented)
- `project_drr_cuboid_mask`
- `resample_sinogram3d_cuda_texture` (only implemented for CUDA)

### Structures

The following structures are provided. These are defined in [structs.py](structs.py), and are almost direct mappings to
C++ structures documented at the link above.

- `CUDATexture2D`
- `CUDATexture3D`

## MPS Development

For developing MPS implementations in Objective-C++ (.mm sources) and Metal (.metal sources), one can perfectly well use
Clion. As of writing this, Xcode surprisingly does not provide any more functionality than Clion for Objective-C++ or 
Metal shader development.

If you nevertheless want to use Xcode, create a subdirectory of `reg23/` called something like `reg23/build-xcode`,
navigate inside and generate a `.xcodeproj` according to `reg23/CMakeLists.txt` like so:
```bash
source .venv/bin/activate
cd reg23
mkdir build-xcode
cd build-xcode
cmake -G Xcode -DPython3_EXECUTABLE=$(which python) ..
```

The resulting `.xcodeproj` directory can be opened in Xcode.

The Xcode project will only include files that are included as sources in the CMake project. If by mistake some are
missed, modify the calls to `add_library` in `reg23/CMakeLists.txt` such that all source files that you wish to be
included in the Xcode project are globbed.