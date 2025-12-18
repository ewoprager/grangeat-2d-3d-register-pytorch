# Grangeat-based 2D/3D image registration

## Documentation

The documentation for this extension is available here:

[https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/reg23](https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/reg23)

# Package contents

```text
.
├── src/                    # 
│   ├── backend/            #
│   │   ├── include/reg23/  # Header files shared by all operator implementations
│   │   ├── cpu/            # C++ source files used for CPU implementations of operators
│   │   ├── cuda/           # CUDA source files used for CUDA implementations of operators
│   │   └── main.cpp        # Declares all bindings for operators of all implementations
│   └── reg23/              #
│       ├── ops.py          # Thin wrappers for all PyTorch extension functions
│       ├── structs.py      # Wrapper classes for C++ structures that were given python bindings
│       └── autograd.py     # Implementations of backward passes using `torch.library.register_autograd`
├── tests/                  # Pytest test functions
├── Docs/                   # Doxygen stuff for generation of the documentation linked below.
├── README.md               #
├── mainpage.md             # The main page used for the documentation webpage.
├── Conventions.md          # Contains details of C++ coding conventions, regarding style and structure.
├── pyproject.toml          # The project configuration file used by `uv` to setup the environment and dependencies of the reg23 library
├── setup.py                # A script used by `setuptools` to build the package.
└── CMakeLists.txt          # Cmake is not used for building, but this file allows IDEs to find headers and process files properly for useful autocompletion and syntax checking.
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