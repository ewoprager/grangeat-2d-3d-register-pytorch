# Grangeat-based 2D/3D image registration

## Build

To do this directly with setuptools, starting in the root directory of the repository:

```bash
source .venv/bin/activate
cd Extension
python setup.py develop
```

- In debug: `--debug`
- Without CUDA: `--no-cuda`

## Documentation

The documentation for this extension is uploaded
at https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/index.html.

## Module contents

See [`__init__.py`](__init__.py) for all the names that can be imported.

### Functions

The following functions are provided. These map directly to the C++ functions documented at the link above. The mappings
are given in [ops.py](ops.py). Unless stated otherwise, all functions are implemented for both CPU and CUDA.

- `radon2d`
- `radon2d_v2`
- `d_radon2d_dr`
- `radon3d`
- `radon3d_v2`
- `d_radon3d_dr`
- `d_radon3d_dr_v2`
- `resample_sinogram3d`
- `normalised_cross_correlation`
- `grid_sample3d`
- `project_drr`
- `project_drr_cuboid_mask`
- `resample_sinogram3d_cuda_texture` (only implemented for CUDA)

### Structures

The following structures are provided. These are defined in [structs.py](structs.py), and are almost direct mappings to
C++ structures documented at the link above.

- `CUDATexture2D`
- `CUDATexture3D`