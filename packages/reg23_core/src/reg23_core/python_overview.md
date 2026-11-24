# Python layer overview

`reg23_core` is the importable Python package that wraps the compiled [backend](../backend/backend_overview.md). It is kept
thin: no algorithm is implemented here.

| Module        | Purpose                                                                          |
|---------------|----------------------------------------------------------------------------------|
| `ops.py`      | A wrapper function per operator, calling `torch.ops.reg23_core.<name>.default`   |
| `structs.py`  | `CUDATexture2D` and `CUDATexture3D`, wrapping the pybind11 texture classes       |
| `autograd.py` | Backward passes, registered with `torch.library.register_autograd`               |

Importing the package loads the compiled library, which registers the operators with PyTorch, and then registers the
backward passes. Everything public is re-exported, so `from reg23_core import project_drr` works.

## Wrapper functions

Wrappers give the operators keyword arguments and type hints. Most also convert spacings and matrices to `torch.float64`,
which the backend requires. Image and volume tensors must be `torch.float32` and are passed through unchanged.

Vector arguments are ordered (x, y) or (x, y, z). Image and volume tensors are ordered (rows, columns) or
(z, y, x).

Some names exist only when `torch.cuda.is_available()` is true: the batched projections, `objective_function`,
`resample_sinogram3d_cuda_texture` and both texture classes. Guard imports of these if code should also run on the CPU.

## Textures

A `CUDATexture3D` copies a CUDA tensor into a texture that later calls can reuse. Construction is the expensive part, so make
one per volume and keep it. The GPU memory is released when the Python object is deleted.

## Autograd

Two operators have backward passes:

- `project_drr`, with respect to the inverse homography matrix. No gradient is produced for the volume or the
  detector geometry.
- `normalised_cross_correlation`, with respect to its first argument only.

All others are treated as having no gradient.
