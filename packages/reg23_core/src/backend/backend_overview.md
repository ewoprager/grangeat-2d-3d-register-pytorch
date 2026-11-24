# Backend overview

The backend is the C++/CUDA half of `reg23_core`: a PyTorch extension compiled into the Python module
`backend.reg23_core`. It holds the operations that are too slow to write in Python: ray casting through CT volumes, Radon
transforms, 3D sinogram resampling and image similarity.

For what each operator does, see the [operators overview](operators_overview.md). For how they are called from Python,
see the [Python layer overview](../reg23_core/python_overview.md).

## Layout

| Path                  | Contents                                                                                   |
|-----------------------|--------------------------------------------------------------------------------------------|
| `main.cpp`            | Operator schemas (`TORCH_LIBRARY`), their CPU and CUDA registrations, and pybind11 classes |
| `include/reg23_core/` | Maths types, texture types, and a header per operator family                               |
| `cpu/`                | One `.cpp` per operator family, using plain loops                                          |
| `cuda/`               | One `.cu` per operator family, plus `CUDATexture.cu`                                       |

## How an operator is put together

Each operator family is split so that the CPU and CUDA versions share as much code as possible:

1. A header, e.g. `ProjectDRR.h`, declares one function per platform (`ProjectDRR_CPU`, `ProjectDRR_CUDA`).
2. The same header defines a struct, e.g. `ProjectDRR<texture_t>`, which acts as a namespace for shared code. Its
   `Common()` function validates the inputs and converts tensors into a `CommonData` struct. Code that runs per sample or
   per ray is marked `__host__ __device__`, so both platforms can call it.
3. The `.cpp` calls `Common()` with a CPU texture type and loops over the output.
4. The `.cu` calls `Common()` with a CUDA texture type and launches a kernel, with one thread per output element.
5. `main.cpp` declares the schema and registers each implementation against a dispatch key. PyTorch then picks the
   implementation from the device of the tensor arguments.

The V2 Radon operators, the CUDA similarity, the combined objective function and the DRR backward pass instead reduce over
the threads in a block using shared memory, followed by a sum over blocks with PyTorch. The reduction halves the block at
each step, so it needs a block size that is a power of two.

## Maths types

- `Vec<T, N>` (`Vec.h`): a fixed-size vector derived from `std::array`, with element-wise operators. A matrix is a `Vec` of
  columns, so it is **column-major**: `mat[column][row]`. `Vec::FromTensor2D` converts an ordinary row-major PyTorch matrix
  into this form.
- `Linear<T>` and `Linear2<T>` (`Common.h`): `intercept + gradient * x`, with one and two variables. They can be composed,
  which lets a chain such as *index → world offset → texture coordinate* be collapsed once per line or plane.
- `UnflipSphericalCoordinate` (`Common.h`): maps spherical coordinates back into the range used by the sinograms.

## Textures

A texture is a sampled 2D or 3D array with a size, a spacing, a centre position and per-axis address modes (`ZERO` or
`WRAP`). Sampling uses normalised coordinates in (0, 1), and `MappingWorldToTexCoord()` provides the conversion from mm.

| Type                                            | Role                                                               |
|-------------------------------------------------|--------------------------------------------------------------------|
| `Texture<dim, int, float>`                      | Base class: size, spacing, centre, coordinate mapping              |
| `Texture2DCPU`, `Texture3DCPU`                  | Non-owning pointer to tensor data, with software interpolation     |
| `CUDATexture2D`, `CUDATexture3D`                | Own a CUDA array and texture object; exposed to Python             |
| `Texture2DCUDA`, `Texture3DCUDA`                | Kernel-side view of a CUDA texture, using hardware interpolation   |
| `SinogramClassic3D<tex>`, `SinogramHEALPix<tex>` | Wrap a 3D texture so it can be sampled at spherical (r, theta, phi) |

Sizes and spacings in textures are ordered (X, Y, Z), the reverse of `tensor.size()`. All textures also provide the
derivative of the interpolated value with respect to texture coordinate, which the derivative operators rely on.

The CPU textures do not own their data, so whatever creates one must keep the underlying tensor alive.

## Coordinate conventions

These are set out in the [conventions](../../Conventions.md). The points that matter most when reading the backend:

- The detector lies in the plane z = 0, and the X-ray source is at (0, 0, source distance).
- Operators take the **inverse** homography matrix, which maps world space to CT space.
- Row and column indices of a DRR increase with world y and x respectively.
- The derivative-of-Radon operators differentiate with respect to the **unsigned** distance |r|.

## Precision

Geometry is evaluated in double precision and image data is stored in single precision. Sums for similarity metrics are
accumulated in double precision.

## Adding an operator

1. Add a header with the function declarations, documentation and a shared `Common()` struct.
2. Add the implementations in `cpu/` and `cuda/`. `setup.py` collects source files by glob, so no build change is needed.
3. Add the schema and registrations to `main.cpp`.
4. Add a wrapper to `ops.py`, and to its `__all__`. Put CUDA-only operators inside the `torch.cuda.is_available()` block.
5. If it should be differentiable, register a backward pass in `autograd.py`.
6. Add a test to `tests/` that compares the CPU and CUDA results.
