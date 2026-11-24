# Operators overview

What each operator computes and where it is implemented. Argument-level documentation is in the headers. See the
[backend overview](backend_overview.md) for how the implementations are organised.

| Operator                             | Header                     | CPU | CUDA | Backward        |
|--------------------------------------|----------------------------|-----|------|-----------------|
| `project_drr`                        | `ProjectDRR.h`             | yes | yes  | w.r.t. matrix   |
| `project_drrs_batched`               | `ProjectDRR.h`             |     | yes  |                 |
| `project_drr_cuboid_mask`            | `ProjectDRRCuboidMaskCPU.h` | yes | yes  |                 |
| `project_drr_cuboid_masks_batched`   | `ProjectDRRCuboidMaskCPU.h` |     | yes  |                 |
| `objective_function`                 | `ObjectiveFunction.h`      |     | yes  |                 |
| `normalised_cross_correlation`       | `Similarity.h`             | yes | yes  | w.r.t. first arg |
| `radon2d`, `radon2d_v2`              | `Radon2D.h`                | yes | yes  |                 |
| `d_radon2d_dr`                       | `Radon2D.h`                | yes | yes  |                 |
| `radon3d`, `radon3d_v2`              | `Radon3D.h`                | yes | yes  |                 |
| `d_radon3d_dr`, `d_radon3d_dr_v2`    | `Radon3D.h`                | yes | yes  |                 |
| `resample_sinogram3d`                | `ResampleSinogram3D.h`     | yes | yes  |                 |
| `resample_sinogram3d_cuda_texture`   | `ResampleSinogram3D.h`     |     | yes  |                 |
| `grid_sample3d`                      | `GridSample3D.h`           | yes | yes  |                 |

On the CPU, the `_v2` operators call the same code as the originals.

## DRR projection

`project_drr` generates a digitally reconstructed radiograph (DRR). For each detector pixel, a ray is cast from the source,
transformed into CT space with the inverse homography matrix, and sampled at even steps. The samples span the sphere that
encloses the volume, and their number equals the volume's largest dimension. The result is the sum of samples multiplied
by the step length.

`project_drrs_batched` does this for many transformations in one kernel launch, which is what population-based optimisers
use. The output has the batch dimensions of the matrices, followed by the image dimensions.

`project_drr_backward` gives the derivative of a loss with respect to the inverse homography matrix. It is called by
autograd rather than directly.

## Cuboid mask projection

`project_drr_cuboid_mask` describes, for each pixel, how much of its ray is covered by the CT volume. The CT is treated as a
cuboid, with a further cuboid four times as tall placed above and below it in z. The output is:

```
distance in CT / (distance in CT + distance in cuboids above and below)
```

This is 1 where the CT covers the ray completely, and falls towards 0 where the ray passes through anatomy that is cut off
at the top or bottom of a truncated CT. Rays that miss everything are given 1. The experiments library uses this as a
scaling or weighting image. Only the size, spacing and pose of the volume are needed, so it is far cheaper than a DRR.

The intersection lengths come from `RayConvexPolyhedronDistance`, which handles any convex polyhedron.

## Combined objective function

`objective_function` (CUDA only) fuses the DRR, the mask and a weighted similarity into a single kernel. For each
transformation in the batch, it evaluates the weighted zero-normalised cross-correlation between the DRR and the fixed
image multiplied by the mask. Weights come from the mask through a smooth step raised to the power `1 / alpha^2`. This
avoids storing a batch of intermediate images.

## Similarity

`normalised_cross_correlation` returns the zero-normalised cross-correlation (ZNCC) of two tensors, plus the intermediate
sums that its backward pass reuses. The Python wrapper returns only the ZNCC.

## Radon transforms

`radon2d` approximates line integrals through an image, and `radon3d` plane integrals through a volume, at arrays of polar
or spherical coordinates. The `d_..._dr` versions evaluate the derivative with respect to the unsigned distance from the
origin, differentiating the interpolation analytically rather than taking finite differences. These derivatives are the
quantities related by Grangeat's relation.

The original versions launch one kernel, in which each thread evaluates a whole integral. The V2 versions launch one kernel
per integral and share the samples between threads, which suits a small number of finely sampled integrals.

## Sinogram resampling

`resample_sinogram3d` is the inner step of Grangeat-based registration. For each (phi, r) of the fixed image's 2D sinogram:

1. the line on the detector and the source define a plane, found using the transposed projection matrix;
2. the point on that plane closest to the CT origin is converted to spherical coordinates;
3. the precomputed 3D sinogram is sampled there; and
4. the sign is flipped where the point on the line closest to the detector origin lies inside the circle whose diameter
   joins the detector origin and the projected CT origin.

The sinogram layout is named by a string: `"classic"` is an even grid in (phi, theta, r), and `"healpix"` distributes
orientations evenly over the sphere using the HEALPix mapping. Both expect the padding added by the Python sinogram classes
in `reg23_experiments.data.sinogram`.

`resample_sinogram3d_cuda_texture` is the same operation on a `CUDATexture3D` uploaded in advance, using hardware
interpolation. It is the fast path for repeated evaluation.

## Grid sampling

`grid_sample3d` is similar to `torch.nn.functional.grid_sample`, with coordinates in (-1, 1), but the address mode can be
chosen per axis and the output can be written to an existing tensor.
