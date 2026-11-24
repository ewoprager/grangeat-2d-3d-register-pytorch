Operations on data. Apart from `data_manager`, these are functions and small classes without global state.

| Module                   | Purpose                                                                           |
|--------------------------|-----------------------------------------------------------------------------------|
| `geometry`               | Coordinate conversions, DRRs and masks, projection of points, automatic cropping  |
| `grangeat`               | Both sides of Grangeat's relation: CT sinogram and fixed-image sinogram           |
| `similarity_metric`      | ZNCC, gradient correlation, gradient difference, mutual information               |
| `swarm`                  | Batched particle swarm optimisation                                               |
| `optimisation`           | Parameter scaling, local search, and an older swarm based on `pyswarms`           |
| `optimisation_instances` | Optimisers behind the common `OptimisationInstance` interface                     |
| `cmaes`                  | CMA-ES optimiser; not used                                                        |
| `objective_function`     | Standalone Grangeat objective functions, from before the data graph               |
| `image`, `volume`        | Blurring, downsampling and frequency filtering in 2D and 3D, line fitting         |
| `signal`                 | 1D Gaussian kernel                                                                |
| `ct`                     | Conversion from Hounsfield Units to attenuation coefficients                      |
| `drr`                    | Synthetic X-rays with scatter and noise, for use as targets                       |
| `fiducials`              | Synthesis and sub-voxel localisation of ball-bearing markers                      |
| `backprojection`         | Rays through segmented points, and their intersection with a sphere               |
| `autograd_impl`          | Custom differentiable transformation-to-matrix function                           |
| `new_simpleitk`          | Conversion between our parameters and SimpleITK transforms                        |
| `data_manager`           | The directed acyclic data graph                                                   |

## Geometry

`generate_drr` and `generate_drr_cuboid_mask` wrap the compiled operators, taking a `Transformation` and inverting it as
needed. `generate_drr_python` is a slow reference version for tests. `project_vectors` projects 3D points to the detector.
`get_crop_nonzero_drr` and `get_crop_full_depth_drr` return the smallest crop that contains the projected CT, and the crop
in which rays pass through the full depth of a truncated CT.

`fixed_polar_to_moving_spherical` is the geometric core of the Grangeat method: it maps a line on the detector to the plane
in CT space containing that line and the source.

## Grangeat's relation

- `calculate_radon_volume` evaluates the derivative of the 3D Radon transform of the CT over a grid. It is done once per
  volume and cached.
- `calculate_fixed_image` evaluates the matching quantity for the X-ray: cosine weighting, the derivative of the 2D Radon
  transform, then scaling.
- The two are compared after resampling the first with the methods of `reg23_experiments.data.sinogram`.

## Similarity metrics

All metrics take two broadcastable tensors and optional weights, and return a larger value for a better match. They work over
the last two dimensions, or over `dim` where that is supported, so a batch of images gives a batch of values. Optimisers
minimise, so callers negate the result. Metrics are selected by name with
`reg23_experiments.experiments.helpers.string_to_sim_met`.

## Optimisation

Optimisers work on a scaled 6-vector: rotation is multiplied by 32, the in-plane translations are unchanged, and the
out-of-plane translation is multiplied by 0.05. A unit step then has a similar effect on the image in each direction.
Convert with `mapping_transformation_to_parameters` and `mapping_parameters_to_transformation`.

`Swarm` evaluates its objective function in batches, taking a `(batch, 6)` tensor and returning `(batch,)`. The caller
runs each iteration, so it can change the configuration, or the objective function itself, between iterations.

## Fiducials

Markers are refined from a rough position by gradient descent. In 3D and for disks in 2D, the objective rewards image gradients
that point inwards around a circle or sphere of the known radius. `refine_spherical_fiducial_2d` instead correlates the patch
with the projection of a sphere.
