This library contains tools for 2D/3D radiographic image registration using both DRR-based and Grangeat-based
approaches, leveraging the GPU-accelerated implementations from the `reg23_core` package.

## The two registration approaches

Both approaches search for the rigid transformation of a CT volume that best explains an X-ray image. They differ in how a
candidate transformation is compared with the X-ray:

- **DRR-based**: project a digitally reconstructed radiograph (DRR) through the CT at the candidate transformation, and
  compare it with the X-ray.
- **Grangeat-based**: precompute the derivative of the 3D Radon transform of the CT once, resample it at the candidate
  transformation, and compare the result with a sinogram derived from the X-ray. The per-evaluation cost no longer depends
  on ray casting through the volume.

## Main concepts

- A `reg23_experiments.data.structs.Transformation` is an axis-angle rotation and a translation in mm, applied to the CT.
- Optimisers work on a scaled 6-vector of *parameters* rather than on the transformation itself. The mapping is in
  `reg23_experiments.ops.optimisation`.
- Intermediate images are nodes in a directed acyclic data graph, which re-evaluates them only when needed. See
  `reg23_experiments.ops.data_manager`.
- Expected failures are returned as `reg23_experiments.data.structs.Error` values, not raised.

## Subpackages

### [analysis](reg23_experiments/analysis.html)

Curve fitting, plotting, and reshaping of experimental results.

### [data](reg23_experiments/data.html)

Data structures, sinograms, parameters, and the storage of manual annotations.

### [experiments](reg23_experiments/experiments.html)

Experiment configuration and execution, and the registration pipelines.

### [io](reg23_experiments/io.html)

Reading and writing images, volumes, caches and serialised data.

### [ops](reg23_experiments/ops.html)

Operations on data: geometry, similarity metrics, optimisers, filtering and the data graph.

### [utils](reg23_experiments/utils.html)

Non-domain-specific helpers.
