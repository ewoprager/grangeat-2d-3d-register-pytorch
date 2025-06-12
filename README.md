# A PyTorch extension for fast 2D/3D radiographic image registration using Grangeat's relation

Based on this paper:

Frysch R, Pfeiffer T, Rose G. A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. Med
Image Anal. 2021 Jan;67:101815. doi: 10.1016/j.media.2020.101815. Epub 2020 Sep 30. PMID: 33065470.

This project is very much in the experimental stages, so none of the code is very usable yet. It is being actively
developed as part of a PhD.

## Setup

The (much slower) CPU implementations should work on all platforms.

CUDA is required for the more expensive functionality to run quickly. NVCC version information:

```bash
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

The build can be done faster with Ninja installed. For Clion users: note that `setuptools` won't find the Ninja
installation within
Clion.

On Ubuntu:

```bash
sudo apt install ninja-build
```

### With `uv`

Initialise the virtual environment:

```bash
uv venv
source .venv/bin/activate
```

## Scripts you can run

### A Qt-based interface using `napari`

This can be run for interactive manipulation and registration of a CT or synthetic volume with an X-ray image or DRR:
```bash
uv run interface.py -h
uv run interface.py --ct-nrrd-path "/path/to/ct.nrrd" --x-ray "/path/to/x_ray.dcm"
```

Controls:
- With the DRR selected in the 'layer list' window on the left, hold `alt` and drag with the left and right mouse  
  buttons pressed to change the rotation and translation transformation parameters respectively. The sensitivity of this
  is controlled in the 'View Options' window on the bottom left.
- The numerical values of the transformation parameters can be changed, saved and loaded in the 'Transformations' 
  tab on the right.
- A lot of useful information is printed to std out, including warnings and errors so keep an eye on this while 
  using the interface.

![interface_2025-05-30.png](figures/interface_2025-05-30.png)

### Run Radon transform algorithms on CPU and GPU (CUDA) to compare performance:

```bash
uv run benchmark_radon2d.py "/path/to/x_ray.dcm"
uv run benchmark_radon3d.py "/path/to/ct.nrrd"
```

### Run the Grangeat-based resampling algorithms on CPU and GPU (CUDA) to compare performance:

```bash
uv run benchmark_resample_sinogram3d.py -h
uv run benchmark_resample_sinogram3d.py --no-load --no-save --sinogram-size 64 # run on synthetic data
uv run benchmark_resample_sinogram3d.py --ct-nrrd-path "/path/to/ct.nrrd"
```

### Run registration experiments:

```bash
uv run register.py -h
uv run register.py --no-load --no-save --sinogram-size 64 # run on synthetic data
uv run register.py --ct-nrrd-path "/path/to/ct.nrrd"
```

### Dev scripts

Scripts can also be run directly with the python binary from your virtual environment, e.g. as follows:
```bash
python interface.py --help
```
This is useful if you want to run with a debugger attached (e.g. if you have this as a run configuration in an IDE),
but note that this will not check for correctly install packages, nor initialise the build of the extension if the
source code has changed, as `uv` is not run here, so make sure to run `uv sync` beforehand if you have changed any
dependencies or the extension.

Any scripts that aren't contained in the root directory must from the root directory, with the `PYTHONPATH` variable
set to the root directory.

For example, to run the script `registration/lib/dev_scripts/dev_sinogram.py`:
```bash
PYTHONPATH=$PWD uv run registration/lib/dev_scripts/dev_sinogram.py --help 
```
or
```bash
PYTHONPATH=$PWD python registration/lib/dev_scripts/dev_sinogram.py --help 
```

## Experiments so far

DRR (= g) generated at random transformation:

![ground_truth_drr.png](figures/ground_truth_drr.png)

The associated fixed image (= 1/cos^2 alpha * d/ds R2\[cos gamma * g\])

![dds_R2_gtilde_ground_truth.png](figures/dds_R2_gtilde_ground_truth.png)

The 3D Radon transform of the volume data (= R3\[mu\]), resampled according to the ground truth transformation (this
should roughly
match the above image):

![ddr_R3_mu_resampled_ground_truth.png](figures/ddr_R3_mu_resampled_ground_truth.png)

A plot of the -ZNCC landscape over 2 dimensions (two angular components of transformation) between the fixed image and
the resampled Radon-transformed volume, with the ground truth transformation at the centre:

![landscape.png](figures/landscape.png)

Starting from a different random transformation, optimising the ZNCC between these images over the transformation using
the basin-hopping algorithm:

(specifically `scipy.optimize.basinhopping` with `T=1.0`, `minimizer_kwargs={"method": 'Nelder-Mead'}`)

![rotation_params_against_iteration.png](figures/rotation_params_against_iteration.png)
![translation_params_against_iteration.png](figures/translation_params_against_iteration.png)
![loss_against_iteration.png](figures/loss_against_iteration.png)

The fixed image (= 1/cos^2 alpha * d/ds R2\[cos gamma * g\]) at the converged transformation:

![ddr_R3_mu_resampled_converged.png](figures/ddr_R3_mu_resampled_converged.png)

DRR generated at the converged transformation:

![converged_drr.png](figures/converged_drr.png)

Optimisation completed in 125.508 seconds, performing a total of 45,472 function evaluations.

Here is a plot of the -ZNCC similarity between the fixed image and the resampled moving image against the distance in
SE3 between the transformation and the ground truth transformation for 1000 random transformations:

![loss_vs_distance.png](figures/loss_vs_distance.png)

[//]: # (# Resampling)

[//]: # ()

[//]: # (The 3D sinogram image is stored as a 3D grid of values, where the dimensions correspond to different values of phi,)

[//]: # (theta and r. While this is very efficient for resampling, having the same number of value of phi for every value of)

[//]: # (theta results in memory inefficiency and an extremely high densities of values near theta = +/- pi/2.)

[//]: # ()

[//]: # (Smoothing the sampling consistently over S^2 to eliminate the second of these effects demonstrates that the line of)

[//]: # (discontinuity visible in the resampling of the sinogram is due to this effect:)

[//]: # ()

[//]: # (![dds_R2_gtilde_ground_truth_no_sample_smoothing.png]&#40;figures/dds_R2_gtilde_ground_truth_no_sample_smoothing.png&#41;)

[//]: # (![ddr_R3_mu_resampled_ground_truth_no_sample_smoothing.png]&#40;figures/ddr_R3_mu_resampled_ground_truth_no_sample_smoothing.png&#41;)

[//]: # (![ddr_R3_mu_resampled_ground_truth_sample_smoothing.png]&#40;figures/ddr_R3_mu_resampled_ground_truth_sample_smoothing.png&#41;)

[//]: # ()

[//]: # (Although there is no significant difference manifest in the resulting optimisation landscape:)

[//]: # ()

[//]: # (![landscape_no_sample_smoothing.png]&#40;figures/landscape_no_sample_smoothing.png&#41;)

[//]: # (![landscape_sample_smoothing.png]&#40;figures/landscape_sample_smoothing.png&#41;)

## IDE integration

All the following IDE integration advice is based on CLion 2024.3.1.1.

'Extension/CMakeLists.txt' exists exclusively to aid your IDE with syntax highlighting and error detection in the
extension .cpp and .cu source files. Configure a CMake project in your IDE to make use of this.