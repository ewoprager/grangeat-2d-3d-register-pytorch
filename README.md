# CUDA-accelerated PyTorch extension for 2D/3D radiographic image registration using Grangeat's relation

Based on this paper:

Frysch R, Pfeiffer T, Rose G. A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. Med
Image Anal. 2021 Jan;67:101815. doi: 10.1016/j.media.2020.101815. Epub 2020 Sep 30. PMID: 33065470.


## Setup

CUDA is required. NVCC version information:

```bash
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```


### With `uv`

Initialise the virtual environment:
```bash
uv venv
source .venv/bin/activate
```

Run:
```bash
uv run main.py "/path/to/ct.nrrd"
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
the Nelder-Mead
algorithm:

![params_against_iteration.png](figures/params_against_iteration.png)

![loss_against_iteration.png](figures/loss_against_iteration.png)

DRR generated at the converged transformation:

![converged_drr.png](figures/converged_drr.png)

Optimisation completed in 2.280 seconds, performing a total of 586 function evaluations.

## IDE integration

All the following IDE integration advice is based on CLion 2024.3.1.1.

'Extension/CMakeLists.txt' exists exclusively to aid your IDE with syntax highlighting and error detection in the
extension .cpp and .cu source files. Configure a CMake project in your IDE to make use of this.

To use