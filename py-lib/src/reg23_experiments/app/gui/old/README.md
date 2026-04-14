# A Qt-based interface using `napari`

This can be run for interactive manipulation and registration of a CT or synthetic volume with an X-ray image or DRR:

```bash
uv run scripts/interface.py -h
uv run scripts/interface.py --ct-path "/path/to/ct.nrrd or /path/to/dicom_directory" --xray-path "/path/to/x_ray.dcm"
```

![interface_2025-09-05.png](figures/readme/interface_2025-09-05.png)

## Features

### General controls

- With the DRR selected in the 'layer list' window on the left, hold `control` and drag with the left and right mouse
  buttons pressed to change the rotation and translation transformation parameters respectively. The sensitivity of this
  is controlled in the 'View Options' window on the bottom left.
- The numerical values of the transformation parameters can be changed, saved and loaded in the 'Transformations'
  tab on the right.
- A lot of useful information is printed to std out, including warnings and errors so keep an eye on this while
  using the interface.
- If you suspect the CT and X-ray images are flipped with respect to one another, the button 'Flip' in the 'Register'
  tab
  will flip the X-ray horizontally.

### Grangeat's relation-based registration

Grangeat's relation-based registration employs a new objective function that makes use of pre-computed Radon transforms
of the X-ray image and CT volume. The new fixed image is the Radon transform of the X-ray, and is rendered in yellow
below the X-ray itself. The new moving image is a resampling of the Radon transform of the CT volume, according to the
current transformation, and is rendered in blue on top. The fixed and moving images are compared using the NCC, in the
same way that the X-ray and DRR are compared in the standard DRR-based method.

The current moving sinogram can be rendered according to the current transformation by clicking the
'Regen moving sinogram' button in the 'Sinograms' tab. Ticking the 'continuous' box will cause it to update
continuously.

To use the Grangeat's relation-based objective function for registrations, choose the option 'grangeat' in the
'Obj. func.' combo box in the 'Register' tab. To evaluate the chosen objective function once and display the result,
click the button 'Evaluate once'.

### Cropping

The region of the X-ray with which the generated DRR will be compared (and corresponding images will be generated and
compared using the Grangeat-method) can be adjusted using the sliders at the top of the 'Register' tab. The current
cropping settings can be saved, renamed and loaded using the box below.

### Masking

A mask can be applied to the fixed image to account for the CT volume not spanning the whole of the patient's head. The
mask is a function of the current transformation, but is not automatically updated continually. To regenerate the mask
at the current transformation, click 'Regenerate mask' in the 'Register' tab.

To have the fixed image rendered with the
mask applied, tick the box 'Render fixed image with mask' in the 'View Options' window (by default located in the
bottom left).

The mask can be regenerated automatically every $N$ objective function evaluations. To set the value of $N$, use the
spin box labelled 'Evals/regen. mask' in the 'Register' tab. If this is set to 0, the mask is never automatically
regenerated.

### Downsampling

Upon loading of a CT volume and fixed image, they will be downsampled by every factor of 2 (generating mipmaps). The
corresponding Radon transforms for the Grangeat method will also be calculated.

Change the level of downsampling currently being used with the spin box labelled 'Downsample level' in the 'Register'
tab.

### Optimisation algorithms

Two algorithms are currently available:

- Particle swarm optimisation
- Local search

To choose the algorithm you want to use, select it in the combo box in the 'Register' tab. The parameters specific to
the selected optimisation algorithm will be customisable below.

To run a registration, click the 'Register' button in the 'Register' tab. This will run the registration in a second
thread, which will allow the user to interact with the interface while the registration is running, but note that
changing parameters that affect the optimisation, like the cropping, will crash the software. Unfortunately, it is not
currently possible to terminate a registration prematurely without closing the whole application. Notable parameters
that can safely be modified while a registration is running:

- The view in the Napari image viewer, and any of the controls in the napari 'layer controls' and 'layer list' windows.
- 'Render fixed image with mask'
- 'Evals./regen. mask'
- 'Evals./re-plot'


## Experiments so far

DRR (= g) generated at random transformation:

![ground_truth_drr.png](figures/readme/ground_truth_drr.png)

The associated fixed image (= 1/cos^2 alpha * d/ds R2\[cos gamma * g\])

![dds_R2_gtilde_ground_truth.png](figures/readme/dds_R2_gtilde_ground_truth.png)

The 3D Radon transform of the volume data (= R3\[mu\]), resampled according to the ground truth transformation (this
should roughly
match the above image):

![ddr_R3_mu_resampled_ground_truth.png](figures/readme/ddr_R3_mu_resampled_ground_truth.png)

A plot of the -ZNCC landscape over 2 dimensions (two angular components of transformation) between the fixed image and
the resampled Radon-transformed volume, with the ground truth transformation at the centre:

![landscape.png](figures/readme/landscape.png)

Starting from a different random transformation, optimising the ZNCC between these images over the transformation using
the basin-hopping algorithm:

(specifically `scipy.optimize.basinhopping` with `T=1.0`, `minimizer_kwargs={"method": 'Nelder-Mead'}`)

![rotation_params_against_iteration.png](figures/readme/rotation_params_against_iteration.png)
![translation_params_against_iteration.png](figures/readme/translation_params_against_iteration.png)
![loss_against_iteration.png](figures/readme/loss_against_iteration.png)

The fixed image (= 1/cos^2 alpha * d/ds R2\[cos gamma * g\]) at the converged transformation:

![ddr_R3_mu_resampled_converged.png](figures/readme/ddr_R3_mu_resampled_converged.png)

DRR generated at the converged transformation:

![converged_drr.png](figures/readme/converged_drr.png)

Optimisation completed in 125.508 seconds, performing a total of 45,472 function evaluations.

Here is a plot of the -ZNCC similarity between the fixed image and the resampled moving image against the distance in
SE3 between the transformation and the ground truth transformation for 1000 random transformations:

![loss_vs_distance.png](figures/readme/loss_vs_distance.png)

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