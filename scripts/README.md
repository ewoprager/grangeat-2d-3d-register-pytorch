# Scripts

Run from the repository root:

```bash
uv run --extra <platform> --extra dev scripts/<script>.py <args...>
```

Use `--help` to list arguments, where a script takes any.

The *Imports* column says whether the script's imports of this project's own packages resolve, as of 2026-11-24. Scripts marked
**no** were written against earlier package layouts (`registration`, `notification`, `reg23_experiments.registration`,
`reg23_experiments.ui`) or use names that have since been removed. They are kept as a record of past experiments.

## Experiments

| Script                        | Purpose                                                                                 | Imports |
|-------------------------------|-----------------------------------------------------------------------------------------|---------|
| `program_truncation.py`       | The main experiment: registration accuracy over a set of configurations                 | yes     |
| `analysis.py`                 | Angles between the gold-standard poses of a set of X-rays                               | **no**  |
| `register.py`                 | One Grangeat-based registration, with plots                                             | **no**  |
| `landscapes.py`               | Objective function landscapes around the gold standard                                  | **no**  |
| `truncation.py`               | Earlier study of the effect of truncation                                               | **no**  |
| `truncation_theory_sandbox.py` | Numerical exploration behind the truncation study                                      | **no**  |
| `compare_smoothed_sampling.py` | Sinogram resampling with and without sample smoothing                                  | **no**  |
| `interface.py`                | The GUI that came before `reg23_app`                                                    | **no**  |

### `program_truncation.py`

The configuration is written in the script itself, as an `ExperimentSetConfig`. Give it a directory containing a CT series, and
an X-ray file or directory:

```bash
just experiment <CT directory> <X-ray directory> [name]
```

Results go to `experimental_results/program_truncation/<timestamp>[_<name>]/`, as one parquet file per configuration, with the
configuration itself in `variables.txt`. A dry run of every configuration comes first. `--fill-gaps` continues an interrupted
run, `--show` displays the images for the first configuration, and `--notify` sends a notification when finished.

## Data preparation

| Script                    | Purpose                                                                                     | Imports |
|---------------------------|---------------------------------------------------------------------------------------------|---------|
| `add_synth_fiducials.py`  | Adds synthetic ball bearings to a CT, at landmarks exported from StradView                  | yes     |
| `cylinder_crop.py`        | Interactive elliptical crop of a CT series, saved as a new DICOM series                     | yes     |
| `weight_var_update.py`    | Converts old results that used `apply_weighting` to `weighting_method`                      | yes     |
| `tif_ct_test.py`          | Prints metadata from a directory of XTek `.tif` slices                                      | yes     |

## Sinogram development

| Script                  | Purpose                                                              | Imports |
|-------------------------|----------------------------------------------------------------------|---------|
| `draw_sinogram_grid.py` | 3D plot of sample positions in the two sinogram layouts              | yes     |
| `dev_sinogram.py`       | Checks of the HEALPix mapping and sinogram values                    | **no**  |
| `debug_tests.py`        | Runs one test under a debugger                                       | **no**  |

## `benchmarking/`

Timing of each operator and of whole registrations. None currently run.

| Script                             | Measures                                                         | Imports |
|------------------------------------|------------------------------------------------------------------|---------|
| `benchmark_radon2d.py`             | 2D Radon transforms                                              | **no**  |
| `benchmark_radon3d.py`             | 3D Radon transforms                                              | **no**  |
| `benchmark_resample_sinogram3d.py` | Sinogram resampling                                              | **no**  |
| `benchmark_similarity.py`          | ZNCC and its gradient                                            | **no**  |
| `benchmark_moving_image.py`        | DRR projection against sinogram resampling                       | **no**  |
| `benchmark_register.py`            | Whole registrations                                              | **no**  |

## `plotting/`

| Script                             | Plots                                                                    | Imports |
|------------------------------------|--------------------------------------------------------------------------|---------|
| `plot_program_truncation.py`       | Results of `program_truncation.py`                                       | yes     |
| `geometric_weighting.py`           | Figures for particular result directories, whose paths are in the script | yes     |
| `fit_program_truncation.py`        | Fitted models of accuracy against truncation                             | **no**  |
| `program_truncation_data_summary.py` | Text summary of a directory of results                                 | **no**  |
| `plot_truncation.py`               | Results of `truncation.py`                                               | **no**  |
| `plot_landscapes.py`               | Results of `landscapes.py`                                               | **no**  |
| `plot_benchmark_register.py`       | Results of `benchmark_register.py`                                       | **no**  |
| `plot_drr_grangeat_comparison.py`  | Results of `benchmark_moving_image.py`                                   | **no**  |
