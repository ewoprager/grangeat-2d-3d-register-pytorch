# Data

Annotations and results that are kept under version control. Run scripts from the repository root, as they refer to this
directory by relative path.

## Save data

Each directory holds one table of manual annotations, with a full history. The format is described in
[io](../packages/reg23_experiments/src/reg23_experiments/io/package_documentation.md#versioned-save-data), and the columns in
[data.save_data](../packages/reg23_experiments/src/reg23_experiments/data/save_data/package_documentation.md).

| Directory                      | Contents                                                             | Manager                    |
|--------------------------------|----------------------------------------------------------------------|----------------------------|
| `app_transformation_save_data` | Named transformations, including the `gold_standard` ones            | `TransformationSaveManager` |
| `xray_reg_save_data`           | Flip and cropping for each X-ray                                     | `XRayRegSaveManager`       |
| `xray_fiducial_save_data`      | Fiducial markers in X-rays                                           | `XRayFiducialSaveManager`  |
| `ct_fiducial_save_data`        | Fiducial markers in CT volumes                                       | `CTFiducialSaveManager`    |
| `app_electrode_save_data`      | Electrodes in X-rays                                                 | `XRayElectrodeSaveManager` |
| `ct_electrode_save_data`       | Electrodes in CT volumes                                             | `CTElectrodeSaveManager`   |

Each contains directories named by timestamp, each with a `snapshot.parquet` and the `log.jsonl` of later changes. The latest is
the current state. Change them through the managers rather than by hand.

## Benchmark results

| File                           | Contents                                                              |
|--------------------------------|-----------------------------------------------------------------------|
| `drr_vs_grangeat.pkl`          | Times for DRR projection against sinogram resampling                  |
| `register_plot_data.pkl`       | Times and accuracy of whole registrations                             |
| `sinogram2d_evaluation_times.txt`, `sinogram3d_evaluation_times.txt` | Times to compute sinograms      |

The pickles hold classes from `reg23_experiments.data.plot_data`, at the module path used when they were saved.

`data/temp/` is for scratch output and is not tracked.
