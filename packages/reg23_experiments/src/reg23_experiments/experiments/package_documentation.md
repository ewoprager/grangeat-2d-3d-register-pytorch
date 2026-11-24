Everything needed to define, run and store registration experiments, plus the registration pipelines that they and the app
share.

| Module or package       | Purpose                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------|
| `experiment_set_config` | Describes the set of configurations to run                                               |
| `run_experiments`       | Runs an experiment function for each configuration and saves the results                 |
| `registration`          | Runs one registration with particle swarm optimisation                                   |
| `reg_experiment`        | The registration accuracy experiment, and the data graph updaters behind it              |
| `cached_ops`            | Expensive operations with a disk cache                                                   |
| `helpers`               | Similarity metrics by name, output directories                                           |
| `electrode_experiment`  | Updaters for locating cochlear implant electrodes                                        |
| `sim_met_figures`       | Updaters that show where in the image a similarity comes from                            |
| `back_proj_vis`         | Runnable 3D view of back-projected electrodes                                            |
| `script`                | Out of date: helper for older scripts                                                    |

## Defining a set of configurations

`ExperimentSetConfig` maps each variable name to one of:

| Class       | Meaning                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------|
| `Constant`  | One value throughout                                                                              |
| `Cartesian` | A list of values; every combination with other Cartesian variables is run                         |
| `Zipped`    | A list of values that advances together with all other zipped variables (at least two)            |
| `Range`     | A continuous, integer or listed range, sampled together with other ranges using a Sobol sequence  |

`iterable()` yields a name and a dictionary for each configuration. The total is the product of the number of Cartesian
combinations, zipped rows and range samples. Names such as `c0-1_s3_z2` give the index within each group, and become file
names. Make a new iterable for each pass. The configuration can be saved and reloaded with `serialize` and `deserialize`.

## Running experiments

`run_experiments` takes an *experiment* function, which receives an `ExperimentConfig` and returns a DataFrame (or `None` if
there is nothing to do), and a constructor that turns a dictionary into the experiment's parameters. For each configuration
it skips any whose result file exists, runs the experiment, adds the configuration as extra columns, and writes
`data_<name>.parquet`. An exception in one configuration is logged, and the rest continue unless `throw` is set. A dry run
checks that every configuration can be evaluated before a long run begins.

## Running one registration

`register` runs a swarm from given starting parameters, and returns the best position and value at each iteration. Periodic
behaviours are callbacks that run every N iterations with the best position so far. They are used to update cropping and
weighting as the estimate improves.

## Cached operations

`cached_calculate_vif` computes the sinogram of a CT volume. The cache key covers the series, the size and type of sinogram,
the range of r, and any downsampling, filtering and truncation.
