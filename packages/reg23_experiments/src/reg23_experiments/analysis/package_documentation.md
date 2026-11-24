Tools for turning experiment results into fits, tables and figures. Used mainly by the scripts in `scripts/plotting/`.

| Module         | Purpose                                                                            |
|----------------|------------------------------------------------------------------------------------|
| `fit`          | Least-squares polynomial and power-law fitting with `torch`                        |
| `structs`      | Objects representing fits to a data series                                         |
| `manipulation` | Conversion of results from `pandas` DataFrames into tensors                        |
| `format`       | Number, variable-name and LaTeX formatting, plus matplotlib defaults               |
| `helpers`      | Export of colour maps in a form LaTeX can plot                                     |
| `plot`         | Figure helpers, and a 3D view of sinogram planes as points                         |

## Fits

`DataFit` is the common interface: it provides a text equation, generates fitted values, and plots them. Implementations
are `LinearFit`, `QuadraticFit`, `PowerFit` (a linear fit in log-log space, built with `PowerFit.build`, which returns
`None` if the data are not all positive) and `CustomFitPowerRatio`. The data are passed as a `Series`, which records
whether the curve passes through the origin and any vertical offset to remove before fitting.

## Results as tensors

`reg23_experiments.experiments.run_experiments` writes one parquet file per configuration. Concatenated, these form a long
table with a column per variable. `dataframe_to_cartesian_zipped_tensors` converts that table to a
`CartesianZippedTensors`, which holds:

- a tensor per dependent variable, with one axis per *Cartesian* variable and, if present, a final axis for the *zipped*
  variables, which vary together;
- the values along each axis; and
- the variables that were constant.

`CartesianZippedTensors.collapse` removes an axis by taking the last value (e.g. the final iteration) or the mean. These
terms match the configuration classes in `reg23_experiments.experiments.experiment_set_config`.

Every variable named as Cartesian or dependent must take more than one value in the table, as constant columns are removed.
