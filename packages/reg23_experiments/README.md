# reg23_experiments

The research library for 2D/3D registration. It builds on the operators in [reg23_core](../reg23_core) and is used by the
app ([reg23_app](../reg23_app)) and by the [scripts](../../scripts/README.md).

Coding and geometry conventions are in [Conventions.md](Conventions.md). API documentation is generated with `pdoc`; see
[development.md](../../docs/development.md).

## Subpackages

| Subpackage                                                                  | Purpose                                                        |
|-----------------------------------------------------------------------------|----------------------------------------------------------------|
| [analysis](src/reg23_experiments/analysis/package_documentation.md)         | Curve fitting, plotting and reshaping of experiment results    |
| [data](src/reg23_experiments/data/package_documentation.md)                 | Core data structures, and [save data](src/reg23_experiments/data/save_data/package_documentation.md) for manual annotations |
| [experiments](src/reg23_experiments/experiments/package_documentation.md)   | Experiment configuration, execution and registration pipelines |
| [io](src/reg23_experiments/io/package_documentation.md)                     | Loading images and volumes, caching, serialisation             |
| [ops](src/reg23_experiments/ops/package_documentation.md)                   | Operations on data, including the [data manager](src/reg23_experiments/ops/data_manager/package_documentation.md) |
| [utils](src/reg23_experiments/utils/package_documentation.md)               | Logging, notifications, reflection                             |

## Dependencies between subpackages

`utils` and `data.structs` sit at the bottom, `ops` and `io` build on them, and `experiments` sits on top. The
layering is not strict: `data.sinogram` uses `ops.geometry`, for example. `analysis` works on results rather than on
the pipeline, apart from one plotting helper used by `data.sinogram`.

## Error handling

Functions that can fail in expected ways return `T | Error`, where `Error` is a small class in `data.structs` holding a
description. Callers check with `isinstance(result, Error)`. Exceptions are kept for programming mistakes and for failures
that should stop the program.

## Tests

See [tests/README.md](tests/README.md).
