# Documentation map

Start with the [project README](../README.md) for setup and the app, then [architecture.md](architecture.md) for how the
parts fit together.

| Topic                                              | Document                                                                                             |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| How the repository fits together                   | [architecture.md](architecture.md)                                                                   |
| Building, testing, CI, publishing docs, Docker     | [development.md](development.md)                                                                     |
| Known bugs, as of 2026-11-24                       | [bug_audit.md](bug_audit.md)                                                                         |
| **C++/CUDA extension**                             | [reg23_core README](../packages/reg23_core/README.md)                                                |
| Backend structure                                  | [backend_overview.md](../packages/reg23_core/src/backend/backend_overview.md)                        |
| What each operator computes                        | [operators_overview.md](../packages/reg23_core/src/backend/operators_overview.md)                    |
| Python wrappers                                    | [python_overview.md](../packages/reg23_core/src/reg23_core/python_overview.md)                       |
| C++ and geometry conventions                       | [Conventions.md](../packages/reg23_core/Conventions.md)                                              |
| **Research library**                               | [reg23_experiments README](../packages/reg23_experiments/README.md)                                  |
| Data structures                                    | [data](../packages/reg23_experiments/src/reg23_experiments/data/package_documentation.md)            |
| Manual annotations                                 | [data.save_data](../packages/reg23_experiments/src/reg23_experiments/data/save_data/package_documentation.md) |
| Operations                                         | [ops](../packages/reg23_experiments/src/reg23_experiments/ops/package_documentation.md)              |
| Data graph                                         | [ops.data_manager](../packages/reg23_experiments/src/reg23_experiments/ops/data_manager/package_documentation.md) |
| Experiments                                        | [experiments](../packages/reg23_experiments/src/reg23_experiments/experiments/package_documentation.md) |
| Registration experiment and its graph              | [experiments.reg_experiment](../packages/reg23_experiments/src/reg23_experiments/experiments/reg_experiment/package_documentation.md) |
| Loading and saving                                 | [io](../packages/reg23_experiments/src/reg23_experiments/io/package_documentation.md)                |
| CT volume loaders                                  | [io.volume](../packages/reg23_experiments/src/reg23_experiments/io/volume/package_documentation.md)  |
| External datasets                                  | [io.external_datasets](../packages/reg23_experiments/src/reg23_experiments/io/external_datasets/README.md) |
| Analysis of results                                | [analysis](../packages/reg23_experiments/src/reg23_experiments/analysis/package_documentation.md)    |
| Helpers                                            | [utils](../packages/reg23_experiments/src/reg23_experiments/utils/package_documentation.md)          |
| Python conventions                                 | [Conventions.md](../packages/reg23_experiments/Conventions.md)                                       |
| **Scripts**                                        | [scripts](../scripts/README.md)                                                                      |
| **Stored data**                                    | [data](../data/README.md)                                                                            |

The `reg23_app` GUI is described in the [project README](../README.md#the-app).
