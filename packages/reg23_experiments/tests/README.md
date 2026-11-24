# reg23_experiments tests

Run from the repository root, with the `dev` extra installed:

```bash
python -m pytest packages/reg23_experiments/tests
```

| File                                | Covers                                                                                |
|-------------------------------------|---------------------------------------------------------------------------------------|
| `test_data/test_structs.py`         | Runs the custom transformation-to-matrix autograd function beside the kornia version   |
| `test_ops/test_data_manager.py`     | Lazy evaluation in a graph, and independence of a child graph from its parent          |
| `test_ops/test_fiducials.py`        | Refinement of a synthetic ball-bearing marker in 3D and 2D                             |
| `test_ops/test_geometry.py`         | Coordinate conversions, DRRs and masks against Python versions, plane integrals        |
| `test_ops/test_similarity_metric.py` | ZNCC against `reg23_core`, and behaviour of the other metrics                         |
| `test_ops/test_swarm.py`            | Runs a particle swarm optimisation                                                     |

`test_structs.py` and `test_swarm.py` run the code without asserting on the result. There are no tests yet for `io`,
`experiments`, `analysis`, the save data managers or the sinogram classes.
