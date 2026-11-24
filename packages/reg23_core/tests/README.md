# reg23_core tests

Run from the repository root, with the `dev` extra installed:

```bash
python -m pytest packages/reg23_core/tests
```

| File                         | Covers                                                                                    |
|------------------------------|-------------------------------------------------------------------------------------------|
| `test_grid_sample3d.py`      | Shape checks, input validation, CPU against CUDA, comparison with `grid_sample` in PyTorch |
| `test_project_drr.py`        | DRRs and masks: validation, detector offset, CPU against CUDA, batched against unbatched   |
| `test_ncc.py`                | ZNCC value and its gradient, checked against finite differences                           |
| `test_cudatexture.py`        | Texture construction, validation and lifetime (CUDA only)                                  |
| `test_objective_function.py` | Runs the fused objective function beside its Python equivalent (CUDA only)                 |

CUDA tests are skipped or shortened when no GPU is present, which includes CI. Run the suite on a GPU machine after any
change to a `.cu` file.

`test_project_drr_autograd` and `test_objective_function` currently run the code without asserting on the result. The Radon
and sinogram resampling operators are not tested in this package; `radon3d` is covered in the `reg23_experiments` tests.
