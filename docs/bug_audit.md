# Bug audit, 2026-11-24

Covers everything except the `reg23_app` GUI, at commit `a61adb7`. Nothing listed here has been fixed.

**Evidence**

- **Run**: reproduced by running the code on a CPU build (macOS arm64, PyTorch 2.7.1). `zarr`, `pydantic` and `pyswarms` were
  replaced by minimal stand-ins, as they were missing or could not be loaded. Only the save data results (3, 4) depend on
  one (`pydantic`).
- **Read**: found by reading the code. There was no GPU, so nothing in the `.cu` files has been run.

Paths are relative to `packages/reg23_experiments/src/reg23_experiments/` (**E/**) and
`packages/reg23_core/src/` (**C/**).

## High: wrong results, data loss, or a crash on a main path

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 1 | `scripts/program_truncation.py:150,156` | Calls `check_xray_path(path)`, whose arguments are keyword-only and include `ct_series_uid`, so the script always raises `TypeError` here. With that fixed, `E/experiments/reg_experiment/_setup.py:61,66` raises `NameError` (`p`) in place of the intended error. A single X-ray given by file is also stored as a `pathlib.Path`, which `yaml.safe_dump` rejects. | Run (last: Read) |
| 2 | `E/ops/swarm.py:147,158,227` | `_global_best_position` is a view of a particle's current position, so it changes when that particle moves. The reported position then disagrees with the reported value, and the social term pulls towards the wrong point. With 200 particles on a test function, it was wrong in 35 of 600 iterations over the first 6, and the final answer was wrong in 2 of 100 runs. It affects the distance recorded at each iteration of the experiments. Clone it. | Run |
| 3 | `E/io/save_data.py:150` | Snapshot directories have a resolution of one second. Two snapshots in the same second share a directory, and the old log is replayed on top of the new snapshot. Setting 40 electrodes at once gives 72 on reload; 70 gives 134. No duplicates were found in the snapshots in `data/`. | Run |
| 4 | `E/io/save_data.py:169` | Any error while loading, such as a half-written last log line, is logged and replaced by a new, **empty** snapshot, which becomes the latest. Earlier data remain on disk but are no longer loaded. | Run |
| 5 | `E/experiments/reg_experiment/_dadg_updaters/unbatched.py:127,132`; `E/experiments/back_proj_vis/__main__.py:72,98` | Points are placed relative to the image centre using `0.5 * spacing * size`, but points picked in the app have pixel `i` at `i * spacing`, so the centre is `0.5 * spacing * (size - 1)`. With 2 mm voxels and 0.5 mm pixels, projected points were 0.5 mm from the DRR of the same point, and 0.01 mm with `size - 1`. Fiducial registrations, hence gold standards, inherit the offset. | Run |
| 6 | `E/data/sinogram.py:136,143` and `:675` | The spacing of r does not match the grid. `SinogramClassic` uses cell-centred samples but passes the end-point spacing, so `resample` scales r by (N-1)/N. `SinogramHEALPix` is the reverse: `sample`, used by `resample_python`, is wrong, and `resample` is right. With N = 16, r = 59.9 was read as 56.1 and 63.9. | Run |
| 7 | `E/experiments/cached_ops.py:60` | Multiplies spacing (x, y, z) by size (z, y, x) without flipping. For 512 x 512 x 60 voxels of 0.4 x 0.4 x 3 mm, the diagonal comes out as 1550 mm rather than 341 mm, so most of the sinogram is empty. Other shapes can truncate it. Cached VIFs are affected. | Run |
| 8 | `E/ops/grangeat.py:56,59` | The cosine weighting and fixed scaling measure from the image centre and ignore `fixed_image_offset`, although Grangeat's relation needs distances from the foot of the source's perpendicular. The weighting is wrong for cropped or offset images. | Read |
| 9 | `C/backend/cpu/ProjectDRRCPU.cpp:91`; `C/backend/cuda/ProjectDRRCUDA.cu:259` | The gradient with respect to the inverse homography matrix is returned transposed, as the column-major matrix is copied out directly. Against finite differences the error is 49%, or 2% after transposing. The assertion in `test_project_drr_autograd` that would catch this is commented out. | Run (CPU), Read (CUDA) |
| 10 | `C/backend/include/reg23_core/Texture2DCPU.h:47`, `Texture3DCPU.h:47` | `contiguous()` makes a temporary copy of a non-contiguous tensor, and the texture keeps a pointer to it after it is freed. With freed memory overwritten, every permuted volume gave a wrong DRR; without, results happened to be right. `ct_permutation` produces such volumes. | Run |
| 11 | `C/backend/include/reg23_core/CUDATexture.h:26,105` | Members and destructor exist only under `__CUDACC__`, so `main.cpp`, compiled by the host compiler, sees a class of 1 byte with a trivial destructor (checked by compiling it). It allocates 1 byte and the CUDA code writes 24. This is undefined behaviour, and explains why the destructor never runs from Python. Once fixed, `CleanUp` must clear its handles, or it will free them twice. | Run (size), Read (effect) |
| 12 | `C/backend/cuda/SimilarityCUDA.cu:31`; `ProjectDRRCUDA.cu:212` | No `__syncthreads()` between writing the shared buffer and the first reduction step: a data race. | Read |
| 13 | `C/backend/cuda/ObjectiveFunctionCUDA.cu:162` | Sets `denominatorsLeft` twice and never `denominatorsRight`, so an invalid image can give `NaN`. | Read |

## Medium: wrong or failing in particular situations

### Similarity and optimisation

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 14 | `C/backend/cpu/SimilarityCPU.cpp:32`; `E/ops/similarity_metric.py:66,83`; `E/ops/swarm.py:223-227` | ZNCC of a constant, non-zero image can be `NaN`, as rounding makes the variance negative: 84 of 200 such images, and 81 of 200 when weighted. (An all-zero DRR is safe.) In `Swarm`, `min` over a batch containing a `NaN` is `NaN`, so that batch can never improve the global best. The CUDA version has the same form. | Run |

### Data graph (`E/ops/data_manager/`)

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 15 | `_dadg_standalone.py:77`, `_dadg_child.py:54` | If an updater raises, or cleaning fails with `panic_on_failed_eager_evaluation`, `__in_top_level_call` is never restored, and eager evaluation stops for good. Needs `try`/`finally`. | Run |
| 16 | `_dadg_standalone.py:318`, `_dadg_child.py:341` | Adding an updater that creates a cycle recurses until `RecursionError`, because nodes are marked dirty before the check for cycles. | Run |
| 17 | `_dadg_standalone.py:214`, `_dadg_child.py:200` | When a node already has an updater, `add_updater` fails without undoing its changes. It cannot then be added again, and removing it corrupts the existing updater. | Run |
| 18 | `_data.py:20`, `_directed_acyclic_data_graph.py:44` | Copying a node to a child copies the `NoNodeData` marker, so the `is NoNodeData` test fails, and the child returns the marker as a value. | Run |
| 19 | `_dadg_child.py:104` | A child ignores the node's stored `check_equality`, and setting it changes the parent's node. | Run |
| 20 | `_dadg_standalone.py:330`, `_dadg_child.py:353` | Removing an updater sends children the wrong node, so dependents disappear from them too, and they return stale values. | Run |
| 21 | `_namespaces.py:85` | An updater that returns an `Error` raises `TypeError` when captured in a namespace. | Run |
| 22 | `_dadg_standalone.py:387` | `RuntimeError` if an observer or updater adds a node during eager evaluation. | Run |

### Experiments

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 23 | `E/.../_dadg_updaters/drr_reg.py:60` | With no saved configuration for an X-ray, `target_flipped` is missing from the returned values, so the graph reports an error. | Run |
| 24 | `E/.../_dadg_updaters/batched.py:204` | `device == torch.device("cuda")` is false for `cuda:0`, which is what tensors report, so the GPU texture path is never used. | Run (comparison) |
| 25 | `E/.../_dadg_updaters/grangeat.py:125` | Always uses the CUDA texture, so it fails on the CPU. | Read |
| 26 | `E/.../_dadg_updaters/drr_reg.py:132`, `_init_dadg.py:40` | `ct_permutation` permutes spacing (x, y, z) with the indices meant for sizes (z, y, x). | Read |
| 27 | `E/.../_init_dadg.py:63` | Unpacks a possible `Error` as a tuple. | Run |
| 28 | `E/.../_dadg_updaters/batched.py:152`, and similar | Assumes spacing doubles at each pyramid level. Sizes are rounded down, so an axis of 101 becomes 50 and the true factor is 2.02. | Run |
| 29 | `E/.../_dadg_updaters/drr_reg.py` (`set_synthetic_target_image`), `E/ops/drr.py:82` | Synthetic targets cannot work in the current graph: no `xray_sop_instance_uid`, and spacing is `float32`, on the CPU, and ordered (y, x). | Run (spacing), Read |
| 30 | `E/.../_of_together.py:84,138` | Pass weights to metrics by position, but metrics take them by keyword. | Run |
| 31 | `E/experiments/experiment_set_config.py:133` | A config iterable with Cartesian variables is empty the second time it is iterated. | Run |
| 32 | `E/experiments/experiment_set_config.py:189` | Sobol samples are not seeded. With `Range` variables, `--fill-gaps` and the dry run use different values under the same file names. | Read |
| 33 | `E/experiments/registration.py:115` | Periodic crop and weight updates change the objective function, but stored best values are from the old one and are not re-evaluated. | Read |

### Loading and saving

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 34 | `E/io/volume/_dicom.py:87,238` | Rescale slope and intercept are applied here and again in `convert_ct_to_mu`. Water gives 0 rather than 0.02. Used by `add_synth_fiducials.py`. | Run |
| 35 | `E/io/volume/_dicom.py:195,226` | Unevenly spaced slices raise `AttributeError`, and slices without positions raise `IndexError`, instead of using the fallback. | Run |
| 36 | `E/io/volume/_nrrd.py:38` | The array is not reversed to (z, y, x), unlike NIfTI. This is consistent with files written by `add_synth_fiducials.py`, but mirrors ordinary NRRD files. | Run |
| 37 | `E/io/volume/_xtekct.py:68` | Stored as `int16`; unsigned data above 32767 wraps to negative. | Run (conversion) |
| 38 | `E/io/large.py:25` | If streaming stops part way, the partial array is later treated as complete. A missing `.xtekVolume` also leaves a spacing of `None`, which fails when read back. | Read |
| 39 | `E/io/helpers.py:52` | The content hash is cached by directory path only, so it is stale if the files change. | Read |
| 40 | `E/io/caching.py:68` | Two values cached in the same second overwrite each other. Tensors are pickled on their device, so a GPU cache cannot be read without one. | Read |
| 41 | `E/io/sitk.py:71` | A directory with no series raises `ValueError` instead of returning an `Error`. | Run |
| 42 | `E/io/sitk.py` (`save_ct_series`) | File metadata keeps the template's SOP instance UID, values are truncated rather than rounded, and signed data are written whatever the template's `PixelRepresentation`. | Read |
| 43 | `scripts/cylinder_crop.py:157,176` | The saved crop is mirrored in x relative to the preview (30 instead of 70 in a test). The template slice is an arbitrary file from the directory. | Run, Read |
| 44 | `E/utils/data.py:21` | Objects inside lists and dicts are not cloned, so a registration worker shares `xray_parameters` with the interface. | Read |

### Code that always fails

| # | Where | Problem | Evidence |
|---|-------|---------|----------|
| 45 | `E/ops/objective_function.py:22` | `evaluate` multiplies `float64` by `float32`. `evaluate_direct` needs 1D grids, but is used with 2D ones (`E/ops/grangeat.py:100`). | Run |
| 46 | `E/data/sinogram.py:128` | `SinogramClassic.grid` passes an argument that does not exist. So does `scripts/draw_sinogram_grid.py`. | Run |
| 47 | `E/experiments/script.py`, `E/data/plot_data.py` | Cannot be imported (`pre_computed` and `SinogramType` no longer exist). | Run |
| 48 | `E/ops/autograd_impl.py` | The backward pass is wrong (sign, and a missing identity matrix); it disagrees with finite differences in every rotation component. | Run |
| 49 | `E/ops/cmaes.py:80` | Multiplies mismatched shapes. Also uses sigma rather than sigma squared, a wrong `h_sig`, and a wrong way of making the covariance symmetric. Unused. | Run, Read |
| 50 | `E/ops/fiducials.py:92` | `refine_spherical_fiducial_2d` raises if the marker is outside the image, as the returned loss has no gradient. | Run |
| 51 | `E/ops/optimisation.py:84` | `pso` calls `iteration_callback` even when it is `None`, which is the default. | Read |
| 52 | 21 of the 29 files in `scripts/` | Import modules or names that no longer exist; see [scripts/README.md](../scripts/README.md). | Run |
| 53 | `justfile:19,22` | `runapp` and `devapp` run `scripts/app.py`, which does not exist. | Run |
| 54 | `E/io/external_datasets/gold_hip/_image.py:137` | `load_xray` always searches and then calls `exit(0)`. | Read |

## Low

| Where | Problem | Evidence |
|-------|---------|----------|
| `C/.../Common.h:119`, `E/data/structs.py:476` | Unflipping is wrong when both angles are out of range. Current callers only reach this for exactly vertical planes. | Run |
| `C/.../Radon2D.h:122`, and the 3D equivalent | Scaled by L/N, but samples are L/(N-1) apart: 4% low for 50 samples, 0.2% for 400. | Run |
| `C/backend/cuda/Radon3DCUDA.cu:155,244` | `threadIdx.y` should be `threadIdx.y == 0`. Usually harmless, but a block with one valid row is lost. | Read |
| `C/backend/cuda/Radon3DCUDA.cu:175,265`; `ProjectDRRCUDA.cu` backward | Reductions assume a power of two (and a square, for 3D), which is not enforced, so samples can be dropped. | Read |
| `C/backend/cuda/ObjectiveFunctionCUDA.cu:74` | Sums are lost if the last block has fewer than 6 pixels. | Read |
| `C/reg23_core/ops.py:110,129` | Masks do not convert `output_offset` to `float64` as DRRs do, so a `float32` offset raises. `project_drr_backward` has the same gap when called directly. | Run |
| Mask and sinogram operators | Outputs use the default dtype, so they fail after `torch.set_default_dtype(torch.float64)`. | Run |
| `grid_sample3d` | Raises for non-contiguous grids (`view` instead of `reshape`). | Run |
| Kernel launches | Grid sizes are computed as `int` and can overflow for very large batches. | Read |
| `E/ops/geometry.py:243` | `generate_drr_python` reads detector spacing as (y, x). Wrong for non-square pixels. | Run |
| `E/ops/similarity_metric.py:152,156,230` | Weighted gradient metrics fail for `float64`. | Run |
| `E/ops/optimisation.py:52` | `local_search` modifies the caller's step size. | Run |
| `E/ops/swarm.py` | Fails with one particle, which `PsoParameters` allows. The docstring example uses old names. | Run |
| `E/ops/fiducials.py:253` | Tests convergence with the sum of the step, so opposite components can cancel. | Read |
| `E/experiments/helpers.py:53` | `bandpass` is recorded as `highpass` in the cache key (no collision, but misleading). | Run |
| `E/io/volume/_nii.py:25` | Wrong argument name, so the series is listed under an empty key. | Run |
| `E/utils/logs_setup.py:18` | `AttributeError` when two runs start in the same second. | Read |
| `E/analysis/format.py:66` | `latex_escape` escapes the braces that it has just added for a backslash. | Run |
| `E/analysis/structs.py:184` | `CustomFitPowerRatio.generate_ys` leaves out the y offset. | Read |
| `E/analysis/manipulation.py:178` | `KeyError` if a Cartesian or dependent column is constant. | Read |
| `Cropping` | The offset uses unrounded edges, but pixels are cropped at rounded ones: up to half a pixel. | Read |

## Packaging, tests and documentation

- `tifffile` is imported by `E/io/volume` (and so by `E/ops/ct.py` and `reg_experiment`) but is not a declared dependency. In
  `uv.lock` it comes only through `napari` and `scikit-image`.
- `packages/reg23_core/Docs/Doxyfile:952` lists `../include/reg23`, which has moved.
- `docs.yml` runs `pdoc` without `--mermaid`, so the graph diagrams are not rendered.
- The README refers to `reg23/` paths and `scripts/app.py`-era layouts in places, and the StandaloneDADG docstring
  to methods that have been renamed.
- `Conventions.md` says y points up as seen from the source. In the DRR code, the row index increases with y, so in images drawn
  with row 0 at the top, y points down. `get_crop_full_depth_drr` assumes this.
- `test_fiducials` fails on macOS arm64: a 1.6 micron error is compared with a tolerance of 1e-12. The other 22 tests pass.
- Four tests run code without asserting on the main result: `test_project_drr_autograd`, `test_objective_function`,
  `test_transformation_matrix_custom` and `test_swarm`. All four cover code with a bug above.

## Suspected, not confirmed

- CUDA textures are destroyed as soon as a kernel is launched, and kernels are asynchronous. This is safe only if destruction
  waits for the kernel.
- `mutual_information` of a constant image converts `NaN` to an integer, whose result depends on the platform. It gave 0 on
  arm64, but may give an index out of range on x86.
