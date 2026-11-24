Data structures shared by the whole library.

| Module         | Purpose                                                                                |
|----------------|----------------------------------------------------------------------------------------|
| `structs`      | Geometry, transformations, croppings, sinogram grids, errors, the optimiser interface  |
| `sinogram`     | 3D sinograms of CT volumes, and how to sample them                                     |
| `parameters`   | Configuration of the app and of registration, as `traitlets` classes                   |
| `segmentation` | Sets of 2D and 3D points, either ordered (electrodes) or named (fiducial markers)      |
| `plot_data`    | Containers for pickled benchmark results                                               |
| `save_data`    | Versioned storage of manual annotations                                                |

`_convert_path_to_uid` is a one-off script that converted old save data from file paths to DICOM UIDs.

## Structs

- `Transformation`: an axis-angle rotation and a translation in mm, both `torch.float64`. Provides the homogeneous matrix
  (`get_h`), inversion, a distance measure between two transformations, and random perturbation at a chosen distance.
- `SceneGeometry`: source distance and detector offset, and the projection matrix onto the detector plane.
- `Cropping`: a rectangle given as fractions of an image, with right and down positive. Its values are validated so that the
  rectangle stays inside the image and never collapses.
- `LinearRange` and `LinearMapping`: a 1D interval, and conversion between intervals. A range can generate grids that
  either include its end points (`generate_grid`) or sit at cell centres (`generate_tex_coord_grid`). The spacing
  function must match the grid: `get_spacing` and `get_tex_coord_spacing` respectively.
- `Sinogram2dGrid` and `Sinogram3dGrid`: coordinates at which to evaluate sinograms. `Sinogram2dGrid.shifted` moves a grid to
  account for an off-centre detector. `Sinogram3dGrid.unflip` brings angles into the range used for storage.
- `Error`: returned in place of a value when an operation fails in an expected way.
- `GrowingTensor`: a tensor that can be appended to.
- `OptimisationInstance`: the interface of a running optimisation, which can be stepped and queried for its best result.

## Sinograms

A `Sinogram` holds the derivative of the 3D Radon transform of a CT volume, sometimes called a *VIF* (volume intermediate
function). There are two layouts:

- `SinogramClassic`: an even grid in (phi, theta, r).
- `SinogramHEALPix`: spheres of evenly distributed orientations, so no samples are wasted near the poles.

Each can generate its own grid (`build_grid`), and pads itself on construction so that interpolation is continuous across
the edges of the parametrisation. `resample` uses the compiled operator, `resample_cuda_texture` uses a texture held on the
GPU, and `resample_python` is a reference implementation.

## Parameters

`Parameters` describes everything the user can configure in the app. Traits tagged `ui=True` appear in the generated
interface. Changing `optimisation_algorithm` or `sim_metric` swaps in a matching parameter struct, and remembers the
previous one in case the user switches back. `Context` bundles parameters with a data graph and a namespace, so that a
worker thread can be handed one self-contained object.
