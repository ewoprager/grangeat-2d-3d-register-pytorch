Loading of CT volumes from several file formats behind one interface.

## Interface

- `find_ct_series` lists the series available at a path.
- `load_ct_series` loads one of them.
- `load_one_ct_series` loads the first that is found.
- `get_input_ct_series_choice` asks the user to choose.

The result is a `Volume`, holding the data with dimensions (z, y, x), the voxel spacing in mm as (x, y, z), an identifier, the
DICOM rescale values and the position of the first voxel if known. Convert it to attenuation coefficients with
`reg23_experiments.ops.ct.convert_ct_to_mu`, which applies the rescale values.

## Loaders

| Loader               | Input                                              | Series per path |
|----------------------|----------------------------------------------------|-----------------|
| `NiiVolumeLoader`    | a `.nii` file                                      | one             |
| `NrrdVolumeLoader`   | a `.nrrd` file, first array axis taken as z        | one             |
| `DICOMVolumeLoader`  | a directory of `.dcm` files (or with no extension) | many            |
| `XTekCTVolumeLoader` | a directory of `.tif` slices with a `.xtekVolume`  | one             |

Each derives from `VolumeLoader` and is tried in turn. A loader that can hold only one series reports `OneSeries`; the others
return a description of each series by UID.

The DICOM loader groups slices by in-plane position, keeps the largest group, sorts by z, removes duplicate positions and
checks that the spacing is even.

## Adding a format

Derive from `VolumeLoader` in a new module, return nothing from `series_available` for paths that do not match, and add the
class to `_registry` in `_public.py`.
