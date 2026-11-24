Reading and writing data.

| Module              | Purpose                                                                              |
|---------------------|--------------------------------------------------------------------------------------|
| `image`             | Reads X-ray DICOM files, and cached synthetic X-rays                                 |
| `sitk`              | Finds, loads and saves CT series using SimpleITK                                     |
| `volume`            | Loads CT volumes using a set of format-specific loaders                              |
| `large`             | Streams a stack of slices that is too big for memory into a `zarr` array             |
| `stradview`         | Reads landmarks exported from StradView                                              |
| `save_data`         | Versioned storage: snapshots plus a log of changes                                   |
| `caching`           | Cache of expensive results, found by matching parameters                             |
| `serialize`         | Conversion between `traitlets` objects and plain JSON-compatible data                |
| `helpers`           | Hashing, and listing of slice files                                                  |
| `command_line`      | Prompting for input until it is valid                                                |
| `external_datasets` | Loaders for public datasets                                                          |

`_old_volume` is kept for reference and is not used.

## Loading images

`image.read_dicom` returns the pixel data as a `float32` tensor, the pixel spacing as (x, y), the scene geometry and the
UID. The spacing is taken from `ImagerPixelSpacing` with the source-to-detector distance if present, and otherwise from
`PixelSpacing` with the source-to-patient distance.

There are two ways to load a CT. The current pipelines use `sitk`, then convert to attenuation with
`reg23_experiments.ops.ct.convert_ct_to_mu_sitk`. The `volume` subpackage is used by tools that need more formats.

## Versioned save data

`SaveState` keeps a DataFrame in memory and on disk. Each change is appended to `log.jsonl`. After a set number of changes
(32 by default), a new directory, named with a timestamp, is created with a full `snapshot.parquet` and an empty log. On
startup, the latest snapshot is loaded and its log replayed. Old directories are kept, so earlier states can be recovered
with `SaveConfig.load_specific_save`. The classes in `reg23_experiments.data.save_data` supply the schema.

## Caching

`save_to_cache` and `load_from_cache` store pickled values in `cache/<type name>/<timestamp>/`, next to a `params.yaml`. A
cached value is used when its parameters equal those requested, so `params` must include everything the value depends on.

## Serialisation

`serialize_recursive` and `deserialize_recursive` walk nested `traitlets` objects, lists and dicts. Deserialising can update
an existing object in place, which lets the app restore what it can from an older settings file.
