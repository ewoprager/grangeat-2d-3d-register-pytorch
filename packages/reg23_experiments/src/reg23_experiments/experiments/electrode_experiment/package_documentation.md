Data graph updaters for locating cochlear implant electrodes, using a micro-CT scan to provide accurate positions.

| Updater                       | Reads                                          | Writes                                |
|-------------------------------|------------------------------------------------|---------------------------------------|
| `get_micro_ct_uid`            | path to one slice                              | a UID derived from the volume contents |
| `get_micro_ct_array`          | path, UID, path to the `zarr` store            | the volume as a `zarr` array, spacing |
| `transform_micro_fiducials`   | micro-CT to CT transformation, micro fiducials | fiducials in CT space                 |
| `transform_micro_electrodes`  | micro-CT to CT transformation, micro electrodes | electrodes in CT space               |
| `backproject_direct`          | electrodes and transformations for two X-rays  | placeholder; not yet implemented      |

A micro-CT volume is usually too big for memory, so it is streamed once into a `zarr` store and read in chunks from then on.
Its UID is a hash of the contents, since the slices carry no DICOM UID. The hash is remembered per directory.

The micro-CT and clinical CT are aligned using fiducial markers that are segmented in both. The electrodes marked in the
micro-CT can then be mapped into the clinical CT, as a reference for positions estimated from X-rays.
