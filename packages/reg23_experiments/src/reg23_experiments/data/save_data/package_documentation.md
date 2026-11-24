Managers for annotations made by hand, such as segmented points and gold-standard transformations. Each stores a
`pandas.DataFrame` in a subdirectory of `data/`, using the versioning scheme in `reg23_experiments.io.save_data`.

| Module            | Stores                                        | Index                       | Columns               |
|-------------------|-----------------------------------------------|-----------------------------|-----------------------|
| `transformations` | Named transformations between images          | source UID, dest. UID, name | `x0` ... `x5`         |
| `xray_regs`       | Horizontal flip and cropping for an X-ray     | X-ray UID                   | flip and four edges   |
| `xray_fiducials`  | Named fiducial markers in X-rays              | X-ray UID, name             | `x`, `y`              |
| `ct_fiducials`    | Named fiducial markers in CT or micro-CT      | CT UID, name                | `x`, `y`, `z`         |
| `xray_electrodes` | Ordered electrode positions in X-rays         | X-ray UID, index            | `x`, `y`              |
| `ct_electrodes`   | Ordered electrode positions in CT             | CT UID, index               | `x`, `y`, `z`         |

Positions are in mm, as used by the image layers of the app. Transformations are stored as an axis-angle rotation in radians
followed by a translation in mm. X-rays are identified by `SOPInstanceUID`, CT scans by series UID.

## Common structure

Every module has the same parts:

1. `pydantic` models for each kind of change, combined into a `Change` with an `action` discriminator. A change is one line
   of JSON in the log.
2. `_incorporate_change`, which applies a change to the DataFrame.
3. For point sets, a function that compares the old and new points and returns the changes between them.
4. A `...SaveManager` class, which is the only part used elsewhere. Reading returns `None` or an `Error` when nothing is
   stored.

Electrodes are ordered, so points are only added to and removed from the end. Fiducials are matched by name, so that the
same marker can be found in different images.

## Adding a new kind of save data

Copy an existing module, change the models and columns, and document the schema in the module docstring. Files already on
disk are read with the current models, so a change to an existing schema needs a migration of the stored files.
