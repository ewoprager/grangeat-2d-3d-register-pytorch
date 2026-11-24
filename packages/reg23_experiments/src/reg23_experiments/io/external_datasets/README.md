# External datasets

Loaders for public datasets used for comparison with other work. Each subpackage covers one dataset, and its README says
where to download it and how its coordinate system relates to ours.

| Subpackage                        | Dataset                                                          |
|-----------------------------------|------------------------------------------------------------------|
| [gold_hip](gold_hip/README.md)    | Gold-standard 2D-3D registration dataset of the hip joint        |

`gold_hip` imports code from the downloaded archive, so it can only be imported once that archive is in place. Its pose
conversion is unfinished: `load_xray` currently searches over candidate conversions and then exits the program.

This package is left out of the generated API documentation because it cannot be imported without the downloads.
