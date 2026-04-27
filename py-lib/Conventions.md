# Conventions

## Geometry

### CT coordinate system

A CT volume can be loaded from a single file of one the following types:

- .nrrd,
- .nii,

or from a DICOM series (specified by "SeriesNumber") in directory of DICOM files.

The world coordinates $\mathbf{x}_{i,j,k}$ of the i,j,kth voxel is given by:
$$
\mathbf{x}_{i,j,k} = \mathbf{origin} + \begin{bmatrix}
i \cdot \mathbf{spacing}_x \\
j \cdot \mathbf{spacing}_y \\
k \cdot \mathbf{spacing}_z \\
\end{bmatrix}
$$
where $\mathbf{origin}$ is the position of the voxel (0, 0, 0) relative to the origin of the world coordinate system,
and $\mathbf{spacing}$ is the size of the voxel in millimetres.

$\mathbf{origin}$ is consistent with the convention for the header field "space origin" in the .nrrd file format.

If a series of DICOM files is loaded, the $\mathbf{origin}$ is set to the value of the header field
"ImagePositionPatient" of the slice for which the $z$-component of "ImagePositionPatient" is closest to $-\infty$.

## Code style

### Class definitions

Fields in a class are declared in the following order:

1. class variables
2. static methods
3. class methods
4. `__init__` method
5. `__setstate__` and `__getstate__` methods
5. properties
6. normal instance methods

### Importing

Standard library imports are together at the top.

After a space, third party library imports.

After a further space, internal imports.

Importing all names from a module with `import *` is generally avoided.

### Module layout

The contents of an importable module are generally in the following order:

1. Imports
2. `__all__ = [...]` with a list of all importable members
3. `logger = logging.getLogger(__name__)`, if logging is desired
4. Class and function definitions
5. If the module can also be run as a script:
    1. `def main(...): ...`; the script body
    2. `if __name__ == "__main__": ...`; the script setup, e.g. reading arguments; calls `main()`.

### Script layout

The contents of a script are generally in the following order:

1. Imports, generally including

```python
import argparse

from reg23_experiments.notification import logs_setup
```

2. Class and function definitions
3. `def main(...): ...` the script body
4. The script setup, which generally looks like this:

```python
if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    ...

    main( < argument >= args. < argument >, ...)
```

### Commonly used imports

- `pathlib` is generally used for manipulating path strings, files and directories
- `argparse` is used for reading script arguments
- `tqdm` is used for progress bars; the custom logger is set up to work with it
- `traitlets` is used for configuration structs and runtime type-checked values