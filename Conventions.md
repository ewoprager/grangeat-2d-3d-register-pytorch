# Conventions

## Class definitions

- Fields in a class are declared in the following order:
  1. class variables
  2. static methods
  3. class methods
  4. `__init__` method
  5. `__setstate__` and `__getstate__` methods
  5. properties
  6. normal instance methods


## Importing

Standard library imports are together at the top.

After a space, third party library imports.

After a further space, internal imports.

Importing all names from a module with `import *` is generally avoided.