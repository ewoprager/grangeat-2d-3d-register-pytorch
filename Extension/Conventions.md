<!--- \page conventions Conventions

This file describes all mathematical conventions used throughout the library.
-->

# Conventions

## Cartesian coordinates

### 2D

In image data, the dimensions from largest to smallest are (Y, X), so the 1D index in a 2D array is given by
`width * Y + X`.

### 3D

(X, Y, Z) is always right-handed.

In volume data, the origin is placed at the centre, and the dimensions from largest to smallest are (Z, Y, X), so the 1D
index in a 3D array is given by `width * height * Z + width * Y + X`.

## Polar coordinates

In 2D sinograms, lines in 2D space defined in polar coordinates are done so as follows:

The vector from the origin to the closest point on the line (which is therefor normal to the line) defines the line, and
is parametrised by the following 2-vector:
```
(r, phi)
```
where
- `r` is the **signed** length of the vector (the signed shortest distance between the line and the origin),
- `phi` in (-pi/2, pi/2) is the angle anti-clockwise from the positive x-direction to the vector.

## Spherical coordinates

In 3D sinograms, planes in 3D space defined in spherical coordinates are done so as follows:

The vector from the origin to the closest point on the plane (which is therefore normal to the plane) defines the plane,
and is parametrised by the following 3-vector:

```
(r, theta, phi)
```

where

- `r` is the **signed** length of the vector (the signed shortest distance between the plane and the origin),
- `theta` in (-pi/2, pi/2) is the angle of an initial rotation of the vector, left-hand-rule about the y-axis, taking
  the vector's initial position to be from the origin to the point `(r, 0, 0)`,
- `phi` in (-pi/2, pi/2) is the angle of a second rotation of the vector, right-hand-rule about the z-axis.