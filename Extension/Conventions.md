# Conventions

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