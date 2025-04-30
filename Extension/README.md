# Grangeat-based 2D/3D image registration

## Build

### Directly with setuptools:

`python setup.py develop`

- In debug: `--debug`
- Without CUDA: `--no-cuda`

## Documentation

The documentation for this extension is uploaded
at https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/index.html.

## Radon Transforms

The $n$-dimensional Radon Transform $\mathcal{R}_n$ maps from one scalar field to another:
$$
\mathcal{R}_n : f \to g, \quad f, g : \mathbb{R}^n \to \mathbb{R}
$$
It is equal to the set of all $(n-1)$-dimensional hyper-plane integrals:
$$
\mathcal{R}_n[f](\mathbf{\hat{n}, d}) = \int \dots \int_{\mathbb{R}^n} \! f(\mathbf{x}) \delta(\mathbf{x} \cdot
\mathbf{\hat{n}} - d) \, \mathrm{d} x^n
$$

The implementations provided here are for 2- and 3-dimensional Radon transforms.