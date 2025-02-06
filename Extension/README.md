# Build

## Directly with setuptools:
`python setup.py develop`
- In debug: `--debug`
- Without CUDA: `--no-cuda`


# Radon Transforms

The $n$-dimensional Radon Transform $\mathcal{R}_n$ maps from one scalar field to another:
$$
\mathcal{R}_n : f \to g, \quad f, g : \mathbb{R}^n \to \mathbb{R}
$$
It is equal to the set of all $(n-1)$-dimensional hyper-plane integrals:
$$
\mathcal{R}_n[f](\mathbf{\hat{n}, d}) = \int \dots \int_{\mathbb{R}^n} \! f(\mathbf{x}) \delta(\mathbf{x} \cdot
\mathbf{\hat{n}} - d) \, \mathrm{d} x^n
$$

The implementations provided here are for 2- and 3-dimensional Radon transforms which take as input:
- the input field $f$ as a 2/3-dimensional tensor of
`torch.float32`s, interpreted as begin evenly spaced samples in 2/3-dimensional Cartesian space respectively,
- the output polar/spherical coordinates give as 2/3 1-D tensors of `torch.float32`s
- the number of samples to sum over along each line/over each plane when calculating the approximation of a
hyper-plane integral

and give as output a 2/3-dimensional tensor of `torch.float32`s corresponding to the values of the hyperplane
integrals through $f$, where each hyper-plane defined by normal $\mathbf{\hat{n}}$ and origin distance $d$ corresponds
to the vector $d \mathbf{\hat{n}}$ defined in polar/spherical coordinates in
```python
torch.stack(torch.meshgrid(phi_values, r_values), dim=-1)
```
or
```python
torch.stack(torch.meshgrid(phi_values, theta_values, r_values), dim=-1)
```