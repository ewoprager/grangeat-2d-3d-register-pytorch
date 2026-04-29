<!---
Note: Inline maths is written between $ ... \f$ as this works with MathJax when uploaded as documentation.
-->

For setup instructions, see
the [README on GitHub](https://github.com/ewoprager/grangeat-2d-3d-register-pytorch/tree/main/reg23).

Conventions obeyed by the implementations are detailed in \ref conventions.

<h2> The PyTorch extension </h2>

This library is a PyTorch extension, providing a number of functions for:

- grid sampling of 3-dimensional tensors with different flexibilities to `torch.nn.functional.grid_sample`,
- evaluating approximations of Radon transforms of 2- and 3-D images, and
- resampling of 3D sinograms and evaluation of similarity metrics between images for the purposes of 2D/3D image
  registration (see paper referenced in
  the [README](https://github.com/ewoprager/grangeat-2d-3d-register-pytorch/tree/main/reg23)).

The functions are documented here: \ref pytorch_functions