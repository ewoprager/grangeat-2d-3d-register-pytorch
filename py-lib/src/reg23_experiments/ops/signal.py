import torch

__all__ = ["gaussian_kernel_1d"]


def gaussian_kernel_1d(*, sigma: float, cutoff_at_sigmas: float = 3.0, **tensor_kwargs) -> torch.Tensor:
    """
    :param sigma: The standard deviation of the Gaussian function used.
    :param cutoff_at_sigmas: Determines the size of the kernel generated. Will extend to a radius of
    `cutoff_at_sigmas * sigma` either side of the centre.
    :param tensor_kwargs: Keyword arguments to pass to the tensor when constructed.
    :return: A 1D tensor containing the coefficients of a Gaussian kernel generated with the given sigma.
    """
    radius = int(cutoff_at_sigmas * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, **tensor_kwargs)
    ret = torch.exp(-0.5 * (x / sigma).square())
    ret /= ret.sum()
    return ret
