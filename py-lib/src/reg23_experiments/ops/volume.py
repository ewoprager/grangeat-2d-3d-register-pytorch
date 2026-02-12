import torch


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


def gaussian_blur_3d(volume: torch.Tensor, *, sigma: float | tuple[float, ...]) -> torch.Tensor:
    """
    :param volume: 3D torch.Tensor
    :param sigma: float | tuple[float, float, float]; the standard deviation(s) of the Gaussian kernel to blur with in
    each direction. If one value is given, it is used in all directions.
    :return: A blurred copy of `volume` using the given sigma, matching `volume` in size.
    """
    if isinstance(sigma, float):
        sigma = (sigma, sigma, sigma)

    kz = gaussian_kernel_1d(sigma=sigma[0], dtype=volume.dtype, device=volume.device)
    ky = gaussian_kernel_1d(sigma=sigma[1], dtype=volume.dtype, device=volume.device)
    kx = gaussian_kernel_1d(sigma=sigma[2], dtype=volume.dtype, device=volume.device)

    ret = torch.nn.functional.conv3d(volume.unsqueeze(0).unsqueeze(0), kz.view(-1, 1, 1).unsqueeze(0).unsqueeze(0),
                                     padding='same')
    ret = torch.nn.functional.conv3d(ret, ky.view(1, -1, 1).unsqueeze(0).unsqueeze(0), padding='same')
    return torch.nn.functional.conv3d(ret, kx.view(1, 1, -1).unsqueeze(0).unsqueeze(0), padding='same')[0, 0]


def downsample_trilinear_antialiased(volume: torch.Tensor,
                                     scale_factor: float | tuple[float, float, float]) -> torch.Tensor:
    """
    :param volume:
    :param scale_factor:
    :return: A downsampled copy of `volume` by the given scale factor. Gaussian blurring is used to prevent aliasing.
    """
    if isinstance(scale_factor, float):
        scale_factor = (scale_factor, scale_factor, scale_factor)

    sigma = tuple(max((1.0 / s - 1.0) * 0.5, 0.001) for s in scale_factor)

    ret = gaussian_blur_3d(volume, sigma=sigma)

    return torch.nn.functional.interpolate(  #
        ret.unsqueeze(0).unsqueeze(0),  #
        scale_factor=scale_factor,  #
        mode='trilinear',  #
        recompute_scale_factor=True  #
    )[0, 0]
