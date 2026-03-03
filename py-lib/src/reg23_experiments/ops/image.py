import torch

from reg23_experiments.ops.signal import gaussian_kernel_1d

__all__ = ["gaussian_blur_2d"]


def gaussian_blur_2d(image: torch.Tensor, *, sigma: float | tuple[float, float]) -> torch.Tensor:
    """
    :param image: 2D torch.Tensor
    :param sigma: float | tuple[float, float]; the standard deviation(s) of the Gaussian kernel to blur with in
    each direction. If one value is given, it is used in all directions.
    :return: A blurred copy of `image` using the given sigma, matching `image` in size.
    """
    if isinstance(sigma, float):
        sigma = (sigma, sigma)

    ky = gaussian_kernel_1d(sigma=sigma[0], dtype=image.dtype, device=image.device)
    kx = gaussian_kernel_1d(sigma=sigma[1], dtype=image.dtype, device=image.device)

    ret = torch.nn.functional.conv2d(image.unsqueeze(0).unsqueeze(0), ky.view(-1, 1).unsqueeze(0).unsqueeze(0),
                                     padding='same')
    return torch.nn.functional.conv2d(ret, kx.view(1, -1).unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
