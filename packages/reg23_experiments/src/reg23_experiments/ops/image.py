from typing import Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

from reg23_experiments.ops.signal import gaussian_kernel_1d

__all__ = ["gaussian_blur_2d", "frequency_filter"]


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


@jaxtyped(typechecker=typechecker)
def frequency_filter(  #
        image: Float[torch.Tensor, "... n m"],  #
        spacing: Float[torch.Tensor, "2"],  #
        function: Callable[[torch.Tensor], torch.Tensor],  #
) -> torch.Tensor:
    """
    Apply a radial frequency-domain filter to a 2D image.

    Radial distance is measured in cycles per mm (assuming spacing is given in mm).

    :param image:
    :param spacing: A tensor of size (2,): the spacing of the image pixels (w, h).
    :param function: A function that maps, element-wise, frequencies to filter weights.
    """
    f_image = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))

    fy = torch.fft.fftshift(torch.fft.fftfreq(image.size()[-2], d=spacing[1], device=image.device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(image.size()[-1], d=spacing[0], device=image.device))
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")

    radii = (fx.square() + fy.square()).sqrt()
    mask = function(radii)
    result = torch.fft.ifft2(torch.fft.ifftshift(f_image * mask, dim=(-2, -1)))

    return result.real
