import numpy as np
import torch

from reg23_experiments.ops.signal import gaussian_kernel_1d

__all__ = ["gaussian_blur_3d", "downsample_trilinear_antialiased", "fit_line_3d", "point_line_distance_3d"]


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


def unfold_3d(volume: torch.Tensor, kernel_size: int | tuple[int, int, int],
              stride: int | tuple[int, int, int]) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    padding = [(k - s + 1) // 2 for k, s in zip(kernel_size, stride)]
    # in x direction
    volume = torch.cat((  #
        volume[:, :, 0].unsqueeze(2).expand(-1, -1, padding[2]),  #
        volume,  #
        volume[:, :, -1].unsqueeze(2).expand(-1, -1, padding[2]),  #
    ), dim=2)
    # in y direction
    volume = torch.cat((  #
        volume[:, 0, :].unsqueeze(1).expand(-1, padding[1], -1),  #
        volume,  #
        volume[:, -1, :].unsqueeze(1).expand(-1, padding[1], -1),  #
    ), dim=1)
    # in z direction
    volume = torch.cat((  #
        volume[0, :, :].unsqueeze(0).expand(padding[0], -1, -1),  #
        volume,  #
        volume[-1, :, :].unsqueeze(0).expand(padding[0], -1, -1),  #
    ), dim=0)
    volume = volume.unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1]).unfold(2, kernel_size[2],
                                                                                                     stride[2])
    return volume


def downsample_trilinear_antialiased_memory_constrained(*, volume: torch.Tensor,
                                                        scale_factor: float | tuple[float, float, float],
                                                        max_bytes: int) -> torch.Tensor:
    """
    (n + 2a)(n + 2b)(n + 2c) = b
    n^3 + 2(a + b + c)n^2 + 4(ab + ac + bc)n + 8abc - b = 0

    :param volume:
    :param scale_factor:
    :param max_bytes:
    :return:
    """
    if isinstance(scale_factor, float):
        scale_factor = (scale_factor, scale_factor, scale_factor)

    sigma = tuple(max((1.0 / s - 1.0) * 0.5, 0.001) for s in scale_factor)
    cutoff_at_sigmas = 3.0  # default used in gaussian_kernel_1d
    halo_radii = [int(cutoff_at_sigmas * s + 0.5) for s in sigma]
    max_simultaneous_images = 2.0
    max_bytes_per_image = float(max_bytes) / max_simultaneous_images
    halo_radii_f = np.array(halo_radii, dtype=float)
    a = 1.0
    b = 2.0 * np.sum(halo_radii_f)
    c = 4.0 * (halo_radii_f[0] * halo_radii_f[1] + halo_radii_f[1] * halo_radii_f[2] + halo_radii_f[0] * halo_radii_f[
        2])
    d = 8.0 * np.prod(halo_radii_f)
    chunk_nominal_side_length = int(np.max(np.roots([a, b, c, d])))
    kernel_size = tuple(chunk_nominal_side_length + 2 * r for r in halo_radii)
    chunks = unfold_3d(volume, kernel_size=kernel_size, stride=chunk_nominal_side_length)


def fit_line_3d(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a straight line to a cloud of points in 3D space.
    :param points: (N, 3), matrix of points in 3D
    :return: (point on line, direction), both size (3,)
    """
    centroid = points.mean(dim=0)
    matrix = points - centroid
    return centroid, torch.linalg.svd(matrix)[2][0]


def point_line_distance_3d(*, points: torch.Tensor, line_point: torch.Tensor,
                           line_direction: torch.Tensor) -> torch.Tensor:
    """

    :param points: (N, 3)
    :param line_point: (3,)
    :param line_direction: (3,)
    :return: tensor of size (N,)
    """
    from_points = points - line_point  # size = (N, 3)
    dots = from_points @ line_direction  # size = (N,)
    projected_points = dots.unsqueeze(-1) * line_direction  # size = (N, 3)
    line_offsets = from_points - projected_points  # size = (N, 3)
    return torch.linalg.vector_norm(line_offsets, dim=-1)  # size = (N,)
