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
