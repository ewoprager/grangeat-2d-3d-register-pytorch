from typing import Tuple, Literal
import logging

from sympy.physics.quantum.gate import zx_basis_transform

logger = logging.getLogger(__name__)

import torch
import numpy as np
import matplotlib.pyplot as plt

import Extension

from registration.lib.structs import SceneGeometry, Transformation, Sinogram2dGrid
from registration.lib.sinogram import Sinogram, SinogramClassic
import registration.lib.grangeat as grangeat
import registration.lib.plot as myplt


def ncc(xs: torch.Tensor, ys: torch.Tensor, *, dims: Tuple | torch.Size | None = None) -> torch.Tensor:
    assert xs.size() == ys.size()
    if dims is None:
        dims = torch.Size(range(len(xs.size())))
    sum_x = xs.sum(dim=dims)
    sum_y = ys.sum(dim=dims)
    sum_x2 = xs.square().sum(dim=dims)
    sum_y2 = ys.square().sum(dim=dims)
    sum_prod = (xs * ys).sum(dim=dims)
    n = float(xs.numel() // sum_x.numel())
    num = n * sum_prod - sum_x * sum_y
    den = (n * sum_x2 - sum_x.square()).sqrt() * (n * sum_y2 - sum_y.square()).sqrt()
    return num / (den + 1e-10)


def local_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    """
    Divides the two input images into patches of size `kernel_size`, evaluates the NCC between each corresponding pair
    of patches and returns the mean of the resulting NCC values over all the patches.

    :param xs: [tensor of size (n, m)] one input image
    :param ys: [tensor of size (n, m)] another input image
    :param kernel_size:
    :return: The mean of the NCCs of the image patches.
    """
    assert xs.size() == ys.size()
    assert len(xs.size()) == 2
    xs_patches = torch.nn.functional.unfold(xs.unsqueeze(0), kernel_size=kernel_size,
                                            stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
    ys_patches = torch.nn.functional.unfold(ys.unsqueeze(0), kernel_size=kernel_size,
                                            stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
    return ncc(xs_patches, ys_patches, dims=(0,)).mean()


def multiscale_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int, llambda: float) -> torch.Tensor:
    return ncc(xs, ys) + llambda * local_ncc(xs, ys, kernel_size=kernel_size)


def gradient_correlation(xs: torch.Tensor, ys: torch.Tensor, *,
                         gradient_method: Literal["sobel"] | Literal["central_difference"] = "sobel") -> torch.Tensor:
    assert xs.size() == ys.size()
    assert len(xs.size()) == 2
    if gradient_method is "sobel":
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        gx_xs = torch.nn.functional.conv2d(xs.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))[0, 0]
        gy_xs = torch.nn.functional.conv2d(xs.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))[0, 0]
        gx_ys = torch.nn.functional.conv2d(ys.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))[0, 0]
        gy_ys = torch.nn.functional.conv2d(ys.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))[0, 0]
    else:  # gradient_method is "central_difference"
        gx_xs = torch.gradient(xs, dim=0)[0]
        gy_xs = torch.gradient(xs, dim=1)[0]
        gx_ys = torch.gradient(ys, dim=0)[0]
        gy_ys = torch.gradient(ys, dim=1)[0]
    return 0.5 * (ncc(gx_xs, gx_ys) + ncc(gy_xs, gy_ys))


def evaluate(fixed_image: torch.Tensor, sinogram3d: Sinogram, *, transformation: Transformation,
             scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, save: bool = False,
             smooth: float | None = None, plot: Tuple[float, float] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sinogram3d.device
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device).to(dtype=torch.float32))

    if smooth is not None and isinstance(sinogram3d, SinogramClassic):
        resampled = sinogram3d.resample_python(ph_matrix=ph_matrix, fixed_image_grid=fixed_image_grid, smooth=smooth,
                                               plot=plot is not None)
    else:
        if smooth is not None:
            logger.warning("Cannot resample smooth as not given a SinogramClassic")
        resampled = sinogram3d.resample(ph_matrix, fixed_image_grid)

    if plot is not None:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampled.clone().cpu(), vmin=plot[0], vmax=plot[1])
        axes.axis('square')
        axes.set_title(
            "d/dr R3 [mu] resampled with sample smoothing" if smooth is not None else "d/dr R3 [mu] resampled")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)
        if save:
            plt.savefig(
                "data/temp/d_dr_R3_mu_resampled_with_sample_smoothing.pgf" if smooth else "data/temp/d_dr_R3_mu_resampled.pgf")

    return ncc(fixed_image,
               resampled), resampled  # return Extension.normalised_cross_correlation(fixed_image, resampled),  # resampled


def evaluate_direct(fixed_image: torch.Tensor, volume_data: torch.Tensor, *, transformation: Transformation,
                    scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, voxel_spacing: torch.Tensor,
                    plot: bool = False) -> torch.Tensor:
    device = volume_data.device
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)

    direct = grangeat.directly_calculate_radon_slice(volume_data, ph_matrix=ph_matrix, output_grid=fixed_image_grid,
                                                     voxel_spacing=voxel_spacing)
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(direct.cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] calculated directly")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return ncc(fixed_image, direct)
