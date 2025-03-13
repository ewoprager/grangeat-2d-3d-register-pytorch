import torch
import numpy as np
import matplotlib.pyplot as plt

import Extension

from registration.lib.structs import *
import registration.lib.grangeat as grangeat
import registration.lib.plot as myplt


def zncc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    n = xs.numel()
    assert (ys.size() == xs.size())
    n = float(n)
    sum_x = xs.sum()
    sum_y = ys.sum()
    sum_x2 = xs.square().sum()
    sum_y2 = ys.square().sum()
    sum_prod = (xs * ys).sum()
    num = n * sum_prod - sum_x * sum_y
    den = (n * sum_x2 - sum_x.square()).sqrt() * (n * sum_y2 - sum_y.square()).sqrt()
    return num / (den + 1e-10)


def zncc2(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    n = xs.numel()
    assert (ys.size() == xs.size())
    n = float(n)
    xsf = xs.flatten()
    ysf = ys.flatten()
    to_sum = torch.stack((xsf, ysf, xsf.square(), ysf.square(), xsf * ysf), dim=0)
    sums = to_sum.sum(dim=1)
    num = n * sums[4] - sums[0] * sums[1]
    den = (n * sums[2] - sums[0].square()).sqrt() * (n * sums[3] - sums[1].square()).sqrt()
    return num / (den + 1e-10)


def evaluate(fixed_image: torch.Tensor, sinogram3d: torch.Tensor, *, transformation: Transformation,
             scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, sinogram3d_range: Sinogram3dRange,
             plot: bool = False, smooth: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if smooth:
        resampled = grangeat.resample_slice(sinogram3d, transformation=transformation, scene_geometry=scene_geometry,
                                            output_grid=fixed_image_grid, input_range=sinogram3d_range, smooth=smooth)
    else:
        device = sinogram3d.device
        source_position = scene_geometry.source_position(device=device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)
        sinogram_range_low = torch.tensor(
            [sinogram3d_range.r.low, sinogram3d_range.theta.low, sinogram3d_range.phi.low], device=device)
        sinogram_range_high = torch.tensor(
            [sinogram3d_range.r.high, sinogram3d_range.theta.high, sinogram3d_range.phi.high], device=device)
        sinogram_spacing = (sinogram_range_high - sinogram_range_low) / (
                torch.tensor(sinogram3d.size(), dtype=torch.float32, device=device) - 1.)
        sinogram_range_centres = .5 * (sinogram_range_low + sinogram_range_high)
        resampled = Extension.resample_sinogram3d(sinogram3d, sinogram_spacing, sinogram_range_centres, ph_matrix,
                                                  fixed_image_grid.phi, fixed_image_grid.r)

    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampled.clone().cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] resampled with sample smoothing" if smooth else "d/dr R3 [mu] resampled")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)
        plt.savefig("data/temp/d_dr_R3_mu_resampled_with_sample_smoothing.pgf" if smooth else "data/temp/d_dr_R3_mu_resampled.pgf")

    return zncc(fixed_image,
                resampled), resampled  # return Extension.normalised_cross_correlation(fixed_image, resampled), resampled


def evaluate_direct(fixed_image: torch.Tensor, volume_data: torch.Tensor, *, transformation: Transformation,
                    scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, voxel_spacing: torch.Tensor,
                    plot: bool = False) -> torch.Tensor:
    direct = grangeat.directly_calculate_radon_slice(volume_data, transformation=transformation,
                                                     scene_geometry=scene_geometry, output_grid=fixed_image_grid,
                                                     voxel_spacing=voxel_spacing)
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(direct.cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] calculated directly")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return zncc(fixed_image, direct)
