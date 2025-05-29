from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

import torch
import matplotlib.pyplot as plt

from registration.lib.structs import *
from registration.lib import geometry

import Extension as ExtensionTest


def calculate_radon_volume(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, samples_per_direction: int = 128,
                           output_grid: Sinogram3dGrid):
    assert output_grid.device_consistent()
    assert volume_data.device == output_grid.phi.device
    return ExtensionTest.d_radon3d_dr(
        volume_data, voxel_spacing, output_grid.phi.to(device=volume_data.device),
        output_grid.theta.to(device=volume_data.device), output_grid.r.to(device=volume_data.device),
        samples_per_direction)


def calculate_fixed_image(drr_image: torch.Tensor, *, source_distance: float, detector_spacing: torch.Tensor,
                          output_grid: Sinogram2dGrid) -> torch.Tensor:
    device = drr_image.device
    assert output_grid.device_consistent()
    assert output_grid.phi.device == device

    img_width = drr_image.size()[1]
    img_height = drr_image.size()[0]

    samples_per_line = int(torch.tensor(drr_image.size()).square().sum().sqrt().ceil().item())

    xs = detector_spacing[0] * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing[1] * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    ys, xs = torch.meshgrid(ys, xs)
    cos_gamma = source_distance / torch.sqrt(xs.square() + ys.square() + source_distance * source_distance)
    g_tilde = cos_gamma.to(device=device) * drr_image

    fixed_scaling = (output_grid.r / source_distance).square() + 1.

    ##
    # no_derivative = ExtensionTest.radon2d(g_tilde, detector_spacing, detector_spacing, phi_values, r_values,
    #                                       samples_per_line)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(no_derivative.cpu())
    # axes.axis('square')
    # axes.set_title("R2 [g^tilde]")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # post_derivative = no_derivative.diff(dim=-1) / torch.abs(r_values[1] - r_values[0])
    # post_derivative[:, :(post_derivative.size()[1] // 2)] *= -1.
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(post_derivative.cpu())
    # axes.axis('square')
    # axes.set_title("diff/ds R2 [g^tilde]")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    ##

    return fixed_scaling * ExtensionTest.d_radon2d_dr(
        g_tilde, detector_spacing, output_grid.phi, output_grid.r, samples_per_line)


def directly_calculate_radon_slice(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, ph_matrix: torch.Tensor,
                                   output_grid: Sinogram2dGrid) -> torch.Tensor:
    assert len(volume_data.size()) == 3
    assert len(voxel_spacing.size()) == 1
    assert voxel_spacing.size()[0] == 3
    assert output_grid.device_consistent()
    assert output_grid.phi.device == volume_data.device
    assert ph_matrix.device == volume_data.device

    output_grid_2d = Sinogram2dGrid(*torch.meshgrid(output_grid.phi, output_grid.r))

    output_grid_cartesian_2d = geometry.fixed_polar_to_moving_cartesian(output_grid_2d, ph_matrix=ph_matrix)

    output_grid_sph_2d = geometry.moving_cartesian_to_moving_spherical(output_grid_cartesian_2d)

    rows = output_grid_sph_2d.phi.size()[0]
    cols = output_grid_sph_2d.phi.size()[1]
    ret = torch.zeros_like(output_grid_sph_2d.phi)
    for n in tqdm(range(rows * cols)):
        i = n % cols
        j = n // cols
        ret[j, i] = ExtensionTest.d_radon3d_dr_v2(
            volume_data, voxel_spacing, output_grid_sph_2d.phi[j, i].unsqueeze(0),
            output_grid_sph_2d.theta[j, i].unsqueeze(0), output_grid_sph_2d.r[j, i].unsqueeze(0), 500)
    return ret
