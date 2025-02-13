import torch
from tqdm import tqdm

from registration.common import *
import registration.geometry as geometry

import Extension as ExtensionTest


def calculate_radon_volume(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, samples_per_direction: int = 128,
                           output_grid: Sinogram3dGrid):
    assert output_grid.device_consistent()
    assert volume_data.device == output_grid.phi.device
    return ExtensionTest.dRadon3dDR(volume_data, voxel_spacing[2].item(), voxel_spacing[1].item(),
                                    voxel_spacing[0].item(), output_grid.phi.to(device=volume_data.device),
                                    output_grid.theta.to(device=volume_data.device),
                                    output_grid.r.to(device=volume_data.device), samples_per_direction)


def calculate_fixed_image(drr_image: torch.Tensor, *, source_distance: float, detector_spacing: torch.Tensor,
                          output_grid: Sinogram2dGrid, samples_per_line: int = 128) -> torch.Tensor:
    device = drr_image.device
    assert output_grid.device_consistent()
    assert output_grid.phi.device == device

    img_width = drr_image.size()[1]
    img_height = drr_image.size()[0]

    xs = detector_spacing[1] * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing[0] * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
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

    return fixed_scaling * ExtensionTest.dRadon2dDR(g_tilde, detector_spacing[1].item(), detector_spacing[0].item(),
                                                    output_grid.phi, output_grid.r, samples_per_line)


def directly_calculate_radon_slice(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor,
                                   transformation: Transformation, scene_geometry: SceneGeometry,
                                   output_grid: Sinogram2dGrid) -> torch.Tensor:
    assert len(volume_data.size()) == 3
    assert len(voxel_spacing.size()) == 1
    assert voxel_spacing.size()[0] == 3
    assert output_grid.device_consistent()
    assert output_grid.phi.device == volume_data.device
    assert transformation.device_consistent()

    output_grid_2d = Sinogram2dGrid(*torch.meshgrid(output_grid.phi, output_grid.r))

    output_grid_cartesian_2d = geometry.fixed_polar_to_moving_cartesian(output_grid_2d, scene_geometry=scene_geometry,
                                                                        transformation=transformation)

    # output_grid_cartesian_2d = geometry.fixed_polar_to_moving_cartesian2(output_grid_2d, scene_geometry=scene_geometry,
    #                                                                      transformation=transformation)

    output_grid_sph_2d = geometry.moving_cartesian_to_moving_spherical(output_grid_cartesian_2d)

    rows = output_grid_sph_2d.phi.size()[0]
    cols = output_grid_sph_2d.phi.size()[1]
    ret = torch.zeros_like(output_grid_sph_2d.phi)
    for n in tqdm(range(rows * cols)):
        i = n % cols
        j = n // cols
        ret[j, i] = ExtensionTest.dRadon3dDR_v2(volume_data, voxel_spacing[2].item(), voxel_spacing[1].item(),
                                                voxel_spacing[0].item(), output_grid_sph_2d.phi[j, i].unsqueeze(0),
                                                output_grid_sph_2d.theta[j, i].unsqueeze(0),
                                                output_grid_sph_2d.r[j, i].unsqueeze(0), 500)
    return ret


def resample_slice(sinogram3d: torch.Tensor, *, input_range: Sinogram3dRange, transformation: Transformation,
                   scene_geometry: SceneGeometry, output_grid: Sinogram2dGrid) -> torch.Tensor:
    assert output_grid.device_consistent()
    assert output_grid.phi.device == sinogram3d.device
    assert transformation.device_consistent()

    output_grid_cartesian = geometry.fixed_polar_to_moving_cartesian(output_grid, scene_geometry=scene_geometry,
                                                                     transformation=transformation)

    output_grid_sph = geometry.moving_cartesian_to_moving_spherical(output_grid_cartesian)

    ## sign changes
    moving_origin_projected = -scene_geometry.source_distance * transformation.translation[0:2] / (
            transformation.translation[2] - scene_geometry.source_distance)
    square_radius: torch.Tensor = .25 * moving_origin_projected.square().sum()
    need_sign_change = ((output_grid.r.unsqueeze(-1) * torch.stack(
        (torch.cos(output_grid.phi), torch.sin(output_grid.phi)), dim=-1) - .5 * moving_origin_projected).square().sum(
        dim=-1) < square_radius)
    ##

    ##
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(phi_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("phi_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(theta_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("theta_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(r_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("r_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    ##

    grid_range = LinearRange.grid_sample_range()

    i_mapping: LinearMapping = grid_range.get_mapping_from(input_range.r)
    j_mapping: LinearMapping = grid_range.get_mapping_from(input_range.theta)
    k_mapping: LinearMapping = grid_range.get_mapping_from(input_range.phi)

    grid = torch.stack((i_mapping(output_grid_sph.r), j_mapping(output_grid_sph.theta), k_mapping(output_grid_sph.phi)),
                       dim=-1)
    ret = torch.nn.functional.grid_sample(sinogram3d[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
    ret[need_sign_change] *= -1.
    return ret
