import torch
from scipy.spatial.transform import Rotation
import kornia

from registration.common import *


def transform(positions_cartesian: torch.Tensor, transformation: Transformation,
              exclude_translation: bool = False) -> torch.Tensor:
    r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(transformation.rotation[None, :])[0].to(
        device=positions_cartesian.device, dtype=torch.float32)
    positions_cartesian = torch.einsum('kl,...l->...k', r, positions_cartesian.to(dtype=torch.float32))
    if not exclude_translation:
        positions_cartesian = positions_cartesian + transformation.translation.to(device=positions_cartesian.device,
                                                                                  dtype=torch.float32)
    return positions_cartesian


def fixed_polar_to_moving_cartesian(input_grid: Sinogram2dGrid, *, scene_geometry: SceneGeometry) -> torch.Tensor:
    hypotenuses = (input_grid.r.square() + scene_geometry.source_distance * scene_geometry.source_distance).sqrt()
    sin_alphas = input_grid.r / hypotenuses
    zs = sin_alphas.square() * (scene_geometry.source_distance - scene_geometry.ct_origin_distance)
    scaled_rs = (
                            scene_geometry.source_distance - scene_geometry.ct_origin_distance - zs) * input_grid.r / scene_geometry.source_distance
    xs = scaled_rs * torch.cos(input_grid.phi)
    ys = scaled_rs * torch.sin(input_grid.phi)
    return torch.stack((xs, ys, zs), dim=-1)


def moving_cartesian_to_moving_spherical(positions_cartesian: torch.Tensor) -> Sinogram3dGrid:
    xs = positions_cartesian[..., 0]
    ys = positions_cartesian[..., 1]
    zs = positions_cartesian[..., 2]
    phis = torch.atan2(ys, xs)
    phis_over = phis > .5 * torch.pi
    phis_under = phis < -.5 * torch.pi
    phis[phis_over] -= torch.pi
    phis[phis_under] += torch.pi
    r2s_x_y = xs.square() + ys.square()
    thetas = torch.atan2(zs, r2s_x_y.sqrt())
    rs = (r2s_x_y + zs.square()).sqrt()
    rs[torch.logical_or(phis_over, phis_under)] *= -1.
    return Sinogram3dGrid(phis, thetas, rs)


def generate_drr(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
                 detector_spacing: torch.Tensor, scene_geometry: SceneGeometry, output_size: torch.Size,
                 samples_per_ray: int = 64) -> torch.Tensor:
    img_width: int = output_size[1]
    img_height: int = output_size[0]
    source_position: torch.Tensor = torch.tensor([0., 0., scene_geometry.source_distance])
    detector_xs: torch.Tensor = detector_spacing[1] * (
            torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    detector_ys: torch.Tensor = detector_spacing[0] * (
            torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    detector_ys, detector_xs = torch.meshgrid(detector_ys, detector_xs)
    directions: torch.Tensor = torch.nn.functional.normalize(
        torch.stack((detector_xs, detector_ys, torch.zeros_like(detector_xs)), dim=-1) - source_position, dim=-1)
    volume_depth: float = volume_data.size()[2] * voxel_spacing[2].item()
    volume_diag: torch.Tensor = (torch.tensor(volume_data.size(), dtype=torch.float32) * voxel_spacing).flip(dims=(0,))
    lambda_start: float = scene_geometry.source_distance - scene_geometry.ct_origin_distance - .5 * volume_depth
    lambda_end: float = torch.linalg.vector_norm(torch.tensor(
        [0., 0., scene_geometry.ct_origin_distance - scene_geometry.source_distance]) - .5 * volume_diag).item()
    step_size: float = (lambda_end - lambda_start) / float(samples_per_ray)
    deltas = directions * step_size
    deltas = transform(deltas, transformation, exclude_translation=True)
    starts = source_position + lambda_start * directions - torch.tensor([0., 0., scene_geometry.ct_origin_distance])
    starts = transform(starts, transformation)

    deltas_texture = (2. * deltas / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    grid = (2. * starts / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    ret = torch.zeros(output_size, device=volume_data.device)
    for i in range(samples_per_ray):
        ret += torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
        grid += deltas_texture
    return step_size * ret
