import torch

from registration.common import *


def fixed_polar_to_moving_cartesian(input_grid: Sinogram2dGrid, *, scene_geometry: SceneGeometry,
                                    transformation: Transformation) -> torch.Tensor:
    device = input_grid.r.device
    source_position = torch.tensor([0., 0., scene_geometry.source_distance], device=device)
    fixed_cartesian = torch.stack((input_grid.r * torch.cos(input_grid.phi), input_grid.r * torch.sin(input_grid.phi),
                                   torch.zeros_like(input_grid.r)), dim=-1)
    from_source = fixed_cartesian - source_position
    lambdas = torch.einsum('i,...i->...', transformation.translation.to(device=device) - source_position,
                           from_source) / torch.einsum('...i,...i->...', from_source, from_source)
    closest_points_fixed = source_position + lambdas.unsqueeze(-1) * from_source
    closest_points_moving = transformation.inverse()(closest_points_fixed)
    return closest_points_moving


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
                 samples_per_ray: int = 500) -> torch.Tensor:
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
    volume_diag: torch.Tensor = (torch.tensor(volume_data.size(), dtype=torch.float32) * voxel_spacing).flip(dims=(0,))
    lambda_start: float = .8 * scene_geometry.source_distance
    lambda_end: float = scene_geometry.source_distance
    step_size: float = (lambda_end - lambda_start) / float(samples_per_ray)
    deltas = directions * step_size
    deltas = transformation.inverse()(deltas, exclude_translation=True)
    starts = source_position + lambda_start * directions
    starts = transformation.inverse()(starts)

    deltas_texture = (2. * deltas / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    grid = (2. * starts / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    ret = torch.zeros(output_size, device=volume_data.device)
    for i in range(samples_per_ray):
        ret += torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
        grid += deltas_texture
    return step_size * ret
