import torch

from registration.lib.structs import *


def fixed_polar_to_moving_cartesian(input_grid: Sinogram2dGrid, *, ph_matrix: torch.Tensor) -> torch.Tensor:
    device = input_grid.phi.device
    assert ph_matrix.device == device
    # source_position = scene_geometry.source_position(device=device)
    # p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    # ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device))
    intermediates = torch.stack(
        (torch.cos(input_grid.phi), torch.sin(input_grid.phi), torch.zeros_like(input_grid.phi), -input_grid.r), dim=-1)
    n_tildes = torch.einsum('ij,...j->...i', ph_matrix.t(), intermediates)
    ns = n_tildes[..., 0:3]
    n_sqmags = torch.einsum('...i,...i->...', ns, ns) + 1e-12
    return -n_tildes[..., 3].unsqueeze(-1) * ns / n_sqmags.unsqueeze(-1)


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
    device = volume_data.device
    assert len(output_size) == 2
    assert voxel_spacing.size() == torch.Size([3])
    assert transformation.device_consistent()
    assert transformation.translation.device == device
    assert voxel_spacing.device == device
    img_width: int = output_size[1]
    img_height: int = output_size[0]
    source_position: torch.Tensor = torch.tensor([0., 0., scene_geometry.source_distance], device=device)
    detector_xs: torch.Tensor = detector_spacing[1] * (
            torch.arange(0, img_width, 1, dtype=torch.float32, device=device) - 0.5 * float(img_width - 1))
    detector_ys: torch.Tensor = detector_spacing[0] * (
            torch.arange(0, img_height, 1, dtype=torch.float32, device=device) - 0.5 * float(img_height - 1))
    detector_ys, detector_xs = torch.meshgrid(detector_ys, detector_xs)
    directions: torch.Tensor = torch.nn.functional.normalize(
        torch.stack((detector_xs, detector_ys, torch.zeros_like(detector_xs)), dim=-1) - source_position, dim=-1)
    volume_diag: torch.Tensor = torch.tensor(volume_data.size(), dtype=torch.float32, device=device).flip(
        dims=(0,)) * voxel_spacing
    volume_diag_length: torch.Tensor = volume_diag.norm()
    lambda_start: torch.Tensor = (source_position - transformation.translation).norm() - .5 * volume_diag_length
    lambda_end: torch.Tensor = lambda_start + volume_diag_length
    step_size: float = (lambda_end - lambda_start).item() / float(samples_per_ray)

    h_matrix_inv = transformation.inverse().get_h(device=device)
    deltas = directions * step_size
    deltas_homogeneous = torch.cat((deltas, torch.zeros_like(deltas[..., 0], device=device).unsqueeze(-1)), dim=-1)
    deltas = torch.einsum('ji,...i->...j', h_matrix_inv, deltas_homogeneous)[..., 0:3]
    starts = source_position + lambda_start * directions
    starts_homogeneous = torch.cat((starts, torch.ones_like(starts[..., 0], device=device).unsqueeze(-1)), dim=-1)
    starts = torch.einsum('ji,...i->...j', h_matrix_inv, starts_homogeneous)[..., 0:3]

    deltas_texture = 2. * deltas / volume_diag
    grid = (2. * starts / volume_diag).to(dtype=torch.float32)
    ret = torch.zeros(output_size, device=device)
    for i in range(samples_per_ray):
        ret += torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
        grid += deltas_texture
    return step_size * ret


def plane_integrals(volume_data: torch.Tensor, *, phi_values: torch.Tensor, theta_values: torch.Tensor,
                    r_values: torch.Tensor, voxel_spacing: torch.Tensor,
                    samples_per_direction: int = 500) -> torch.Tensor:
    assert voxel_spacing.size() == torch.Size([3])
    # devices consistent
    assert volume_data.device == phi_values.device
    assert phi_values.device == theta_values.device
    assert theta_values.device == r_values.device
    # sizes consistent
    assert phi_values.size() == theta_values.size()
    assert theta_values.size() == r_values.size()

    vol_size_world = torch.tensor(volume_data.size(), dtype=torch.float32) * voxel_spacing.flip(dims=(0,))
    plane_size = vol_size_world.square().sum().sqrt()

    def integrate_plane(phi: torch.Tensor, theta: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        ct = torch.cos(theta)
        st = torch.sin(theta)

        u_vector = torch.linspace(-.5 * plane_size, .5 * plane_size, samples_per_direction)
        v_grid, u_grid = torch.meshgrid(u_vector, u_vector)
        x_grid = 2. * (r * ct * cp - sp * u_grid - st * cp * v_grid) / vol_size_world[2]
        y_grid = 2. * (r * ct * sp + cp * u_grid - st * sp * v_grid) / vol_size_world[1]
        z_grid = 2. * (r * st + ct * v_grid) / vol_size_world[0]
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1)
        all_samples = torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[
            0, 0, 0]
        return (plane_size / samples_per_direction).square() * all_samples.sum()

    phi_flat = phi_values.flatten()
    theta_flat = theta_values.flatten()
    r_flat = r_values.flatten()

    ret = torch.zeros_like(phi_values)
    ret_flat = ret.flatten()

    for i in range(ret_flat.numel()):
        ret_flat[i] = integrate_plane(phi_flat[i], theta_flat[i], r_flat[i])

    return ret
