import torch
import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from registration.lib.structs import *
from registration.lib.sinogram import SinogramClassic

import Extension


def fixed_polar_to_moving_cartesian(input_grid: Sinogram2dGrid, *, ph_matrix: torch.Tensor) -> torch.Tensor:
    device = input_grid.phi.device
    assert ph_matrix.device == device
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


def grid_sample_sinogram3d_smoothed(sinogram3d: torch.Tensor, grid: Sinogram3dGrid, *, i_mapping: LinearMapping,
                                    j_mapping: LinearMapping, k_mapping: LinearMapping, a_count: int = 6,
                                    b_count: int = 16, sigma: float | None = None):
    """
    Sample the given sinogram at the given phi, theta, r spherical coordinates, with extra samples in a Gaussian layout
    around the sampling positions to make the sampling more even over S^2, even if the point distribution is less even.

    :param sinogram3d: A 3D tensor
    :param grid: A grid of 3D sinogram coordinates
    :param i_mapping: Mapping from r to sinogram texture x-coordinate (-1, 1)
    :param j_mapping: Mapping from theta to sinogram texture y-coordinate (-1, 1)
    :param k_mapping: Mapping from phi to sinogram texture z-coordinate (-1, 1)
    :param a_count: Number of radial values at which to make extra samples in the Gaussian pattern
    :param b_count: Number of samples to make at each radius in the Gaussian pattern
    :param sigma: The standard deviation of the Gaussian pattern. Optional; if not provided, a sensible value is
    determined from the phi count in the given sinogram
    :return: A tensor matching size of `phi_values`  - the weighted sums of offset samples around the given coordinates.
    """
    device = sinogram3d.device

    assert grid.phi.device == device
    assert grid.device_consistent()
    assert grid.size_consistent()
    assert len(sinogram3d.size()) == 3
    assert sigma is None or sigma > 0.

    # Determine a sensible value of sigma if not provided
    if sigma is None:
        phi_count: int = sinogram3d.size()[0]
        sigma = 2. * torch.pi / (6. * float(phi_count))

    logger.info("Sample smoothing with sigma = {:.3f}".format(sigma))

    # Radial distances in Gaussian pattern
    delta_a: float = 3. * sigma / float(a_count)
    a_values: torch.Tensor = delta_a * torch.arange(0., float(a_count), 1.)
    # Orientations in Gaussian pattern
    b_values = torch.linspace(-torch.pi, torch.pi, b_count + 1)[:-1]
    # Sample weights in Gaussian pattern
    w_values = (-a_values.square() / (2. * sigma * sigma)).exp()
    w_values = w_values / (w_values.sum() * float(b_count))

    b_values, a_values = torch.meshgrid(b_values, a_values)

    # New offset values of phi & theta are determined by rotating the vector (1, 0, 0)^T first by a small perturbation
    # according to the Gaussian pattern, and then by the original rotation according to the old values of phi * theta.

    # Determining a perturbed vector for each offset in the Gaussian pattern:
    ca = a_values.cos()
    sa = a_values.sin()
    cb = b_values.cos()
    sb = b_values.sin()
    offset_vectors = torch.stack((ca, -sa * sb, sa * cb), dim=-1).to(device=device)

    del ca, sa, cb, sb

    # Determining the rotation matrices for each input (phi, theta):
    cp = grid.phi.cos()
    sp = grid.phi.sin()
    ct = grid.theta.cos()
    st = grid.theta.sin()
    row_0 = torch.stack((cp * ct, -sp, -cp * st), dim=-1)
    row_1 = torch.stack((sp * ct, cp, sp * st), dim=-1)
    row_2 = torch.stack((st, torch.zeros_like(st), ct), dim=-1)
    rotation_matrices = torch.stack((row_0, row_1, row_2), dim=-2).to(device=device)
    # Multiplying by the perturbed unit vectors for the perturbed, rotated unit vector:
    rotated_vectors = torch.einsum('...ij,baj->...bai', rotation_matrices, offset_vectors)

    del cp, sp, ct, st, row_0, row_1, row_2, rotation_matrices, offset_vectors

    # Converting the resulting unit vectors back into new values of phi & theta, and expanding the r tensor to match
    # in size:
    new_phis = torch.atan2(rotated_vectors[..., 1], rotated_vectors[..., 0]).flatten(-2)
    new_grid = Sinogram3dGrid(new_phis, torch.clamp(rotated_vectors[..., 2], -1., 1.).asin().flatten(-2),
                              grid.r.unsqueeze(-1).expand(
                                  [-1] * len(grid.r.size()) + [new_phis.size()[-1]]).clone())  # need to
    # clone the r grid otherwise it's just a tensor view, not a tensor in its own right

    del rotated_vectors, new_phis

    new_grid = SinogramClassic.unflip_coordinates(new_grid)

    # Sampling at all the perturbed orientations:
    grid = torch.stack((i_mapping(new_grid.r), j_mapping(new_grid.theta), k_mapping(new_grid.phi)), dim=-1)
    samples = Extension.grid_sample3d(sinogram3d, grid, address_mode="wrap")

    del grid

    ##
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_grid.phi.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average phi_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_grid.theta.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average theta_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_grid.r.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average r_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    ##

    # Applying the weights and summing along the last dimension for an output equal in size to the input tensors of
    # coordinates:
    return torch.einsum('i,...i->...', w_values.repeat(b_count).to(device=device), samples)
