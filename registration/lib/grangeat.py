from tqdm import tqdm
import logging

import Extension

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
    return ExtensionTest.d_radon3d_dr(volume_data, voxel_spacing, output_grid.phi.to(device=volume_data.device),
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

    return fixed_scaling * ExtensionTest.d_radon2d_dr(g_tilde, detector_spacing, output_grid.phi, output_grid.r,
                                                      samples_per_line)


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

    # output_grid_cartesian_2d = geometry.fixed_polar_to_moving_cartesian2(output_grid_2d,
    # scene_geometry=scene_geometry,
    #                                                                      transformation=transformation)

    output_grid_sph_2d = geometry.moving_cartesian_to_moving_spherical(output_grid_cartesian_2d)

    rows = output_grid_sph_2d.phi.size()[0]
    cols = output_grid_sph_2d.phi.size()[1]
    ret = torch.zeros_like(output_grid_sph_2d.phi)
    for n in tqdm(range(rows * cols)):
        i = n % cols
        j = n // cols
        ret[j, i] = ExtensionTest.d_radon3d_dr_v2(volume_data, voxel_spacing, output_grid_sph_2d.phi[j, i].unsqueeze(0),
                                                  output_grid_sph_2d.theta[j, i].unsqueeze(0),
                                                  output_grid_sph_2d.r[j, i].unsqueeze(0), 500)
    return ret


def unflip_angle(angle: float | torch.Tensor, low: float, high: float) -> float | torch.Tensor:
    return low + torch.remainder(angle - low, high - low)


def grid_sample_sinogram3d_smoothed(sinogram3d: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
                                    r_values: torch.Tensor, *, i_mapping: LinearMapping, j_mapping: LinearMapping,
                                    k_mapping: LinearMapping, a_count: int = 6, b_count: int = 16,
                                    sigma: float | None = None):
    """
    Sample the given sinogram at the given phi, theta, r spherical coordinates, with extra samples in a Gaussian layout
    around the sampling positions to make the sampling more even over S^2, even if the point distribution is less even.

    :param sinogram3d: A 3D tensor
    :param phi_values: A tensor
    :param theta_values: A tensor matching size of `phi_values`
    :param r_values: A tensor matching size of `phi_values`
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

    assert phi_values.device == device
    assert theta_values.device == device
    assert r_values.device == device
    assert len(sinogram3d.size()) == 3
    assert phi_values.size() == theta_values.size()
    assert theta_values.size() == r_values.size()
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
    cp = phi_values.cos()
    sp = phi_values.sin()
    ct = theta_values.cos()
    st = theta_values.sin()
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
    new_thetas = torch.clamp(rotated_vectors[..., 2], -1., 1.).asin().flatten(-2)
    new_rs = r_values.unsqueeze(-1).expand(-1, -1, new_phis.size()[
        -1]).clone()  # need to clone otherwise it's just a tensor view, not a tensor in its own right

    del rotated_vectors

    # wrapping sinogram coordinates that lie outside the bounds
    theta_div = torch.div(new_thetas + .5 * torch.pi, torch.pi, rounding_mode="floor")
    theta_flip = torch.fmod(theta_div.to(dtype=torch.int32).abs(), 2).to(dtype=torch.bool)
    phi_div = torch.div(new_phis + .5 * torch.pi, torch.pi, rounding_mode="floor")
    phi_flip = torch.fmod(phi_div.to(dtype=torch.int32).abs(), 2).to(dtype=torch.bool)

    new_thetas -= torch.pi * theta_div
    new_phis -= torch.pi * phi_div

    new_thetas[torch.logical_and(phi_flip, torch.logical_not(theta_flip))] *= -1.
    new_rs[torch.logical_xor(theta_flip, phi_flip)] *= -1.

    del theta_div, theta_flip, phi_div, phi_flip

    # Sampling at all the perturbed orientations:
    grid = torch.stack((i_mapping(new_rs), j_mapping(new_thetas), k_mapping(new_phis)), dim=-1)
    samples = Extension.grid_sample3d(sinogram3d, grid, address_mode="wrap")

    del grid

    ##
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_phis.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average phi_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_thetas.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average theta_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(torch.einsum('i,...i->...', w_values.repeat(b_count), new_rs.cpu()).cpu())
    axes.axis('square')
    axes.set_title("average r_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    ##

    # Applying the weights and summing along the last dimension for an output equal in size to the input tensors of
    # coordinates:
    return torch.einsum('i,...i->...', w_values.repeat(b_count).to(device=device), samples)
