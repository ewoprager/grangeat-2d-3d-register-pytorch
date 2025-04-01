import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from registration.lib.structs import *
import registration.lib.geometry as geometry

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
        ret[j, i] = ExtensionTest.d_radon3d_dr_v2(volume_data, voxel_spacing, output_grid_sph_2d.phi[j, i].unsqueeze(0),
                                                  output_grid_sph_2d.theta[j, i].unsqueeze(0),
                                                  output_grid_sph_2d.r[j, i].unsqueeze(0), 500)
    return ret


def grid_sample_sinogram3d_smoothed(sinogram3d: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
                                    r_values: torch.Tensor, *, i_mapping: LinearMapping, j_mapping: LinearMapping,
                                    k_mapping: LinearMapping, a_count: int = 6, b_count: int = 16,
                                    sigma: float | None = None):
    """
    Sample the given sinogram at the given phi, theta, r spherical coordinates, with extra samples in a Gaussian layout
    around the sampling positions to make the sampling more even over S^2, even if the point distribution is less even.

    :param sinogram3d: A 3D tensor
    :param phi_values: A tensor of size (n, m)
    :param theta_values: A tensor of size (n, m)
    :param r_values: A tensor of size (n, m)
    :param i_mapping: Mapping from r to sinogram texture coordinate (-1, 1)
    :param j_mapping: Mapping from theta to sinogram texture coordinate (-1, 1)
    :param k_mapping: Mapping from phi to sinogram texture coordinate (-1, 1)
    :param a_count: Number of radial values at which to make extra samples in the Gaussian pattern
    :param b_count: Number of samples to make at each radius in the Gaussian pattern
    :param sigma: The standard deviation of the Gaussian pattern. Optional; if not provided, a sensible value is
    determined from the phi count in the given sinogram
    :return: A tensor of size (n, m) - the weighted sums of offset samples around the given coordinates.
    """
    device = sinogram3d.device

    assert phi_values.device == device
    assert theta_values.device == device
    assert r_values.device == device
    assert len(sinogram3d.size()) == 3
    assert len(phi_values.size()) == 2
    assert phi_values.size() == theta_values.size()
    assert theta_values.size() == r_values.size()
    assert sigma is None or sigma > 0.

    # Determine a sensible value of sigma if not provided
    if sigma is None:
        phi_count: int = sinogram3d.size()[0]
        sigma = 2. * torch.pi / (6. * float(phi_count))

    sigma = .05

    print("Sample smoothing with sigma = {:.3f}".format(sigma))

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
    # Multiplying for the perturbed unit vectors:
    offset_vectors = torch.stack((ca, -sa * sb, sa * cb), dim=-1).to(device=device)

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
    rotated_vectors = torch.einsum('...ij,...klj->...kli', rotation_matrices, offset_vectors)

    # Converting the resulting unit vectors back into new values of phi & theta, and expanding the r tensor to match in size:
    new_phis = torch.atan2(rotated_vectors[..., 1], rotated_vectors[..., 0]).flatten(-2)
    new_thetas = rotated_vectors[..., 2].asin().flatten(-2)
    new_rs = r_values.unsqueeze(-1).expand(-1, -1, new_phis.size()[-1])

    phis_under = new_phis < -.5 * torch.pi
    phis_over = new_phis > .5 * torch.pi
    new_phis[phis_under] += torch.pi
    new_phis[phis_over] -= torch.pi

    # new_rs[torch.logical_or(phis_under, phis_over)] *= -1.

    # Sampling at all the perturbed orientations:
    grid = torch.stack((i_mapping(new_rs), j_mapping(new_thetas), k_mapping(new_phis)), dim=-1)
    samples = torch.nn.functional.grid_sample(sinogram3d[None, None, :, :, :], grid[None, :, :, :, :])[0, 0]

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

    # Applying the weights and summing along the last dimension for an output equal in size to the input tensors of coordinates:
    return torch.einsum('i,...i->...', w_values.repeat(b_count).to(device=device), samples)


def resample_slice(sinogram3d: torch.Tensor, *, input_range: Sinogram3dRange, transformation: Transformation,
                   scene_geometry: SceneGeometry, output_grid: Sinogram2dGrid, smooth: bool = False) -> torch.Tensor:
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
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(output_grid_sph.phi.cpu())
    axes.axis('square')
    axes.set_title("phi_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(output_grid_sph.theta.cpu())
    axes.axis('square')
    axes.set_title("theta_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(output_grid_sph.r.cpu())
    axes.axis('square')
    axes.set_title("r_sph resampling values")
    axes.set_xlabel("r_pol")
    axes.set_ylabel("phi_pol")
    plt.colorbar(mesh)
    ##

    grid_range = LinearRange.grid_sample_range()
    i_mapping: LinearMapping = grid_range.get_mapping_from(input_range.r)
    j_mapping: LinearMapping = grid_range.get_mapping_from(input_range.theta)
    k_mapping: LinearMapping = grid_range.get_mapping_from(input_range.phi)

    if smooth:
        ret = grid_sample_sinogram3d_smoothed(sinogram3d, output_grid_sph.phi, output_grid_sph.theta, output_grid_sph.r,
                                              i_mapping=i_mapping, j_mapping=j_mapping, k_mapping=k_mapping)
    else:
        grid = torch.stack(
            (i_mapping(output_grid_sph.r), j_mapping(output_grid_sph.theta), k_mapping(output_grid_sph.phi)), dim=-1)
        ret = torch.nn.functional.grid_sample(sinogram3d[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]

    ret[need_sign_change] *= -1.
    return ret


# def resample_slice_fibonacci()
