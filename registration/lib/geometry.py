import copy

import torch
import pyvista as pv

from registration.lib.structs import Sinogram2dGrid, Sinogram3dGrid, Transformation, SceneGeometry

import reg23


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
    r2s_x_y = xs.square() + ys.square()
    thetas = torch.atan2(zs, r2s_x_y.sqrt())
    rs = (r2s_x_y + zs.square()).sqrt()
    return Sinogram3dGrid(phis, thetas, rs).unflip()


def fixed_polar_to_moving_spherical(input_grid: Sinogram2dGrid, *, ph_matrix: torch.Tensor,
                                    plot: bool = False) -> Sinogram3dGrid:
    assert input_grid.device_consistent()
    assert ph_matrix.device == input_grid.phi.device

    fixed_image_grid_cartesian = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)

    if plot:
        pl = pv.Plotter()
        cartesian_points = fixed_image_grid_cartesian.flatten(end_dim=-2)
        pl.add_points(cartesian_points.cpu().numpy())  # , scalars=scalars.flatten().cpu().numpy())
        pl.show()
        del cartesian_points

    return moving_cartesian_to_moving_spherical(fixed_image_grid_cartesian)


def ray_cuboid_distance(cuboid_centre: torch.Tensor, cuboid_half_sizes: torch.Tensor, ray_points: torch.Tensor,
                        ray_unit_directions: torch.Tensor) -> torch.Tensor:
    """
    Determines the lengths of ray intersections between given rays and an axis-aligned cuboid.
    :param cuboid_centre: A tensor of size (,3); the position of the centre of the cuboid
    :param cuboid_half_sizes: A tensor of size (,3); the half sizes of the cuboid in the x-, y- and z-directions
    :param ray_points: A tensor of size (..., 3); point(s) through which the input rays each pass, either one for all, or one per ray
    :param ray_unit_directions: A tensor of size matching ray_points or (..., 3) if ray_points is of size (,3); unit vectors in the directions of each input ray
    :return: A tensor of size ray_points.size()[:-1]; the lengths of the intersections of each ray with the cuboid
    """
    assert cuboid_centre.device == cuboid_half_sizes.device
    assert cuboid_half_sizes.device == ray_points.device
    assert ray_points.device == ray_unit_directions.device
    assert cuboid_centre.size() == torch.Size([3])  # size = (,3)
    assert cuboid_half_sizes.size() == torch.Size([3])  # size = (,3)
    assert ray_unit_directions.size()[-1] == 3  # size = (..., 3)
    assert ray_points.size() == ray_unit_directions.size() or ray_points.size() == torch.Size([3])  # size = ([...], 3)

    # -----
    # The 6 planes of the cuboid, each defined by intersection point and unit normal
    # -----
    # for calculating intersection points and normals for each of the 6 planes of the cuboid
    deltas = torch.concat((  #
        cuboid_half_sizes.unsqueeze(0).expand(3, -1),  #
        (-cuboid_half_sizes).unsqueeze(0).expand(3, -1)  #
    ))  # size = (6, 3)
    # a point for each of the 6 planes of the cuboid that each plane intersects
    plane_points = cuboid_centre + deltas  # size = (6, 3)
    # a unit normal vector for each of the 6 planes of the cuboid, pointing out of the cuboid
    plane_normals = torch.nn.functional.normalize(
        torch.eye(3, device=deltas.device).repeat(2, 1) * deltas)  # size = (6, 3)

    # -----
    # Finding where the rays intersect, if they do
    # -----
    # dot products between each of the 6 planes and each of the input ray direction vectors
    dots = torch.einsum("ji,...i->...j", plane_normals, ray_unit_directions)  # size = (..., 6)
    # for dealing with rays near parallel to cuboid planes
    epsilon = 1.0e-8
    # the ray points expanded x6 along the penultimate dimension for manipulating with the cuboid planes
    ray_points_expanded = ray_points.unsqueeze(-2).expand(*ray_points.size()[:-1], 6, 3)  # size = ([...], 6, 3)
    # the lambda value (position along each ray away from the ray point) of each plane for each ray
    lambdas = (torch.einsum("ji,...ji->...j", plane_normals, plane_points - ray_points_expanded)  # size = ([...], 6)
               / dots)  # size = (..., 6)
    # lambda values for planes parallel to rays (`dots` small) or ray is coming out of (dots positive) set to -inf
    lambdas_masked = torch.where(dots < -epsilon, lambdas,
                                 torch.tensor(float("-inf"), device=lambdas.device))  # size = (..., 6)
    # finding entry lambda and associated plane index by finding max of above
    entry_lambdas, entry_indices = lambdas_masked.max(dim=-1)  # size of both = (...)
    # entry positions
    entry_points = ray_points + entry_lambdas.unsqueeze(-1) * ray_unit_directions  # size = (..., 3)

    # -----
    # Checking whether rays intersect, by comparing plane intersection point with other two cuboid plane pairs
    # -----
    # extracting indices of other two cuboid plane pairs from intersection plane index
    check_indices = torch.remainder(
        entry_indices.unsqueeze(-1) + torch.tensor([1, 2, 4, 5], device=entry_indices.device), 6)  # size = (..., 4)
    # expanding this for manipulation with plane points and normals
    indices_expanded = check_indices.unsqueeze(-1).expand(*check_indices.size(), 3)  # size = (..., 4, 3)
    # expanding plane points and normals for manipulation with indices
    plane_points_expanded = plane_points.expand((*check_indices.size()[:-1], 6, 3))  # size = (..., 6, 3)
    plane_normals_expanded = plane_normals.expand((*check_indices.size()[:-1], 6, 3))  # size = (..., 6, 3)
    # filtering plane points and normals by indices
    plane_points_filtered = plane_points_expanded.gather(-2, indices_expanded)  # size = (..., 4, 3)
    plane_normals_filtered = plane_normals_expanded.gather(-2, indices_expanded)  # size = (..., 4, 3)
    # checking entry points against filtered planes to determine if ray misses cuboid entirely
    rays_hit = (torch.einsum("...ji,...ji->...j", entry_points.unsqueeze(-2) - plane_points_filtered,
                             plane_normals_filtered) < epsilon).all(dim=-1)  # size = (...)

    # -----
    # Finding where rays exit cuboid and returning
    # -----
    # lambda values for planes parallel to rays (`dots` small) or ray is going into (dots negative) set to inf
    lambdas_masked = torch.where(dots > epsilon, lambdas,
                                 torch.tensor(float("inf"), device=lambdas.device))  # size = (.., 6)
    # finding exit lambda by finding min of above
    exit_lambdas, _ = lambdas_masked.min(dim=-1)  # size = (...)
    # finding distance from lambda difference and setting distance for rays that miss to zero
    return torch.where(rays_hit, (exit_lambdas - entry_lambdas).abs(),
                       torch.tensor(0.0, device=exit_lambdas.device))  # size = (...)


# This uses diffdrr
# def generate_drr(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
#                  detector_spacing: torch.Tensor, scene_geometry: SceneGeometry, output_size: torch.Size) -> (
#         torch.Tensor):
#     image = torchio.ScalarImage(tensor=volume_data.unsqueeze(0))
#
#     subject = diffdrr.data.read(image, spacing=voxel_spacing, orientation=None)
#
#     del image
#
#     # I believe that the detector array lies on the x-z plane, with x down, and z to the left (and so y outward)
#     drr_generator = diffdrr.drr.DRR(subject,  # An object storing the CT volume, origin, and voxel spacing  #
#                                     sdd=scene_geometry.source_distance,
#                                     # Source-to-detector distance (i.e., focal length)
#                                     height=output_size[0], width=output_size[1],
#                                     # Image height (if width is not provided, the generated DRR is square)
#                                     delx=detector_spacing[0],  # Pixel spacing (in mm)
#                                     dely=detector_spacing[1]).to(volume_data.device)
#
#
#     # affine = torch.eye(4)
#     # affine[0, 0] = voxel_spacing[0]
#     # affine[1, 1] = voxel_spacing[1]
#     # affine[2, 2] = voxel_spacing[2]
#     # subject = diffdrr.data.read(volume=torchio.ScalarImage(tensor=volume_data.unsqueeze(0), affine=affine),
#     #                             orientation=None)
#     #
#     # drr_generator = diffdrr.drr.DRR(subject, sdd=scene_geometry.source_distance, height=output_size[0],
#     # width=output_size[1],
#     #                       delx=detector_spacing[0], dely=detector_spacing[1]).to(volume_data.device)
#
#     del subject
#
#     ret = drr_generator(transformation.rotation.unsqueeze(0), transformation.translation.unsqueeze(0),
#                         parameterization="axis_angle")[0, 0]
#
#     del drr_generator
#
#     return ret

def generate_drr(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
                 detector_spacing: torch.Tensor, scene_geometry: SceneGeometry, output_size: torch.Size) -> (
        torch.Tensor):
    assert len(output_size) == 2
    assert voxel_spacing.size() == torch.Size([3])
    assert detector_spacing.size() == torch.Size([2])
    assert transformation.device_consistent()
    img_width: int = output_size[1]
    img_height: int = output_size[0]
    h_matrix_inv = transformation.inverse().get_h()

    return reg23.project_drr(volume_data, voxel_spacing, h_matrix_inv, scene_geometry.source_distance, img_width,
                             img_height, scene_geometry.fixed_image_offset.to(dtype=torch.float64),
                             detector_spacing)


def generate_drr_cuboid_mask(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
                             detector_spacing: torch.Tensor, scene_geometry: SceneGeometry,
                             output_size: torch.Size) -> (torch.Tensor):
    device = volume_data.device
    assert len(output_size) == 2
    assert voxel_spacing.size() == torch.Size([3])
    assert detector_spacing.size() == torch.Size([2])
    assert detector_spacing.device == device
    assert transformation.device_consistent()
    assert transformation.translation.device == device
    assert voxel_spacing.device == device
    img_width: int = output_size[1]
    img_height: int = output_size[0]
    h_matrix_inv = transformation.inverse().get_h(device=device)

    return reg23.project_drr_cuboid_mask(torch.tensor(volume_data.size(), device=volume_data.device).flip(dims=(0,)),
                                         voxel_spacing, h_matrix_inv, scene_geometry.source_distance, img_width,
                                         img_height,
                                         scene_geometry.fixed_image_offset.to(device=device, dtype=torch.float64),
                                         detector_spacing)


def generate_drr_python(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
                        detector_spacing: torch.Tensor, scene_geometry: SceneGeometry, output_size: torch.Size,
                        get_ray_intersection_fractions: bool = False) -> torch.Tensor:
    device = volume_data.device
    assert len(output_size) == 2
    assert voxel_spacing.size() == torch.Size([3])
    assert detector_spacing.size() == torch.Size([2])
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
    samples_per_ray: int = torch.tensor(volume_data.size()).max().item()
    step_size: float = volume_diag_length.item() / float(samples_per_ray)

    h_matrix_inv = transformation.inverse().get_h(device=device).to(dtype=torch.float32)

    starts = source_position + lambda_start * directions
    starts_homogeneous = torch.cat((starts, torch.ones_like(starts[..., 0], device=device).unsqueeze(-1)), dim=-1)
    starts = torch.einsum('ji,...i->...j', h_matrix_inv, starts_homogeneous)[..., 0:3]

    directions_homogeneous = torch.cat((directions, torch.zeros_like(directions[..., 0], device=device).unsqueeze(-1)),
                                       dim=-1)
    directions = torch.einsum('ji,...i->...j', h_matrix_inv, directions_homogeneous)[..., 0:3]
    deltas = directions * step_size

    deltas_texture = 2. * deltas / volume_diag
    grid = (2. * starts / volume_diag).to(dtype=torch.float32)
    ret: torch.Tensor = torch.zeros(output_size, device=device)
    for i in range(samples_per_ray):
        ret += torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
        grid += deltas_texture
    ret = step_size * ret

    if get_ray_intersection_fractions:
        cuboid_in_half_sizes = 0.5 * volume_diag
        cuboid_in_centre = torch.zeros(3, device=cuboid_in_half_sizes.device)
        cuboid_above_half_sizes = copy.deepcopy(cuboid_in_half_sizes)
        cuboid_above_half_sizes[2] *= 4.0
        cuboid_below_half_sizes = cuboid_above_half_sizes
        z_sum = cuboid_in_half_sizes[2] + cuboid_above_half_sizes[2]
        cuboid_above_centre = torch.tensor([0.0, 0.0, z_sum], device=cuboid_in_half_sizes.device)
        cuboid_below_centre = torch.tensor([0.0, 0.0, -z_sum], device=cuboid_in_half_sizes.device)
        ray_point = torch.einsum(  #
            'ji,i->j',  #
            h_matrix_inv,  #
            torch.cat((source_position, torch.tensor([1.0], device=source_position.device)), dim=-1)  #
        )[0:3]
        ray_distances_in = ray_cuboid_distance(cuboid_in_centre, cuboid_in_half_sizes, ray_point, directions)
        denominator = ray_distances_in + (
                ray_cuboid_distance(cuboid_above_centre, cuboid_above_half_sizes, ray_point, directions) +  #
                ray_cuboid_distance(cuboid_below_centre, cuboid_below_half_sizes, ray_point, directions))
        mask = torch.where(denominator > 1.0e-8, ray_distances_in / denominator, 1.0)
        ret = torch.stack((ret, mask))

    return ret


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
