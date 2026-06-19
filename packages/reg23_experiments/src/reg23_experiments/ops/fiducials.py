import logging
from typing import Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Float32, Float64, jaxtyped

from reg23_experiments.ops.optimisation import local_search

__all__ = ["ball_bearing_marker_function", "construct_synthetic_fiducial_3d", "refine_spherical_fiducial_2d",
           "refine_spherical_fiducial_3d", "refine_disk_fiducial_2d"]

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=typechecker)
def ball_bearing_marker_function(  #
        coordinates: Float64[torch.Tensor, "... 3"],  #
        *,  #
        value=30000,  #
        radius=2.5) -> Float32[torch.Tensor, "..."]:
    """
    Maps positions given in 3D Cartesian coordinates at mm scale to intensities
    :param coordinates: tensor of size (..., 3)
    :return: tensor of size (...)
    """
    dist = torch.linalg.vector_norm(coordinates, dim=-1)
    return (value * (dist < radius)).to(dtype=torch.float32)


@jaxtyped(typechecker=typechecker)
def construct_synthetic_fiducial_3d(  #
        *,  #
        size: torch.Size,  #
        spacing: Float64[torch.Tensor, "3"],  #
        origin: Float64[torch.Tensor, "3"],  #
        marker_function: Callable[[Float64[torch.Tensor, "... 3"]], Float32[torch.Tensor, "..."]],  #
        marker_positions: list[Float64[torch.Tensor, "3"]]) -> Float32[torch.Tensor, "p q r"]:
    """
    Constructs a tensor of the given size as the sum of the given markers.
    :param size: (z, y, x)
    :param spacing: (x, y, z)
    :param origin: (x, y, z) position of the (0, 0, 0) voxel relative to the origin
    :param marker_function: Function that maps a tensor of 3D positions (x, y, z) relative to the centre of a fiducial
    marker, to their intensity values in the 3D volume.
    :param marker_positions: A list of marker positions (x, y, z) relative to `origin`
    :return: A tensor of size `size` with the described markers added, and zeros everywhere else.
    """
    assert len(size) == 3

    voxel_x_positions = origin[0] + torch.arange(size[2], dtype=torch.float64) * spacing[0]
    voxel_y_positions = origin[1] + torch.arange(size[1], dtype=torch.float64) * spacing[1]
    voxel_z_positions = origin[2] + torch.arange(size[0], dtype=torch.float64) * spacing[2]
    voxel_z_positions, voxel_y_positions, voxel_x_positions = torch.meshgrid(voxel_z_positions, voxel_y_positions,
                                                                             voxel_x_positions)
    voxel_positions = torch.stack((voxel_x_positions, voxel_y_positions, voxel_z_positions), dim=-1)

    ret = torch.zeros(size=size, dtype=torch.float32)
    for pos in marker_positions:
        ret += marker_function(voxel_positions - pos)
    return ret


@jaxtyped(typechecker=typechecker)
def refine_spherical_fiducial_2d(*, image: Float32[torch.Tensor, "n m"], spacing: Float64[torch.Tensor, "2"],
                                 position: Float64[torch.Tensor, "2"], radius: float) -> Float64[torch.Tensor, "2"]:
    """
    Refines the segmented position of a highly attenuating spherical fiducial marker (e.g. a steel ball bearing) in a 2D
    image, given an initial segmented position and a known sphere radius.

    Maximises the following objective function: the linear correlation between the circular patch around the point, and
    the set of intensities expected when integrating through a sphere along one direction.
    :param image: The image in which the marker resides
    :param spacing: The spacing between the pixels in the image: (x, y)
    :param position: The initial segmented position of the marker, measured in mm from the top left of the image
    :param radius: The known radius of the sphere, in mm
    :return: A refined segmented position of the marker, measured in mm from the top left of the image
    """
    assert radius > 0.0
    image = image.cpu()
    spacing = spacing.cpu()
    output_device = position.device
    position = position.cpu()

    def objective(pos: torch.Tensor) -> torch.Tensor:
        pos_nograd = pos.detach()
        i0: int = max(0, int(((pos_nograd[0] - radius) / spacing[0]).floor().item()))
        i1: int = min(image.size()[1] - 1, int(((pos_nograd[0] + radius) / spacing[0]).ceil().item()))
        j0: int = max(0, int(((pos_nograd[1] - radius) / spacing[1]).floor().item()))
        j1: int = min(image.size()[0] - 1, int(((pos_nograd[1] + radius) / spacing[1]).ceil().item()))
        if i1 <= i0 or j1 <= j0:
            return torch.tensor(1.0)
        i1 += 1
        j1 += 1
        patch_pixel_xs = spacing[0] * torch.arange(i0, i1, dtype=torch.float64) - pos[0]
        patch_pixel_ys = spacing[1] * torch.arange(j0, j1, dtype=torch.float64) - pos[1]
        patch_pixel_ys, patch_pixel_xs = torch.meshgrid(patch_pixel_ys, patch_pixel_xs)
        square_distances = patch_pixel_xs.square() + patch_pixel_ys.square()
        patch_mask = square_distances < radius * radius
        patch = (radius * radius - square_distances[patch_mask]).sqrt()
        image_patch = image[j0:j1, i0:i1][patch_mask].to(dtype=torch.float64)
        return -torch.corrcoef(torch.stack((patch, image_patch)))[0, 1]

    if False:
        return local_search(  #
            starting_position=position,  #
            initial_step_size=spacing,  #
            objective_function=objective  #
        ).to(device=output_device)

    x = position.clone().detach().requires_grad_(True)
    lr = 0.1
    for _ in range(200):
        if x.grad is not None:
            x.grad.zero_()
        loss = objective(x)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
    return x.detach().to(device=output_device)


@jaxtyped(typechecker=typechecker)
def refine_disk_fiducial_2d(*, image: Float32[torch.Tensor, "n m"], spacing: Float64[torch.Tensor, "2"],
                            position: Float64[torch.Tensor, "2"], radius: float) -> Float64[torch.Tensor, "2"]:
    assert radius > 0.0
    image = image.cpu()
    spacing = spacing.cpu()
    output_device = position.device
    position = position.cpu()

    border_thickness = spacing.max()
    sq_inner_rad = (radius - border_thickness) * (radius - border_thickness)
    sq_outer_rad = (radius + border_thickness) * (radius + border_thickness)
    image_gradients = torch.stack(torch.gradient(image, spacing=tuple(s.item() for s in spacing.flip(dims=(0,)))),
                                  dim=-1).flip(dims=(-1,))
    image_gradients /= image_gradients.max()

    def objective(pos: torch.Tensor) -> torch.Tensor:
        pos_nograd = pos.detach()
        i0: int = max(0, int(((pos_nograd[0] - radius) / spacing[0]).floor().item()))
        i1: int = min(image.size()[1] - 1, int(((pos_nograd[0] + radius) / spacing[0]).ceil().item()))
        j0: int = max(0, int(((pos_nograd[1] - radius) / spacing[1]).floor().item()))
        j1: int = min(image.size()[0] - 1, int(((pos_nograd[1] + radius) / spacing[1]).ceil().item()))
        if i1 <= i0 or j1 <= j0:
            return pos.sum() * 0.0 + 1.0
        i1 += 1
        j1 += 1
        patch_pixel_xs = spacing[0] * torch.arange(i0, i1, dtype=torch.float64) - pos[0]
        patch_pixel_ys = spacing[1] * torch.arange(j0, j1, dtype=torch.float64) - pos[1]
        patch_pixel_positions = torch.stack(torch.meshgrid(patch_pixel_ys, patch_pixel_xs), dim=-1).flip(
            dims=(-1,))  # (n, m, 2)
        square_distances = patch_pixel_positions.square().sum(dim=-1)  # (n, m)
        patch_mask = torch.logical_and(square_distances > sq_inner_rad, square_distances < sq_outer_rad)
        unit_directions = -(
                patch_pixel_positions[patch_mask] / square_distances[patch_mask].sqrt().unsqueeze(-1)).flatten()
        img_grad_patch = image_gradients[j0:j1, i0:i1, :][patch_mask].flatten().to(dtype=torch.float64)
        return -(unit_directions * img_grad_patch).mean()

    if False:
        return local_search(  #
            starting_position=position,  #
            initial_step_size=torch.full((3,), 0.25 * radius, dtype=torch.float64),  #
            objective_function=objective  #
        ).to(device=output_device)

    x = position.clone().detach().requires_grad_(True)
    lr = 10.0
    for _ in range(200):
        if x.grad is not None:
            x.grad.zero_()
        loss = objective(x)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
    return x.detach().to(device=output_device)


@jaxtyped(typechecker=typechecker)
def refine_spherical_fiducial_3d(*, volume: Float32[torch.Tensor, "p q r"], spacing: Float64[torch.Tensor, "3"],
                                 position: Float64[torch.Tensor, "3"], radius: float) -> Float64[torch.Tensor, "3"]:
    assert radius > 0.0
    volume = volume.cpu()
    spacing = spacing.cpu()
    output_device = position.device
    position = position.cpu()

    border_thickness = spacing.max()
    sq_inner_rad = (radius - border_thickness) * (radius - border_thickness)
    sq_outer_rad = (radius + border_thickness) * (radius + border_thickness)
    volume_gradients = torch.stack(torch.gradient(volume, spacing=tuple(s.item() for s in spacing.flip(dims=(0,)))),
                                   dim=-1).flip(dims=(-1,))
    volume_gradients /= volume_gradients.max()

    def objective(pos: torch.Tensor) -> torch.Tensor:
        pos_nograd = pos.detach()
        i0: int = max(0, int(((pos_nograd[0] - radius) / spacing[0]).floor().item()))
        i1: int = min(volume.size()[2] - 1, int(((pos_nograd[0] + radius) / spacing[0]).ceil().item()))
        j0: int = max(0, int(((pos_nograd[1] - radius) / spacing[1]).floor().item()))
        j1: int = min(volume.size()[1] - 1, int(((pos_nograd[1] + radius) / spacing[1]).ceil().item()))
        k0: int = max(0, int(((pos_nograd[2] - radius) / spacing[2]).floor().item()))
        k1: int = min(volume.size()[0] - 1, int(((pos_nograd[2] + radius) / spacing[2]).ceil().item()))
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            return pos.sum() * 0.0 + 1.0
        i1 += 1
        j1 += 1
        k1 += 1
        patch_pixel_xs = spacing[0] * torch.arange(i0, i1, dtype=torch.float64) - pos[0]
        patch_pixel_ys = spacing[1] * torch.arange(j0, j1, dtype=torch.float64) - pos[1]
        patch_pixel_zs = spacing[2] * torch.arange(k0, k1, dtype=torch.float64) - pos[2]
        patch_pixel_positions = torch.stack(torch.meshgrid(patch_pixel_zs, patch_pixel_ys, patch_pixel_xs),
                                            dim=-1).flip(dims=(-1,))  # (p, q, r, 3)
        square_distances = patch_pixel_positions.square().sum(dim=-1)  # (p, q, r)
        patch_mask = torch.logical_and(square_distances > sq_inner_rad, square_distances < sq_outer_rad)
        unit_directions = -(
                patch_pixel_positions[patch_mask] / square_distances[patch_mask].sqrt().unsqueeze(-1)).flatten()
        vol_grad_patch = volume_gradients[k0:k1, j0:j1, i0:i1, :][patch_mask].flatten().to(dtype=torch.float64)
        return -(unit_directions * vol_grad_patch).mean()

    if False:
        return local_search(  #
            starting_position=position,  #
            initial_step_size=torch.full((3,), 0.25 * radius, dtype=torch.float64),  #
            objective_function=objective  #
        ).to(device=output_device)

    x = position.clone().detach().requires_grad_(True)
    lr = 10.0
    for _ in range(200):
        if x.grad is not None:
            x.grad.zero_()
        loss = objective(x)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
    return x.detach().to(device=output_device)
