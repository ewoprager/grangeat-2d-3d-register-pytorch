import logging

import torch
from jaxtyping import Float32, Float64

from reg23_experiments.ops.optimisation import local_search

__all__ = ["refine_spherical_fiducial_2d"]

logger = logging.getLogger(__name__)


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
        i0: int = max(0, int(((pos[0] - radius) / spacing[0]).floor().item()))
        i1: int = min(image.size()[1] - 1, int(((pos[0] + radius) / spacing[0]).ceil().item()))
        j0: int = max(0, int(((pos[1] - radius) / spacing[1]).floor().item()))
        j1: int = min(image.size()[0] - 1, int(((pos[1] + radius) / spacing[1]).ceil().item()))
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

    return local_search(  #
        starting_position=position,  #
        initial_step_size=spacing,  #
        objective_function=objective  #
    ).to(device=output_device)


def refine_spherical_fiducial_3d(*, volume: Float32[torch.Tensor, "p q r"], spacing: Float64[torch.Tensor, "3"],
                                 position: Float64[torch.Tensor, "3"], radius: float) -> Float64[torch.Tensor, "3"]:
    assert radius > 0.0
    volume = volume.cpu()
    spacing = spacing.cpu()
    output_device = position.device
    position = position.cpu()

    marker_volume = 4.0 * torch.pi * radius * radius * radius / 3.0
    voxel_volume = spacing.prod()
    min_voxels_per_marker = int((0.9 * marker_volume / voxel_volume).floor().item())
    sorted_intensities, _ = volume.flatten().sort()
    estimated_marker_intensity = sorted_intensities[-min_voxels_per_marker:].median()

    def objective(pos: torch.Tensor) -> torch.Tensor:
        i0: int = max(0, int(((pos[0] - radius) / spacing[0]).floor().item()))
        i1: int = min(volume.size()[2] - 1, int(((pos[0] + radius) / spacing[0]).ceil().item()))
        j0: int = max(0, int(((pos[1] - radius) / spacing[1]).floor().item()))
        j1: int = min(volume.size()[1] - 1, int(((pos[1] + radius) / spacing[1]).ceil().item()))
        k0: int = max(0, int(((pos[2] - radius) / spacing[2]).floor().item()))
        k1: int = min(volume.size()[0] - 1, int(((pos[2] + radius) / spacing[2]).ceil().item()))
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            return torch.tensor(1.0)
        i1 += 1
        j1 += 1
        k1 += 1
        patch_pixel_xs = spacing[0] * torch.arange(i0, i1, dtype=torch.float64) - pos[0]
        patch_pixel_ys = spacing[1] * torch.arange(j0, j1, dtype=torch.float64) - pos[1]
        patch_pixel_zs = spacing[2] * torch.arange(k0, k1, dtype=torch.float64) - pos[2]
        patch_pixel_zs, patch_pixel_ys, patch_pixel_xs = torch.meshgrid(patch_pixel_zs, patch_pixel_ys, patch_pixel_xs)
        square_distances = patch_pixel_xs.square() + patch_pixel_ys.square() + patch_pixel_zs.square()
        patch_mask = square_distances < radius * radius
        volume_patch = volume[k0:k1, j0:j1, i0:i1][patch_mask]
        return -torch.dot(volume_patch, torch.full_like(volume_patch, estimated_marker_intensity))

    return local_search(  #
        starting_position=position,  #
        initial_step_size=spacing,  #
        objective_function=objective  #
    ).to(device=output_device)
