import logging
import math

import torch

from reg23_experiments.data.sinogram import DrrSpec
from reg23_experiments.data.structs import SceneGeometry, Transformation
from reg23_experiments.io.helpers import deterministic_hash_string
from reg23_experiments.ops import geometry
from reg23_experiments.ops.image import gaussian_blur_2d

__all__ = ["add_scatter", "add_poisson_noise", "apply_log_transformation", "make_drr_realistic",
           "generate_drr_as_target"]

logger = logging.getLogger(__name__)


def add_scatter(image: torch.Tensor, sigma: float | tuple[float, float], alpha: float = 0.3):
    """
    image: (H, W), linear intensity (before log)
    sigma: Gaussian std dev in pixels
    alpha: scatter strength (SPR approx)
    """
    return image + alpha * gaussian_blur_2d(image, sigma=sigma)


def add_poisson_noise(image: torch.Tensor, photon_count: float = 1e4) -> torch.Tensor:
    """
    :param image: (H, W), linear intensity (before log), positive
    :param photon_count: scaling factor for photon count, controls noise level
    :return: image with poisson noise
    """
    scaled = image * photon_count
    noisy = torch.poisson(scaled)
    noisy = noisy / photon_count
    return noisy


def apply_log_transformation(image: torch.Tensor, epsilon: float = 1.e-4) -> torch.Tensor:
    return -(image + epsilon).log()


def make_drr_realistic(drr: torch.Tensor, *,  #
                       detector_spacing: torch.Tensor,  #
                       scatter_spread_mm: float = 30.0,  #
                       scatter_alpha: float = 0.3,  #
                       poisson_photon_count: float = 1e4) -> torch.Tensor:
    # simulate scatter
    scatter_sigmas = scatter_spread_mm / detector_spacing
    scatter_sigmas = (scatter_sigmas[0].item(), scatter_sigmas[1].item())
    drr = add_scatter(drr, sigma=scatter_sigmas, alpha=scatter_alpha)

    # add poisson noise
    drr = add_poisson_noise(drr, photon_count=poisson_photon_count)

    return drr


def generate_drr_as_target(cache_directory: str, ct_volume_path: str, volume_data: torch.Tensor,
                           voxel_spacing: torch.Tensor, *, save_to_cache=True, size: torch.Size | None = None,
                           transformation: Transformation | None = None):
    # transformation = Transformation(torch.tensor([0., 0., 0.]),
    #                                 torch.tensor([0., 0., 200.])).to(device=device)
    # transformation = Transformation.zero(device=volume_data.device)
    if transformation is None:
        transformation = Transformation.random_uniform(device=volume_data.device)
    logger.info("Generating DRR at transformation:\n\tr = {}\n\tt = {}...".format(transformation.rotation,
                                                                                  transformation.translation))

    # # plot_drr(drr_image, ticks=False)
    # drr_image = drr_image[0, 0]
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_image.cpu())
    # axes.axis('square')
    # plt.colorbar(mesh)

    if size is None:
        side_length = int(
            math.ceil(pow(volume_data.size()[0] * volume_data.size()[1] * volume_data.size()[2], 1.0 / 3.0)))
        size = torch.Size([side_length, side_length])

    detector_spacing = 200.0 / torch.tensor(size)  # assume the detector is 200 x 200 mm in size
    scene_geometry = SceneGeometry(source_distance=1000.)

    drr_image = geometry.generate_drr(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
                                      detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                      output_size=size)

    drr_image = torch.exp(-drr_image)

    drr_image = make_drr_realistic(drr_image, detector_spacing=detector_spacing)

    # apply log transformation
    drr_image = apply_log_transformation(drr_image)

    logger.info("DRR generated.")

    # logger.info("Calculating 2D sinogram (the fixed image)...")
    #
    # sinogram2d_counts = max(drr_image.size()[0], drr_image.size()[1])
    # image_diag: float = (detector_spacing.flip(dims=(0,)) * torch.tensor(
    #     drr_image.size(), dtype=torch.float32)).square().sum().sqrt(
    #
    # ).item()
    # sinogram2d_range = Sinogram2dRange(
    #     LinearRange(-.5 * torch.pi, .5 * torch.pi), LinearRange(-.5 * image_diag, .5 * image_diag))
    # sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)
    #
    # sinogram2d = grangeat.calculate_fixed_image(
    #     drr_image, source_distance=scene_geometry.source_distance, detector_spacing=detector_spacing,
    #     output_grid=sinogram2d_grid)
    #
    # logger.info("DRR sinogram calculated.")

    if save_to_cache and ct_volume_path is not None:
        save_path = cache_directory + "/drr_spec_{}.pt".format(deterministic_hash_string(ct_volume_path))
        torch.save(DrrSpec(ct_volume_path, detector_spacing, scene_geometry, drr_image, transformation), save_path)
        logger.info("DRR sinogram saved to '{}'".format(save_path))

    return detector_spacing, scene_geometry, drr_image, transformation
