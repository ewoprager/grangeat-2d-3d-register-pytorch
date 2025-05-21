import torch

from registration.lib import grangeat
from registration import data
from registration.lib.sinogram import *


def generate_new_drr(cache_directory: str, ct_volume_path: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                     *, device, save_to_cache=True):
    # transformation = Transformation(torch.tensor([0., 0., 0.]),
    #                                 torch.tensor([0., 0., 200.])).to(device=device)
    # transformation = Transformation.zero(device=volume_data.device)
    transformation = Transformation.random(device=volume_data.device)
    logger.info("Generating DRR at transformation:\n\tr = {}\n\tt = {}...".format(transformation.rotation,
                                                                                  transformation.translation))

    # # plot_drr(drr_image, ticks=False)
    # drr_image = drr_image[0, 0]
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_image.cpu())
    # axes.axis('square')
    # plt.colorbar(mesh)

    detector_spacing = torch.tensor([.25, .25])
    scene_geometry = SceneGeometry(source_distance=1000.)

    drr_image = geometry.generate_drr(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
                                      detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                      output_size=torch.Size([1000, 1000]))

    logger.info("DRR generated.")

    logger.info("Calculating 2D sinogram (the fixed image)...")

    sinogram2d_counts = 1024
    image_diag: float = (
            detector_spacing * torch.tensor(drr_image.size(), dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d = grangeat.calculate_fixed_image(drr_image, source_distance=scene_geometry.source_distance,
                                                 detector_spacing=detector_spacing, output_grid=sinogram2d_grid)

    logger.info("DRR sinogram calculated.")

    if save_to_cache and ct_volume_path is not None:
        save_path = cache_directory + "/drr_spec_{}.pt".format(data.deterministic_hash(ct_volume_path))
        torch.save(DrrSpec(ct_volume_path, detector_spacing, scene_geometry, drr_image, sinogram2d, sinogram2d_range,
                           transformation), save_path)
        logger.info("DRR sinogram saved to '{}'".format(save_path))

    return detector_spacing, scene_geometry, drr_image, sinogram2d, sinogram2d_range, transformation
