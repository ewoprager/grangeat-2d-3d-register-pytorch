import logging
import math
import pathlib
from typing import Any, Type

import torch

import reg23
from reg23_experiments.program.lib.structs import Error
from reg23_experiments.registration import data, drr, pre_computed
from reg23_experiments.registration.interface.lib.structs import Cropping, Target
from reg23_experiments.registration.lib import grangeat
from reg23_experiments.registration.lib.sinogram import Sinogram, SinogramType
from reg23_experiments.registration.lib.structs import (LinearRange, Sinogram2dGrid, Sinogram2dRange, Transformation, )
from reg23_experiments.program import dag_updater

__all__ = ["load_ct", "refresh_vif", "load_target_image", "set_synthetic_target_image",
           "refresh_hyperparameter_dependent", "refresh_hyperparameter_dependent_grangeat",
           "refresh_hyperparameter_dependent_grangeat", "refresh_mask_transformation_dependent_grangeat"]

logger = logging.getLogger(__name__)


@dag_updater(names_returned=["ct_volumes", "ct_spacing"])
def load_ct(ct_path: str, device) -> dict[str, Any]:
    ct_volumes, ct_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor="mipmap")
    ct_volumes = [ct_volume.to(device=device, dtype=torch.float32) for ct_volume in ct_volumes]
    ct_spacing = ct_spacing.to(device=device)

    return {"ct_volumes": ct_volumes, "ct_spacing": ct_spacing}


@dag_updater(names_returned=["sinogram_size", "ct_sinograms"])
def refresh_vif(self) -> dict[str, Any] | Error:
    this_sinogram_size = int(
        math.ceil(pow(self.ct_volumes[0].numel(), 1.0 / 3.0))) if self._sinogram_size is None else self._sinogram_size

    def get_sinogram(sinogram_type: Type[SinogramType], downsample_level: int) -> Sinogram | None:
        downsample_factor = int(2 ** downsample_level)
        downsampled_sinogram_size = this_sinogram_size // downsample_factor
        sinogram3d = None
        sinogram_hash = data.deterministic_hash_sinogram(self.ct_path, sinogram_type, downsampled_sinogram_size,
                                                         downsample_factor)
        volume_spec = data.load_cached_volume(self._cache_directory, sinogram_hash)
        if volume_spec is not None:
            _, sinogram3d = volume_spec
        if sinogram3d is None:
            res = pre_computed.calculate_volume_sinogram(self._cache_directory, self.ct_volumes[downsample_level],
                                                         voxel_spacing=self.ct_spacing_original * float(
                                                             downsample_factor), ct_volume_path=self.ct_path,
                                                         volume_downsample_factor=downsample_factor, save_to_cache=True,
                                                         sinogram_size=downsampled_sinogram_size,
                                                         sinogram_type=sinogram_type)
            if res is None:
                return None
            sinogram3d, _ = res
        return sinogram3d

    sinogram3ds = {tp: [get_sinogram(tp, level) for level in range(len(self.ct_volumes))] for tp in
                   self._sinogram_types}

    for tp, sinogram_list in sinogram3ds.items():
        for i, sinogram in enumerate(sinogram_list):
            if sinogram is None:
                return Error("Failed to create sinogram at level {} of type {}; not enough memory?"
                             "".format(i, tp.__name__))

    return {"sinogram_size": this_sinogram_size, "ct_sinograms": sinogram3ds}


@dag_updater(names_returned=["source_distance", "images_2d_full", "fixed_image_spacing", "transformation_gt"])
def load_target_image(ct_spacing: torch.Tensor, target: Target, device) -> dict[str, Any]:
    transformation_ground_truth = None
    # if self.target.xray_path is None:
    #     # Load /
    # else:
    #     Load the given X-ray
    image_2d_full, fixed_image_spacing, scene_geometry = data.read_dicom(  #
        target.xray_path, downsample_to_ct_spacing=ct_spacing.mean().item())
    image_2d_full = image_2d_full.to(device=device)

    if target.flipped:
        logger.info("Flipping target image horizontally.")
        image_2d_full = image_2d_full.flip(dims=(1,))

    # Generating X-ray mipmap
    down_sampler = torch.nn.AvgPool2d(2)
    images_2d_full = [image_2d_full]
    while min(images_2d_full[-1].size()) > 1:
        images_2d_full.append(down_sampler(images_2d_full[-1].unsqueeze(0))[0])

    return {"source_distance": scene_geometry.source_distance, "images_2d_full": images_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}
    self.hyperparameters = HyperParameters.zero(self.images_2d_full[0].size())

    if not self.suppress_callbacks and self._target_change_callback is not None:
        self._target_change_callback()


@dag_updater(names_returned=["source_distance", "images_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_synthetic_target_image(ct_path: str, ct_spacing: torch.Tensor, ct_volumes: list[torch.Tensor],
                               new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool,
                               cache_directory: str) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr and ct_path is not None:
        drr_spec = data.load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, ct_volumes[0], ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size)

    fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    # Generating X-ray mipmap
    down_sampler = torch.nn.AvgPool2d(2)
    images_2d_full = [image_2d_full]
    while min(images_2d_full[-1].size()) > 1:
        images_2d_full.append(down_sampler(images_2d_full[-1].unsqueeze(0))[0])

    return {"source_distance": scene_geometry.source_distance, "images_2d_full": images_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}


@dag_updater(names_returned=["image_2d_downsample_level"])
def refresh_image_2d_downsample_level(fixed_image_spacing: torch.Tensor, downsample_level: int,
                                      ct_spacing: torch.Tensor) -> dict[str, Any]:
    extra_ds_level: int = (ct_spacing.mean() / fixed_image_spacing.mean()).log2().floor().to(dtype=torch.int64).item()
    return {"image_2d_downsample_level": downsample_level + extra_ds_level}


@dag_updater(names_returned=["cropped_target", "fixed_image_offset", "translation_offset", "fixed_image_size"])
def refresh_hyperparameter_dependent(images_2d_full: list[torch.Tensor], fixed_image_spacing: torch.Tensor,
                                     cropping: Cropping, downsample_level: int, source_offset: torch.Tensor,
                                     image_2d_downsample_level: int) -> dict[str, Any]:
    # Cropping for the fixed image
    cropped_target = cropping.to_downsample_level(  #
        downsample_level, image_size=images_2d_full[image_2d_downsample_level].size()  #
    ).apply(images_2d_full[image_2d_downsample_level])

    # The fixed image is offset to adjust for the cropping, and according to the source offset
    # This isn't affected by downsample level
    fixed_image_offset = (fixed_image_spacing * cropping.get_centre_offset(images_2d_full[0].size()) - source_offset)

    # The translation offset prevents the source offset parameters from fighting the translation parameters in
    # the optimisation
    translation_offset = -source_offset

    return {"cropped_target": cropped_target, "fixed_image_offset": fixed_image_offset,
            "translation_offset": translation_offset, "fixed_image_size": cropped_target.size()}


@dag_updater(names_returned=["sinogram2d_grid_unshifted", "sinogram2d_grid"])
def refresh_hyperparameter_dependent_grangeat(cropped_target: torch.Tensor, fixed_image_offset: torch.Tensor,
                                              fixed_image_spacing: torch.Tensor, downsample_level: int, device) -> dict[
    str, Any]:
    cropped_target_size = cropped_target.size()
    sinogram2d_counts = max(cropped_target_size[0], cropped_target_size[1])
    fixed_image_spacing_at_current_level = fixed_image_spacing * 2.0 ** downsample_level
    image_diag: float = (fixed_image_spacing_at_current_level.flip(dims=(0,)) *  #
                         torch.tensor(cropped_target_size, dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid_unshifted = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d_grid = sinogram2d_grid_unshifted.shifted(-fixed_image_offset)

    return {"sinogram2d_grid_unshifted": sinogram2d_grid_unshifted, "sinogram2d_grid": sinogram2d_grid}


@dag_updater(names_returned=["mask", "fixed_image"])
def refresh_mask_transformation_dependent(ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
                                          cropped_target: torch.Tensor, mask_transformation: Transformation | None,
                                          fixed_image_spacing: torch.Tensor, fixed_image_offset: torch.Tensor,
                                          image_2d_downsample_level: int, source_distance: float, device) -> dict[
    str, Any]:
    if mask_transformation is None:
        mask = torch.ones_like(cropped_target)
        fixed_image = cropped_target
    else:
        fixed_image_spacing_at_current_level = fixed_image_spacing * 2.0 ** image_2d_downsample_level
        mask = reg23.project_drr_cuboid_mask(  #
            volume_size=torch.tensor(ct_volumes[0].size(), device=device).flip(dims=(0,)),  #
            voxel_spacing=ct_spacing.to(device=device),  #
            homography_matrix_inverse=mask_transformation.inverse().get_h().to(device=device),  #
            source_distance=source_distance,  #
            output_width=cropped_target.size()[1],  #
            output_height=cropped_target.size()[0],  #
            output_offset=fixed_image_offset.to(device=device, dtype=torch.float64),  #
            detector_spacing=fixed_image_spacing_at_current_level.to(device=device)  #
        )
        fixed_image = mask * cropped_target

    return {"mask": mask, "fixed_image": fixed_image}


@dag_updater(names_returned=["sinogram2d"])
def refresh_mask_transformation_dependent_grangeat(fixed_image: torch.Tensor, source_distance: float,
                                                   fixed_image_spacing: torch.Tensor, image_2d_downsample_level: int,
                                                   sinogram2d_grid_unshifted: Sinogram2dGrid) -> dict[str, Any]:
    fixed_image_spacing_at_current_level = fixed_image_spacing * 2.0 ** image_2d_downsample_level
    sinogram2d = grangeat.calculate_fixed_image(  #
        fixed_image,  #
        source_distance=source_distance, detector_spacing=fixed_image_spacing_at_current_level,
        output_grid=sinogram2d_grid_unshifted)

    return {"sinogram2d": sinogram2d}
