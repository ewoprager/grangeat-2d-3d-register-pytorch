import logging
from typing import Any

import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["refresh_hyperparameter_dependent_grangeat", "refresh_mask_transformation_dependent_grangeat"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["sinogram_size", "ct_sinograms"])
def refresh_vif(*self) -> dict[str, Any] | Error:
    this_sinogram_size = int(
        math.ceil(pow(self.ct_volumes[0].numel(), 1.0 / 3.0))) if self._sinogram_size is None else self._sinogram_size

    def get_sinogram(sinogram_type: Type[SinogramType], downsample_level: int) -> Sinogram | None:
        downsample_factor = int(2 ** downsample_level)
        downsampled_sinogram_size = this_sinogram_size // downsample_factor
        sinogram3d = None
        sinogram_hash = deterministic_hash_sinogram(self.ct_path, sinogram_type, downsampled_sinogram_size,
                                                    downsample_factor)
        volume_spec = load_cached_ct(self._cache_directory, sinogram_hash)
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


@dadg_updater(names_returned=["sinogram2d_grid_unshifted", "sinogram2d_grid"])
def refresh_hyperparameter_dependent_grangeat(*, cropped_target: torch.Tensor, fixed_image_offset: torch.Tensor,
                                              fixed_image_spacing: torch.Tensor, downsample_level: int) -> dict[
    str, Any]:
    device = cropped_target.device
    assert fixed_image_offset.device == device
    assert fixed_image_spacing.device == device

    cropped_target_size = cropped_target.size()
    sinogram2d_counts = max(cropped_target_size[0], cropped_target_size[1])
    fixed_image_spacing_at_current_level = fixed_image_spacing * 2.0 ** downsample_level
    image_diag: float = (fixed_image_spacing_at_current_level.flip(dims=(0,)) *  #
                         torch.tensor(cropped_target_size, device=device)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid_unshifted = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d_grid = sinogram2d_grid_unshifted.shifted(-fixed_image_offset)

    return {"sinogram2d_grid_unshifted": sinogram2d_grid_unshifted, "sinogram2d_grid": sinogram2d_grid}


@dadg_updater(names_returned=["sinogram2d"])
def refresh_mask_transformation_dependent_grangeat(*, fixed_image: torch.Tensor, source_distance: float,
                                                   fixed_image_spacing: torch.Tensor, image_2d_scale_factor: float,
                                                   sinogram2d_grid_unshifted: Sinogram2dGrid) -> dict[str, Any]:
    fixed_image_spacing_at_current_level = fixed_image_spacing / image_2d_scale_factor
    sinogram2d = grangeat.calculate_fixed_image(  #
        fixed_image,  #
        source_distance=source_distance, detector_spacing=fixed_image_spacing_at_current_level,
        output_grid=sinogram2d_grid_unshifted)

    return {"sinogram2d": sinogram2d}
