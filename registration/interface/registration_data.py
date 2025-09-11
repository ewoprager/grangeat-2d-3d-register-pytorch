import logging
from typing import Type, NamedTuple, Callable
import math
import copy

logger = logging.getLogger(__name__)

import torch
import pathlib

from registration.lib.structs import Transformation, SceneGeometry, Sinogram2dGrid, Sinogram2dRange, LinearRange
from registration.interface.lib.structs import HyperParameters, Target
from registration.lib.sinogram import Sinogram, SinogramType
from registration import data, drr, pre_computed
from registration.lib import grangeat

import Extension as reg23


class RegistrationData:
    """
    @brief Class to manage the registration hyperparameters, and the data that are modified according to them.
    """

    class CTPathDependent(NamedTuple):
        """
        @brief Struct of data that is dependent only on the CT path used
        """
        ct_volumes: list[torch.Tensor]  # One for each downsample level
        ct_spacing: torch.Tensor  # At the original resolution (no downsampling)

    class CTPathDependentGrangeat(NamedTuple):
        """
        @brief Struct of Grangeat-related data that is dependent only on the CT path used
        """
        sinogram_size: int  # At the original resolution (no downsampling)
        ct_sinograms: dict[Type[SinogramType], list[
            Sinogram]]  # Map of sinogram type to list of sinograms, one for each downsample level

    class TargetDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path and fixed image used
        """
        source_distance: float
        images_2d_full: list[torch.Tensor]  # One for each downsample level
        fixed_image_spacing: torch.Tensor  # At the original resolution (no downsampling)
        transformation_gt: Transformation | None  # For DRR it is GT, for X-ray it is best known alignment; None indicates unknown.

    class HyperparameterDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path, fixed image and hyperparameters used
        """
        cropped_target: torch.Tensor  # The target image with the cropping applied, but no mask applied
        fixed_image_offset: torch.Tensor
        translation_offset: torch.Tensor

    class HyperParameterDependentGrangeat(NamedTuple):
        """
        @brief Struct of Grangeat-related data that is dependent on the CT path, fixed image and hyperparameters used
        """
        sinogram2d_grid: Sinogram2dGrid
        sinogram2d_grid_unshifted: Sinogram2dGrid

    class MaskTransformationDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path, fixed image, hyperparameters used and the transformation
        at which the mask was generated
        """
        fixed_image: torch.Tensor  # The target image with the cropping and masking applied

    class MaskTransformationDependentGrangeat(NamedTuple):
        """
        @brief Struct of Grangeat-related data that is dependent on the CT path, fixed image, hyperparameters used and
        the transformation at which the mask was generated
        """
        sinogram2d: torch.Tensor

    def __init__(self, *, cache_directory: str, ct_path: str | None, target: Target, load_cached: bool,
                 sinogram_types: list[Type[SinogramType]], sinogram_size: int | None, regenerate_drr: bool,
                 save_to_cache: bool, new_drr_size: torch.Size | None, device,
                 target_change_callback: Callable[[], None] | None = None,
                 hyperparameter_change_callback: Callable[[], None] | None = None,
                 hyperparameter_change_callback_grangeat: Callable[[], None] | None = None,
                 mask_transformation_change_callback: Callable[[], None] | None = None,
                 mask_transformation_change_callback_grangeat: Callable[[], None] | None = None):
        self._cache_directory = cache_directory
        self._load_cached = load_cached
        self._sinogram_types = sinogram_types
        self._sinogram_size = sinogram_size
        self._regenerate_drr = regenerate_drr
        self._save_to_cache = save_to_cache
        self._new_drr_size = new_drr_size
        self._device = device

        self._target_change_callback = target_change_callback
        self._hyperparameter_change_callback = hyperparameter_change_callback
        self._hyperparameter_change_callback_grangeat = hyperparameter_change_callback_grangeat
        self._mask_transformation_change_callback = mask_transformation_change_callback
        self._mask_transformation_change_callback_grangeat = mask_transformation_change_callback_grangeat

        self._ct_path = ct_path
        self._target = target
        self._mask_transformation: Transformation | None = None
        self._ct_path_dirty: bool = True
        self._ct_path_dirty_grangeat: bool = True
        self._target_dirty: bool = True
        self._hyperparameters_dirty: bool = True
        self._hyperparameters_dirty_grangeat: bool = True
        self._mask_transformation_dirty: bool = True
        self._mask_transformation_dirty_grangeat: bool = True

        self._suppress_callbacks = True
        self.refresh_ct_path_dependent_grangeat()  # this initialises self._hyperparameters via self.refresh_target_dependent
        self._suppress_callbacks = False

    @property
    def device(self):
        return self._device

    @property
    def suppress_callbacks(self) -> bool:
        return self._suppress_callbacks

    @suppress_callbacks.setter
    def suppress_callbacks(self, new_value: bool) -> None:
        self._suppress_callbacks = new_value

    # -----
    # CT path and properties that depend on it
    # -----

    @property
    def ct_path(self) -> str:
        return self._ct_path

    @ct_path.setter
    def ct_path(self, new_value: str) -> None:
        if new_value != self._ct_path:
            self._ct_path_dirty = True
            self._ct_path_dirty_grangeat = True
            self._target_dirty = True
            self._hyperparameters_dirty = True
            self._hyperparameters_dirty_grangeat = True
            self._mask_transformation_dirty = True
            self._mask_transformation_dirty_grangeat = True
        self._ct_path = new_value

    @property
    def ct_volumes(self) -> list[torch.Tensor]:
        if self._ct_path_dirty:
            self.refresh_ct_path_dependent()
        return self._ct_path_dependent.ct_volumes

    @property
    def ct_spacing_original(self) -> torch.Tensor:
        if self._ct_path_dirty:
            self.refresh_ct_path_dependent()
        return self._ct_path_dependent.ct_spacing

    @property
    def ct_sinograms(self) -> dict[Type[SinogramType], list[Sinogram]]:
        if self._ct_path_dirty_grangeat:
            self.refresh_ct_path_dependent_grangeat()
        return self._ct_path_dependent_grangeat.ct_sinograms

    # -----
    # Target and properties that depend on it, and all those above
    # -----

    @property
    def target(self) -> Target:
        return self._target

    @target.setter
    def target(self, new_value: Target) -> None:
        if new_value != self._target:
            self._target_dirty = True
            self._hyperparameters_dirty = True
            self._hyperparameters_dirty_grangeat = True
            self._mask_transformation_dirty = True
            self._mask_transformation_dirty_grangeat = True
        self._target = new_value

    @property
    def source_distance(self) -> float:
        if self._target_dirty:
            self.refresh_target_dependent()
        return self._target_dependent.source_distance

    @property
    def images_2d_full(self) -> list[torch.Tensor]:
        if self._target_dirty:
            self.refresh_target_dependent()
        return self._target_dependent.images_2d_full

    @property
    def fixed_image_spacing_original(self) -> torch.Tensor:
        if self._target_dirty:
            self.refresh_target_dependent()
        return self._target_dependent.fixed_image_spacing

    @property
    def transformation_gt(self) -> Transformation | None:
        if self._target_dirty:
            self.refresh_target_dependent()
        return self._target_dependent.transformation_gt

    @property
    def ct_volume_at_current_level(self) -> torch.Tensor:
        if self._target_dirty:
            self.refresh_target_dependent()
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._ct_path_dependent.ct_volumes[self.hyperparameters.downsample_level]

    @property
    def ct_spacing_at_current_level(self) -> torch.Tensor:
        if self._target_dirty:
            self.refresh_target_dependent()
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._ct_path_dependent.ct_spacing * 2.0 ** self.hyperparameters.downsample_level

    @property
    def image_2d_full_at_current_level(self) -> torch.Tensor:
        if self._target_dirty:
            self.refresh_target_dependent()
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._target_dependent.images_2d_full[self.hyperparameters.downsample_level]

    @property
    def fixed_image_spacing_at_current_level(self) -> torch.Tensor:
        if self._target_dirty:
            self.refresh_target_dependent()
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._target_dependent.fixed_image_spacing * 2.0 ** self.hyperparameters.downsample_level

    # -----
    # Hyperparameters and properties that depend on it, and all those above
    # -----

    @property
    def hyperparameters(self) -> HyperParameters:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_value: HyperParameters) -> None:
        self._hyperparameters_dirty = True
        self._hyperparameters_dirty_grangeat = True
        self._mask_transformation_dirty = True
        self._mask_transformation_dirty_grangeat = True
        self._hyperparameters = new_value

    @property
    def cropped_target(self) -> torch.Tensor:
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._hyperparameter_dependent.cropped_target

    @property
    def fixed_image_offset(self) -> torch.Tensor:
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._hyperparameter_dependent.fixed_image_offset

    @property
    def translation_offset(self) -> torch.Tensor:
        if self._hyperparameters_dirty:
            self.refresh_hyperparameter_dependent()
        return self._hyperparameter_dependent.translation_offset

    @property
    def sinogram2d_grid(self) -> Sinogram2dGrid:
        if self._hyperparameters_dirty_grangeat:
            self.refresh_hyperparameter_dependent_grangeat()
        return self._hyperparameter_dependent_grangeat.sinogram2d_grid

    @property
    def sinogram2d_grid_unshifted(self) -> Sinogram2dGrid:
        if self._hyperparameters_dirty_grangeat:
            self.refresh_hyperparameter_dependent_grangeat()
        return self._hyperparameter_dependent_grangeat.sinogram2d_grid_unshifted

    # -----
    # Mask transformation and properties that depend on it, and all those above
    # -----

    @property
    def mask_transformation(self) -> Transformation | None:
        return self._mask_transformation

    @mask_transformation.setter
    def mask_transformation(self, new_value: Transformation | None) -> None:
        wasnt = self._mask_transformation is None
        isnt = new_value is None
        if wasnt != isnt or not isnt:
            self._mask_transformation_dirty = True
            self._mask_transformation_dirty_grangeat = True
        self._mask_transformation = new_value

    @property
    def fixed_image(self) -> torch.Tensor:
        if self._mask_transformation_dirty:
            self.refresh_mask_transformation_dependent()
        return self._mask_transformation_dependent.fixed_image

    @property
    def sinogram2d(self) -> torch.Tensor:
        if self._mask_transformation_dirty_grangeat:
            self.refresh_mask_transformation_dependent_grangeat()
        return self._mask_transformation_dependent_grangeat.sinogram2d

    # -----
    # Data refresh methods
    # -----

    def refresh_ct_path_dependent(self) -> None:
        ct_volumes, ct_spacing = data.load_volume(pathlib.Path(self.ct_path), downsample_factor="mipmap")
        ct_volumes = [ct_volume.to(device=self.device, dtype=torch.float32) for ct_volume in ct_volumes]
        ct_spacing = ct_spacing.to(device=self.device)

        self._ct_path_dependent = RegistrationData.CTPathDependent(ct_volumes=ct_volumes, ct_spacing=ct_spacing)
        self._ct_path_dirty = False

        self.refresh_target_dependent()

    def refresh_ct_path_dependent_grangeat(self) -> None:
        this_sinogram_size = int(math.ceil(
            pow(self.ct_volumes[0].numel(), 1.0 / 3.0))) if self._sinogram_size is None else self._sinogram_size

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
                                                             volume_downsample_factor=downsample_factor,
                                                             save_to_cache=True,
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
                    raise RuntimeError("Failed to create sinogram at level {} of type {}; not enough memory?"
                                       "".format(i, tp.__name__))

        self._ct_path_dependent_grangeat = RegistrationData.CTPathDependentGrangeat(sinogram_size=this_sinogram_size,
                                                                                    ct_sinograms=sinogram3ds)
        self._ct_path_dirty_grangeat = False

        self.refresh_hyperparameter_dependent_grangeat()

    def refresh_target_dependent(self) -> None:
        transformation_ground_truth = None
        if self.target.xray_path is None:
            # Load / generate a DRR through the volume
            drr_spec = None
            if not self._regenerate_drr and self.ct_path is not None:
                drr_spec = data.load_cached_drr(self._cache_directory, self.ct_path)

            if drr_spec is None:
                drr_spec = drr.generate_drr_as_target(self._cache_directory, self.ct_path, self.ct_volumes[0],
                                                      self.ct_spacing_original, save_to_cache=self._save_to_cache,
                                                      size=self._new_drr_size)

            fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
            del drr_spec
        else:
            # Load the given X-ray
            image_2d_full, fixed_image_spacing, scene_geometry = data.read_dicom(  #
                self.target.xray_path, downsample_to_ct_spacing=self.ct_spacing_original.mean().item())
            image_2d_full = image_2d_full.to(device=self.device)

        if self.target.flipped:
            logger.info("Flipping target image horizontally.")
            image_2d_full = image_2d_full.flip(dims=(1,))

        # Generating X-ray mipmap
        down_sampler = torch.nn.AvgPool2d(2)
        images_2d_full = [image_2d_full]
        while min(images_2d_full[-1].size()) > 1:
            images_2d_full.append(down_sampler(images_2d_full[-1].unsqueeze(0))[0])

        self._target_dependent = RegistrationData.TargetDependent(source_distance=scene_geometry.source_distance,
                                                                  images_2d_full=images_2d_full,
                                                                  fixed_image_spacing=fixed_image_spacing,
                                                                  transformation_gt=transformation_ground_truth)
        self._target_dirty = False

        self.hyperparameters = HyperParameters.zero(self.images_2d_full[0].size())

        if not self.suppress_callbacks and self._target_change_callback is not None:
            self._target_change_callback()

        self.refresh_hyperparameter_dependent()

    def refresh_hyperparameter_dependent(self) -> None:
        # Cropping for the fixed image
        cropped_target = self.hyperparameters.downsampled_crop(
            self.images_2d_full[self.hyperparameters.downsample_level].size()).apply(
            self.images_2d_full[self.hyperparameters.downsample_level])

        # The fixed image is offset to adjust for the cropping, and according to the source offset
        # This isn't affected by downsample level
        fixed_image_offset = (self.fixed_image_spacing_original * self.hyperparameters.cropping.get_centre_offset(
            self.images_2d_full[0].size()) - self.hyperparameters.source_offset)

        # The translation offset prevents the source offset parameters from fighting the translation parameters in
        # the optimisation
        translation_offset = -self.hyperparameters.source_offset

        self._hyperparameter_dependent = RegistrationData.HyperparameterDependent(cropped_target=cropped_target,
                                                                                  fixed_image_offset=fixed_image_offset,
                                                                                  translation_offset=translation_offset)
        self._hyperparameters_dirty = False

        if not self.suppress_callbacks and self._hyperparameter_change_callback is not None:
            self._hyperparameter_change_callback()

        self.refresh_mask_transformation_dependent()

    def refresh_hyperparameter_dependent_grangeat(self) -> None:
        cropped_target_size = self.cropped_target.size()
        sinogram2d_counts = max(cropped_target_size[0], cropped_target_size[1])
        image_diag: float = (self.fixed_image_spacing_at_current_level.flip(dims=(0,)) *  #
                             torch.tensor(cropped_target_size, dtype=torch.float32)).square().sum().sqrt().item()
        sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                           LinearRange(-.5 * image_diag, .5 * image_diag))
        sinogram2d_grid_unshifted = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts,
                                                                     device=self.device)

        sinogram2d_grid = sinogram2d_grid_unshifted.shifted(-self.fixed_image_offset)

        self._hyperparameter_dependent_grangeat = RegistrationData.HyperParameterDependentGrangeat(
            sinogram2d_grid_unshifted=sinogram2d_grid_unshifted, sinogram2d_grid=sinogram2d_grid)
        self._hyperparameters_dirty_grangeat = False

        if not self.suppress_callbacks and self._hyperparameter_change_callback_grangeat is not None:
            self._hyperparameter_change_callback_grangeat()

        self.refresh_mask_transformation_dependent_grangeat()

    def refresh_mask_transformation_dependent(self) -> None:
        if self.mask_transformation is None:
            fixed_image = self.cropped_target
        else:
            mask = reg23.project_drr_cuboid_mask(  #
                volume_size=torch.tensor(self.ct_volumes[0].size(), device=self.device).flip(dims=(0,)),  #
                voxel_spacing=self.ct_spacing_original.to(device=self.device),  #
                homography_matrix_inverse=self.mask_transformation.inverse().get_h().to(device=self.device),  #
                source_distance=self.source_distance, output_width=self.cropped_target.size()[1],  #
                output_height=self.cropped_target.size()[0],  #
                output_offset=self.fixed_image_offset.to(device=self.device, dtype=torch.float64),  #
                detector_spacing=self.fixed_image_spacing_at_current_level.to(device=self.device)  #
            )
            fixed_image = mask * self.cropped_target
            del mask

        self._mask_transformation_dependent = RegistrationData.MaskTransformationDependent(fixed_image=fixed_image)
        self._mask_transformation_dirty = False

        if not self.suppress_callbacks and self._mask_transformation_change_callback is not None:
            self._mask_transformation_change_callback()

    def refresh_mask_transformation_dependent_grangeat(self) -> None:
        sinogram2d = grangeat.calculate_fixed_image(  #
            self.fixed_image,  #
            source_distance=self.source_distance, detector_spacing=self.fixed_image_spacing_at_current_level,
            output_grid=self.sinogram2d_grid_unshifted)

        self._mask_transformation_dependent_grangeat = RegistrationData.MaskTransformationDependentGrangeat(
            sinogram2d=sinogram2d)
        self._mask_transformation_dirty_grangeat = False

        if not self.suppress_callbacks and self._mask_transformation_change_callback_grangeat is not None:
            self._mask_transformation_change_callback_grangeat()
