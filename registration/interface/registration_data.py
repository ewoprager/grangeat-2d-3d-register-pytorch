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


class RegistrationData:
    """
    @brief Class to manage the registration hyperparameters, and the data that are modified according to them.
    """

    class CTPathDependent(NamedTuple):
        """
        @brief Struct of data that is dependent only on the CT path used
        """
        from_ct_path: str  # The path of the CT file/directory this data was generated from
        ct_volume: torch.Tensor
        ct_spacing: torch.Tensor
        sinogram_size: int
        ct_sinograms: dict[Type[SinogramType], Sinogram]  # Map of sinogram type to sinogram data

    class TargetDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path and fixed image used
        """
        from_target: Target  # The target this data was generated from
        scene_geometry: SceneGeometry
        image_2d_full: torch.Tensor
        fixed_image_spacing: torch.Tensor
        transformation_gt: Transformation | None  # For DRR it is GT, for X-ray it is best known alignment; None indicates unknown.

    class HyperparameterDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path, fixed image and hyperparameters used
        """
        from_parameters: HyperParameters  # The hyperparameters at which this data was generated
        fixed_image: torch.Tensor
        fixed_image_offset: torch.Tensor
        translation_offset: torch.Tensor
        sinogram2d: torch.Tensor
        sinogram2d_grid: Sinogram2dGrid

    def __init__(self, *, cache_directory: str, ct_path: str | None, target: Target, load_cached: bool,
                 sinogram_types: list[Type[SinogramType]], sinogram_size: int | None, regenerate_drr: bool,
                 save_to_cache: bool, new_drr_size: torch.Size | None, volume_downsample_factor: int,
                 hyperparameter_change_callback: Callable[[], None] | None,
                 target_change_callback: Callable[[], None] | None, device):
        self._cache_directory = cache_directory
        self._load_cached = load_cached
        self._sinogram_types = sinogram_types
        self._sinogram_size = sinogram_size
        self._regenerate_drr = regenerate_drr
        self._save_to_cache = save_to_cache
        self._new_drr_size = new_drr_size
        self._volume_downsample_factor = volume_downsample_factor
        self._device = device

        self._hyperparameter_change_callback = hyperparameter_change_callback
        self._target_change_callback = target_change_callback

        self._ct_path = ct_path
        self._target = target

        self._suppress_callbacks = True
        self.refresh_ct_path_dependent()
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

    @property
    def ct_path(self) -> str:
        return self._ct_path

    @ct_path.setter
    def ct_path(self, new_value: str) -> None:
        self._ct_path = new_value

    @property
    def ct_volume(self) -> torch.Tensor:
        self._check_ct_path_dependent()
        return self._ct_path_dependent.ct_volume

    @property
    def ct_spacing(self) -> torch.Tensor:
        self._check_ct_path_dependent()
        return self._ct_path_dependent.ct_spacing

    @property
    def ct_sinograms(self) -> dict[Type[SinogramType], Sinogram]:
        self._check_ct_path_dependent()
        return self._ct_path_dependent.ct_sinograms

    @property
    def target(self) -> Target:
        return self._target

    @target.setter
    def target(self, new_value: Target) -> None:
        self._target = new_value

    @property
    def scene_geometry(self) -> SceneGeometry:
        self._check_target_dependent()
        return self._target_dependent.scene_geometry

    @property
    def image_2d_full(self) -> torch.Tensor:
        self._check_target_dependent()
        return self._target_dependent.image_2d_full

    @property
    def fixed_image_spacing(self) -> torch.Tensor:
        self._check_target_dependent()
        return self._target_dependent.fixed_image_spacing

    @property
    def transformation_gt(self) -> Transformation | None:
        self._check_target_dependent()
        return self._target_dependent.transformation_gt

    @property
    def hyperparameters(self) -> HyperParameters:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_value: HyperParameters) -> None:
        self._hyperparameters = new_value

    @property
    def fixed_image(self) -> torch.Tensor:
        self._check_hyperparameter_dependent()
        return self._hyperparameter_dependent.fixed_image

    @property
    def fixed_image_offset(self) -> torch.Tensor:
        self._check_hyperparameter_dependent()
        return self._hyperparameter_dependent.fixed_image_offset

    @property
    def translation_offset(self) -> torch.Tensor:
        self._check_hyperparameter_dependent()
        return self._hyperparameter_dependent.translation_offset

    @property
    def sinogram2d(self) -> torch.Tensor:
        self._check_hyperparameter_dependent()
        return self._hyperparameter_dependent.sinogram2d

    @property
    def sinogram2d_grid(self) -> Sinogram2dGrid:
        self._check_hyperparameter_dependent()
        return self._hyperparameter_dependent.sinogram2d_grid

    def refresh_ct_path_dependent(self) -> None:
        ct_volume, ct_spacing = data.load_volume(pathlib.Path(self.ct_path),
                                                 downsample_factor=self._volume_downsample_factor)
        ct_volume = ct_volume.to(device=self.device, dtype=torch.float32)
        ct_spacing = ct_spacing.to(device=self.device)
        this_sinogram_size = int(
            math.ceil(pow(ct_volume.numel(), 1.0 / 3.0))) if self._sinogram_size is None else self._sinogram_size

        def get_sinogram(sinogram_type: Type[SinogramType]) -> Sinogram | None:
            sinogram3d = None
            sinogram_hash = data.deterministic_hash_sinogram(self.ct_path, sinogram_type, this_sinogram_size,
                                                             self._volume_downsample_factor)
            volume_spec = data.load_cached_volume(self._cache_directory, sinogram_hash)
            if volume_spec is not None:
                _, sinogram3d = volume_spec
            if sinogram3d is None:
                res = pre_computed.calculate_volume_sinogram(self._cache_directory, ct_volume, voxel_spacing=ct_spacing,
                                                             ct_volume_path=self.ct_path,
                                                             volume_downsample_factor=self._volume_downsample_factor,
                                                             save_to_cache=True, sinogram_size=this_sinogram_size,
                                                             sinogram_type=sinogram_type)
                if res is None:
                    return None
                sinogram3d, _ = res
            return sinogram3d

        sinogram3ds = {tp: get_sinogram(tp) for tp in self._sinogram_types}

        for s in sinogram3ds:
            if s is None:
                raise RuntimeError("Failed to create sinogram; not enough memory?")

        self._ct_path_dependent = RegistrationData.CTPathDependent(from_ct_path=copy.deepcopy(self.ct_path),
                                                                   ct_volume=ct_volume, ct_spacing=ct_spacing,
                                                                   sinogram_size=this_sinogram_size,
                                                                   ct_sinograms=sinogram3ds)
        self.refresh_target_dependent()

    def refresh_target_dependent(self) -> None:
        transformation_ground_truth = None
        if self.target.xray_path is None:
            # Load / generate a DRR through the volume
            drr_spec = None
            if not self._regenerate_drr and self.ct_path is not None:
                drr_spec = data.load_cached_drr(self._cache_directory, self.ct_path)

            if drr_spec is None:
                drr_spec = drr.generate_drr_as_target(self._cache_directory, self.ct_path, self.ct_volume,
                                                      self.ct_spacing, save_to_cache=self._save_to_cache,
                                                      size=self._new_drr_size)

            fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
            del drr_spec
        else:
            # Load the given X-ray
            image_2d_full, fixed_image_spacing, scene_geometry = data.read_dicom(self.target.xray_path,
                                                                                 downsample_to_ct_spacing=self.ct_spacing.mean().item())
            image_2d_full = image_2d_full.to(device=self.device)

        if self.target.flipped:
            logger.info("Flipping target image horizontally.")
            image_2d_full = image_2d_full.flip(dims=(1,))

        self._target_dependent = RegistrationData.TargetDependent(from_target=copy.deepcopy(self.target),
                                                                  scene_geometry=scene_geometry,
                                                                  image_2d_full=image_2d_full,
                                                                  fixed_image_spacing=fixed_image_spacing,
                                                                  transformation_gt=transformation_ground_truth)
        self.hyperparameters = HyperParameters.zero(self.image_2d_full.size())

        if not self.suppress_callbacks and self._target_change_callback is not None:
            self._target_change_callback()

        self.refresh_hyperparameter_dependent()

    def refresh_hyperparameter_dependent(self) -> None:
        # Cropping for the fixed image
        fixed_image = self.hyperparameters.cropping.apply(self.image_2d_full)

        # The fixed image is offset to adjust for the cropping, and according to the source offset
        fixed_image_offset = (self.fixed_image_spacing * self.hyperparameters.cropping.get_centre_offset(
            self.image_2d_full.size()) - self.hyperparameters.source_offset)

        # The translation offset prevents the source offset parameters from fighting the translation parameters in
        # the optimisation
        translation_offset = -self.hyperparameters.source_offset

        sinogram2d_counts = max(fixed_image.size()[0], fixed_image.size()[1])
        image_diag: float = (self.fixed_image_spacing.flip(dims=(0,)) * torch.tensor(fixed_image.size(),
                                                                                     dtype=torch.float32)).square().sum().sqrt().item()
        sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                           LinearRange(-.5 * image_diag, .5 * image_diag))
        sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=self.device)

        sinogram2d = grangeat.calculate_fixed_image(
            fixed_image * self.hyperparameters.mask.to(device=fixed_image.device),
            source_distance=self.scene_geometry.source_distance, detector_spacing=self.fixed_image_spacing,
            output_grid=sinogram2d_grid)

        sinogram2d_grid = sinogram2d_grid.shifted(-fixed_image_offset)

        self._hyperparameter_dependent = RegistrationData.HyperparameterDependent(
            from_parameters=copy.deepcopy(self.hyperparameters), fixed_image=fixed_image,
            fixed_image_offset=fixed_image_offset, translation_offset=translation_offset, sinogram2d=sinogram2d,
            sinogram2d_grid=sinogram2d_grid)
        if not self.suppress_callbacks and self._hyperparameter_change_callback is not None:
            self._hyperparameter_change_callback()

    def _check_ct_path_dependent(self) -> bool:
        """
        :return: Whether a refresh was performed
        """
        if self._ct_path_dependent.from_ct_path == self.ct_path:
            return False
        else:
            self.refresh_ct_path_dependent()
            return True

    def _check_hyperparameter_dependent(self) -> bool:
        """
        :return: Whether a refresh was performed
        """
        if self._check_target_dependent():
            return True
        else:
            if self._hyperparameter_dependent.from_parameters.is_close(self.hyperparameters):
                return False
            else:
                self.refresh_hyperparameter_dependent()
                return True

    def _check_target_dependent(self) -> bool:
        """
        :return: Whether a refresh was performed
        """
        if self._check_ct_path_dependent():
            return True
        else:
            if self._target_dependent.from_target == self.target:
                return False
            else:
                self.refresh_target_dependent()
                return True
