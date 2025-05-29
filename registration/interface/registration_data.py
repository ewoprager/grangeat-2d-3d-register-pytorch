import logging
from typing import Callable

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets

from registration.lib.structs import *
from registration.interface.lib.structs import *
from registration.lib.sinogram import *
from registration import script, data, drr, objective_function
from registration.lib import grangeat
from registration.interface.registration_constants import RegistrationConstants


class RegistrationData:
    """
    @brief Class to manage the registration hyperparameters, and the data that are modified according to them.
    """

    class Cached(NamedTuple):
        at_parameters: HyperParameters
        fixed_image: torch.Tensor
        fixed_image_offset: torch.Tensor
        translation_offset: torch.Tensor
        sinogram2d: torch.Tensor
        sinogram2d_grid: Sinogram2dGrid

    def __init__(self, *, registration_constants: RegistrationConstants,
                 image_change_callback: Callable[[], None] | None):
        self._registration_constants = registration_constants
        self._image_change_callback = image_change_callback

        self._hyperparameters = HyperParameters.zero(self._registration_constants.image_2d_full.size())

        self._supress_callbacks: bool = True
        self._cached = None
        self._re_calculate_cache()
        self.supress_callbacks = False

    @property
    def device(self):
        return self._registration_constants.device

    @property
    def suppress_callbacks(self) -> bool:
        return self._supress_callbacks

    @suppress_callbacks.setter
    def suppress_callbacks(self, new_value: bool) -> None:
        self._supress_callbacks = new_value

    @property
    def hyperparameters(self) -> HyperParameters:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_value: HyperParameters) -> None:
        self._hyperparameters = new_value

    @property
    def fixed_image(self) -> torch.Tensor:
        if not self._cached.at_parameters.is_close(self.hyperparameters):
            self._re_calculate_cache()
        return self._cached.fixed_image

    @property
    def fixed_image_offset(self) -> torch.Tensor:
        if not self._cached.at_parameters.is_close(self.hyperparameters):
            self._re_calculate_cache()
        return self._cached.fixed_image_offset

    @property
    def translation_offset(self) -> torch.Tensor:
        if not self._cached.at_parameters.is_close(self.hyperparameters):
            self._re_calculate_cache()
        return self._cached.translation_offset

    @property
    def sinogram2d(self) -> torch.Tensor:
        if not self._cached.at_parameters.is_close(self.hyperparameters):
            self._re_calculate_cache()
        return self._cached.sinogram2d

    @property
    def sinogram2d_grid(self) -> Sinogram2dGrid:
        if not self._cached.at_parameters.is_close(self.hyperparameters):
            self._re_calculate_cache()
        return self._cached.sinogram2d_grid

    def _re_calculate_cache(self) -> None:
        # Cropping for the fixed image
        fixed_image = self.hyperparameters.cropping.apply(self._registration_constants.image_2d_full)

        # The fixed image is offset to adjust for the cropping, and according to the source offset
        fixed_image_offset = (
                self._registration_constants.fixed_image_spacing * self.hyperparameters.cropping.get_centre_offset(
            self._registration_constants.image_2d_full.size()) - self.hyperparameters.source_offset)

        # The translation offset prevents the source offset parameters from fighting the translation parameters in
        # the optimisation
        translation_offset = -self.hyperparameters.source_offset

        sinogram2d_counts = max(fixed_image.size()[0], fixed_image.size()[1])
        image_diag: float = (self._registration_constants.fixed_image_spacing.flip(dims=(0,)) * torch.tensor(
            fixed_image.size(), dtype=torch.float32)).square().sum().sqrt().item()
        sinogram2d_range = Sinogram2dRange(
            LinearRange(-.5 * torch.pi, .5 * torch.pi), LinearRange(-.5 * image_diag, .5 * image_diag))
        sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=self.device)

        sinogram2d = grangeat.calculate_fixed_image(
            fixed_image, source_distance=self._registration_constants.scene_geometry.source_distance,
            detector_spacing=self._registration_constants.fixed_image_spacing, output_grid=sinogram2d_grid)

        sinogram2d_grid = sinogram2d_grid.shifted(-fixed_image_offset)

        self._cached = RegistrationData.Cached(
            at_parameters=self.hyperparameters, fixed_image=fixed_image, fixed_image_offset=fixed_image_offset,
            translation_offset=translation_offset, sinogram2d=sinogram2d, sinogram2d_grid=sinogram2d_grid)
        if not self.suppress_callbacks and self._image_change_callback is not None:
            self._image_change_callback()
