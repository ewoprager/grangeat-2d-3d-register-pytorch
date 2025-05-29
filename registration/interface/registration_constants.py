import logging
from typing import Callable

from registration.interface.lib.structs import HyperParameters

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration import script, data, drr, objective_function
from registration.lib import grangeat


class RegistrationConstants:
    def __init__(self, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool,
                 save_to_cache: bool, sinogram_size: int, x_ray: str | None, device,
                 new_drr_size: torch.Size = torch.Size([1000, 1000])):
        self._ct_volume, self._ct_spacing, self._sinogram3d = script.get_volume_and_sinogram(path, cache_directory,
                                                                                             load_cached=load_cached,
                                                                                             save_to_cache=save_to_cache,
                                                                                             sinogram_size=sinogram_size,
                                                                                             device=device)
        self._device = device
        self._transformation_ground_truth = None

        if x_ray is None:
            # Load / generate a DRR through the volume
            drr_spec = None
            if not regenerate_drr and path is not None:
                drr_spec = data.load_cached_drr(cache_directory, path)

            if drr_spec is None:
                drr_spec = drr.generate_drr_as_target(cache_directory, path, self.ct_volume, self.ct_spacing,
                                                      device=self.device, save_to_cache=save_to_cache,
                                                      size=new_drr_size)

            (self._fixed_image_spacing, self._scene_geometry, self._image_2d_full, self._sinogram2d, sinogram2d_range,
             self._transformation_ground_truth) = drr_spec
            del drr_spec

            self._fixed_image = self._image_2d_full
            self._sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, self.sinogram2d.size(),
                                                                     device=self.device)
        else:
            # Load the given X-ray
            self._image_2d_full, self._fixed_image_spacing, self._scene_geometry = data.read_dicom(x_ray,
                                                                                                   downsample_factor=4)
            # Flip horizontally
            self._image_2d_full = self._image_2d_full.flip(dims=(1,)).to(device=self.device)

    @property
    def device(self):
        return self._device

    @property
    def scene_geometry(self) -> SceneGeometry:
        return self._scene_geometry

    @property
    def ct_volume(self) -> torch.Tensor:
        return self._ct_volume

    @property
    def ct_spacing(self) -> torch.Tensor:
        return self._ct_spacing

    @property
    def sinogram3d(self) -> Sinogram:
        return self._sinogram3d

    @property
    def image_2d_full(self) -> torch.Tensor:
        return self._image_2d_full

    @property
    def fixed_image_spacing(self) -> torch.Tensor:
        return self._fixed_image_spacing

    @property
    def transformation_ground_truth(self) -> Transformation | None:
        return self._transformation_ground_truth
