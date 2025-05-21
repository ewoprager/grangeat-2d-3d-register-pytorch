import logging
from typing import Callable

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration import script, data, drr, objective_function
from registration.lib import grangeat


class RegistrationData:
    def __init__(self, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool,
                 save_to_cache: bool, sinogram_size: int, x_ray: str | None, device):
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
                drr_spec = drr.generate_new_drr(cache_directory, path, self.ct_volume, self.ct_spacing,
                                                device=self.device, save_to_cache=save_to_cache)

            (self._fixed_image_spacing, self._scene_geometry, self._fixed_image, self._sinogram2d, sinogram2d_range,
             self._transformation_ground_truth) = drr_spec
            del drr_spec

            self._sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, self.sinogram2d.size(),
                                                                     device=self.device)
        else:
            # Load the given X-ray
            self._fixed_image, self._fixed_image_spacing, self._scene_geometry = data.read_dicom(x_ray,
                                                                                                 downsample_factor=4)
            self._fixed_image = self._fixed_image.to(device=self.device)
            f_middle = 0.4
            # _fixed_image = _fixed_image[int(float(_fixed_image.size()[0]) * .5 * (1. - f_middle)):int(
            #     float(_fixed_image.size()[0]) * .5 * (1. + f_middle)), :]
            self._fixed_image = self._fixed_image[int(float(self._fixed_image.size()[0]) * .5 * (1. - f_middle)):int(
                float(self._fixed_image.size()[0]) * .5 * (1. + f_middle)),
                                int(float(self._fixed_image.size()[1]) * .5 * (1. - 0.7)):int(
                                    float(self._fixed_image.size()[1]) * .5 * (1. + 0.7))]

            logger.info("Calculating 2D sinogram (the fixed image)...")

            sinogram2d_counts = max(self.fixed_image.size()[0], self.fixed_image.size()[1])
            image_diag: float = (self.fixed_image_spacing.flip(dims=(0,)) * torch.tensor(self.fixed_image.size(),
                                                                                         dtype=torch.float32)).square().sum().sqrt().item()
            sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                               LinearRange(-.5 * image_diag, .5 * image_diag))
            self._sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=self.device)

            self._sinogram2d = grangeat.calculate_fixed_image(self.fixed_image,
                                                              source_distance=self.scene_geometry.source_distance,
                                                              detector_spacing=self.fixed_image_spacing,
                                                              output_grid=self._sinogram2d_grid)
            logger.info("X-ray sinogram calculated.")

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
    def fixed_image(self) -> torch.Tensor:
        return self._fixed_image

    @property
    def fixed_image_spacing(self) -> torch.Tensor:
        return self._fixed_image_spacing

    @property
    def sinogram2d(self) -> torch.Tensor:
        return self._sinogram2d

    @property
    def sinogram2d_grid(self) -> Sinogram2dGrid:
        return self._sinogram2d_grid

    @property
    def transformation_ground_truth(self) -> Transformation | None:
        return self._transformation_ground_truth

    def resample_sinogram3d(self, transformation: Transformation) -> torch.Tensor:
        source_position = self.scene_geometry.source_position(device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        return self.sinogram3d.resample(ph_matrix, self.sinogram2d_grid)

    def objective_function_drr(self, transformation: Transformation) -> torch.Tensor:
        moving_image = geometry.generate_drr(self.ct_volume, transformation=transformation.to(device=self.device),
                                             voxel_spacing=self.ct_spacing, detector_spacing=self.fixed_image_spacing,
                                             scene_geometry=self.scene_geometry, output_size=self.fixed_image.size())
        return -objective_function.zncc(self.fixed_image, moving_image)

    def objective_function_grangeat(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self.sinogram2d, self.resample_sinogram3d(transformation))
