import os
import argparse
import logging.config
import pathlib
import time
from typing import NamedTuple, Any
from datetime import datetime

os.environ["QT_API"] = "PyQt6"

import numpy as np
import torch
import napari
import scipy
from magicgui import magicgui, widgets
from PyQt6.QtWidgets import QDockWidget

from registration.lib.sinogram import *
from registration import drr
from registration import data
from registration import script
from registration.lib import geometry
from registration.interface.lib.structs import *
from registration.interface.transformations import TransformationWidget
from registration.interface.view import ViewWidget
from registration.interface.register import RegisterWidget
from registration.lib.geometry import generate_drr
from registration.objective_function import zncc


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
        else:
            # Load the given X-ray
            self._fixed_image, self._fixed_image_spacing, self._scene_geometry = data.read_dicom(x_ray,
                                                                                                 downsample_factor=4)
            self._fixed_image = self._fixed_image.to(device=self.device)
            f_middle = 0.3
            # _fixed_image = _fixed_image[int(float(_fixed_image.size()[0]) * .5 * (1. - f_middle)):int(
            #     float(_fixed_image.size()[0]) * .5 * (1. + f_middle)), :]
            self._fixed_image = self._fixed_image[int(float(self._fixed_image.size()[0]) * .5 * (1. - f_middle)):int(
                float(self._fixed_image.size()[0]) * .5 * (1. + f_middle)),
                                int(float(self._fixed_image.size()[1]) * .5 * (1. - 0.7)):int(
                                    float(self._fixed_image.size()[1]) * .5 * (1. + 0.7))]

            logger.info("Calculating 2D sinogram (the fixed image)...")

            sinogram2d_counts = max(self.fixed_image.size()[0], self.fixed_image.size()[1])
            image_diag: float = (self.fixed_image_spacing * torch.tensor(self.fixed_image.size(),
                                                                         dtype=torch.float32)).square().sum().sqrt(

            ).item()
            sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                               LinearRange(-.5 * image_diag, .5 * image_diag))
            sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=self.device)

            self._sinogram2d = grangeat.calculate_fixed_image(self.fixed_image,
                                                              source_distance=self.scene_geometry.source_distance,
                                                              detector_spacing=self.fixed_image_spacing,
                                                              output_grid=sinogram2d_grid)

            del sinogram2d_grid
            logger.info("X-ray sinogram calculated.")

        self._sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, self.sinogram2d.size(),
                                                                 device=self.device)

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

    def objective_function_drr(self, transformation: Transformation) -> torch.Tensor:
        moving_image = generate_drr(self.ct_volume, transformation=transformation, voxel_spacing=self.ct_spacing,
                                    detector_spacing=self.fixed_image_spacing, scene_geometry=self.scene_geometry,
                                    output_size=self.fixed_image.size())
        return -zncc(self.fixed_image, moving_image)

    def objective_function_grangeat(self, transformation: Transformation) -> torch.Tensor:
        source_position = self.scene_geometry.source_position(device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        resampled = self.sinogram3d.resample(ph_matrix, self.sinogram2d_grid)
        return -zncc(self.sinogram2d, resampled)


class Interface:
    def __init__(self, registration_data: RegistrationData):
        self._registration_data = registration_data

        self._viewer = napari.Viewer()
        self._viewer.bind_key("Alt", self._on_alt_down)

        self._fixed_image_layer = self._viewer.add_image(self.registration_data.fixed_image.cpu().numpy(),
                                                         colormap="yellow", interpolation2d="linear",
                                                         name="Fixed image")
        self._moving_image_layer = self._viewer.add_image(np.zeros(self.registration_data.fixed_image.size()),
                                                          colormap="blue", blending="additive",
                                                          interpolation2d="linear", name="DRR")
        self._moving_image_layer.bind_key('r', self._reset)
        self._moving_image_layer.mouse_drag_callbacks.append(self._mouse_drag)

        self._sinogram2d_layer = self._viewer.add_image(self.registration_data.fixed_image.cpu().numpy(),
                                                        colormap="yellow", interpolation2d="linear", translate=(
                self.registration_data.fixed_image.size()[0] + 24, 0), name="Fixed sinogram")

        self._view_params = ViewParams(translation_sensitivity=0.06, rotation_sensitivity=0.002)

        self._key_states = {"Alt": False}

        self._view_widget = ViewWidget(self.set_view_params)
        self._viewer.window.add_dock_widget(self._view_widget, name="View options", area="right",
                                            menu=self._viewer.window.window_menu)

        initial_transformation = self.registration_data.transformation_ground_truth
        if initial_transformation is None:
            initial_transformation = Transformation.random(device=self.registration_data.device)
        self._transformation_widget = TransformationWidget(initial_transformation=initial_transformation,
                                                           refresh_render_function=self.render_drr,
                                                           save_path=pathlib.Path("cache/saved_transformations.pkl"))
        self._viewer.window.add_dock_widget(self._transformation_widget, name="Transformations", area="right",
                                            menu=self._viewer.window.window_menu)

        self._register_widget = RegisterWidget(transformation_widget=self._transformation_widget,
                                               objective_function=self.registration_data.objective_function_grangeat)
        self._viewer.window.add_dock_widget(self._register_widget, name="Register", area="right",
                                            menu=self._viewer.window.window_menu)

        self.render_drr(self._transformation_widget.get_current_transformation())

    @property
    def registration_data(self) -> RegistrationData:
        return self._registration_data

    def get_view_params(self) -> ViewParams:
        return self._view_params

    def set_view_params(self, value: ViewParams) -> None:
        self._view_params = value

    view_params = property(get_view_params, set_view_params)

    def render_drr(self, transformation: Transformation):
        moved_drr = geometry.generate_drr(self.registration_data.ct_volume, transformation=transformation,
                                          voxel_spacing=self.registration_data.ct_spacing,
                                          detector_spacing=self.registration_data.fixed_image_spacing,
                                          scene_geometry=self.registration_data.scene_geometry,
                                          output_size=self.registration_data.fixed_image.size())
        self._moving_image_layer.data = moved_drr.cpu().numpy()

    # Event callbacks:
    def _on_alt_down(self, viewer):
        self._key_states["Alt"] = True
        yield
        self._key_states["Alt"] = False

    def _reset(self, layer):
        reset_transformation = self.registration_data.transformation_ground_truth
        if reset_transformation is None:
            reset_transformation = Transformation.random(device=self.registration_data.device)
        self._transformation_widget.set_current_transformation(reset_transformation)
        self.render_drr(self._transformation_widget.get_current_transformation())
        logger.info("Reset")

    def _mouse_drag(self, layer, event):
        if event.button == 1 and self._key_states["Alt"]:  # Alt-left click drag
            # mouse down
            dragged = False
            drag_start = np.array([event.position[1], -event.position[0]])
            rotation_start = scipy.spatial.transform.Rotation.from_rotvec(
                rotvec=self._transformation_widget.get_current_transformation().rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._view_params.rotation_sensitivity * (
                        np.array([event.position[1], -event.position[0]]) - drag_start)
                euler_angles = [delta[1], delta[0], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=euler_angles)
                rot_combined = rot_euler * rotation_start
                self._transformation_widget.set_current_transformation(Transformation(
                    rotation=torch.tensor(rot_combined.as_rotvec(),
                                          device=self._transformation_widget.get_current_transformation(

                                          ).rotation.device,
                                          dtype=self._transformation_widget.get_current_transformation(

                                          ).rotation.dtype),
                    translation=self._transformation_widget.get_current_transformation().translation))
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass
        elif event.button == 2 and self._key_states["Alt"]:  # Alt-right click drag
            # mouse down
            dragged = False
            drag_start = torch.tensor(event.position)
            # rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            translation_start = self._transformation_widget.get_current_transformation().translation[0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._view_params.translation_sensitivity * (torch.tensor(event.position) - drag_start).flip(
                    (0,))
                tr = self._transformation_widget.get_current_transformation().translation
                tr[0:2] = (translation_start + delta).to(device=tr.device)
                self._transformation_widget.set_current_transformation(Transformation(translation=tr,
                                                                                      rotation=self._transformation_widget.get_current_transformation().rotation))
                self.render_drr(self._transformation_widget.get_current_transformation())
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass


def main(*, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool, save_to_cache: bool,
         sinogram_size: int, x_ray: str | None = None) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    registration_data = RegistrationData(path, cache_directory, load_cached, regenerate_drr, save_to_cache,
                                         sinogram_size, x_ray, device)

    interface = Interface(registration_data)

    napari.run()

    return 0


if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-nrrd-path", type=str,
                        help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
                             "synthetic data will be used instead - note that in this case, data will not be saved to "
                             "the cache.")
    parser.add_argument("-i", "--no-load", action='store_true',
                        help="Do not load any pre-calculated data from the cache.")
    parser.add_argument("-r", "--regenerate-drr", action='store_true',
                        help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-s", "--sinogram-size", type=int, default=256,
                        help="The number of values of r, theta and phi to calculate plane integrals for, "
                             "and the width and height of the grid of samples taken to approximate each integral. The "
                             "computational expense of the 3D Radon transform is O(sinogram_size^5).")
    parser.add_argument("-x", "--x-ray", type=str,
                        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
                             "this is provided, the X-ray will by used instead of any DRR.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    ret = main(path=args.ct_nrrd_path, cache_directory=args.cache_directory, load_cached=not args.no_load,
               regenerate_drr=args.regenerate_drr, save_to_cache=not args.no_save, sinogram_size=args.sinogram_size,
               x_ray=args.x_ray if "x_ray" in vars(args) else None)

    exit(ret)
