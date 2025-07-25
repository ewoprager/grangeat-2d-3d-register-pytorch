import os
import argparse
import time
from typing import NamedTuple, Any
from datetime import datetime
import copy

os.environ["QT_API"] = "PyQt6"

import pathlib
import numpy as np
import torch
import napari
import scipy
from magicgui import magicgui, widgets
from PyQt6.QtWidgets import QDockWidget
from scipy.signal import resample

import logs_setup
from registration.lib.sinogram import *
from registration.lib import geometry
from registration.interface.lib.structs import *
from registration.interface.registration_constants import RegistrationConstants
from registration.interface.registration_data import RegistrationData
from registration.interface.transformations import TransformationWidget
from registration.interface.view import ViewWidget
from registration.interface.register import RegisterWidget
from registration.interface.grangeat import GrangeatWidget
from registration.interface.plot import PlotWidget
from registration import objective_function


class Interface:
    def __init__(self, registration_constants: RegistrationConstants):
        self._registration_constants = registration_constants
        self._registration_data = RegistrationData(
            registration_constants=self._registration_constants, image_change_callback=self.render_moving_images)

        self._viewer = napari.Viewer()
        self._viewer.bind_key("Alt", self._on_alt_down)
        self._fixed_image_layer = self._viewer.add_image(
            self.registration_data.fixed_image.cpu().numpy(), colormap="yellow", interpolation2d="linear",
            name="Fixed image")
        self._moving_image_layer = self._viewer.add_image(
            np.zeros(self.registration_data.fixed_image.size()), colormap="blue", blending="additive",
            interpolation2d="linear", name="DRR")
        self._moving_image_layer.bind_key('r', self._reset)
        self._moving_image_layer.mouse_drag_callbacks.append(self._mouse_drag)

        self._sinogram2d_layer = self._viewer.add_image(
            self.registration_data.sinogram2d.cpu().numpy(), colormap="yellow", interpolation2d="linear", translate=(
                self.registration_data.fixed_image.size()[0] + 24, 0), name="Fixed sinogram")
        self._moving_sinogram_layer = self._viewer.add_image(
            np.zeros(self.registration_data.sinogram2d.size()), colormap="blue", blending="additive",
            interpolation2d="linear", translate=(
                self.registration_data.fixed_image.size()[0] + 24, 0), name="Moving sinogram")

        self._view_params = ViewParams(translation_sensitivity=0.06, rotation_sensitivity=0.002)

        self._key_states = {"Alt": False}

        self._view_widget = ViewWidget(self.set_view_params)
        self._viewer.window.add_dock_widget(
            self._view_widget, name="View options", area="left", menu=self._viewer.window.window_menu)

        initial_transformation = self.registration_constants.transformation_ground_truth
        if initial_transformation is None:
            initial_transformation = Transformation.random(device=self.registration_constants.device)
        self._transformation_widget = TransformationWidget(
            initial_transformation=initial_transformation, refresh_render_function=self.render_drr,
            save_path=pathlib.Path("cache/saved_transformations.pkl"))
        self._viewer.window.add_dock_widget(
            self._transformation_widget, name="Transformations", area="right", menu=self._viewer.window.window_menu)

        objective_functions = {"drr": self.objective_function_drr, "grangeat": self.objective_function_grangeat}

        self._register_widget = RegisterWidget(
            transformation_widget=self._transformation_widget, objective_functions=objective_functions,
            fixed_image_crop_callback=self._re_crop_fixed_image, hyper_parameter_save_path=pathlib.Path(
                "cache/saved_hyperparameters.pkl"), fixed_image_size=self._registration_data.fixed_image.size())
        self._viewer.window.add_dock_widget(
            self._register_widget, name="Register", area="right", menu=self._viewer.window.window_menu, tabify=True)

        self._grangeat_widget = GrangeatWidget(
            moving_image_changed_signal=self._moving_image_layer.events.data, registration_data=self.registration_data,
            render_moving_sinogram_callback=self.render_moving_sinogram)
        self._viewer.window.add_dock_widget(
            self._grangeat_widget, name="Sinograms", area="right", menu=self._viewer.window.window_menu, tabify=True)

        self._plot_widget = PlotWidget(
            transformation_widget=self._transformation_widget, objective_functions=objective_functions,
            window=self._viewer.window)
        self._viewer.window.add_dock_widget(
            self._plot_widget, name="Landscape plotting", area="right", menu=self._viewer.window.window_menu,
            tabify=True)

        self.render_drr()
        self.render_moving_sinogram()

    @property
    def device(self):
        return self._registration_constants.device

    @property
    def registration_constants(self) -> RegistrationConstants:
        return self._registration_constants

    @property
    def registration_data(self) -> RegistrationData:
        return self._registration_data

    def get_view_params(self) -> ViewParams:
        return self._view_params

    def set_view_params(self, value: ViewParams) -> None:
        self._view_params = value

    view_params = property(get_view_params, set_view_params)

    def resample_sinogram3d(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self._registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        source_position = self.registration_constants.scene_geometry.source_position(device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        return self._registration_constants.sinogram3d.resample(ph_matrix, self._registration_data.sinogram2d_grid)

    def generate_drr(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self._registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        return geometry.generate_drr(
            self._registration_constants.ct_volume, transformation=transformation.to(device=self.device),
            voxel_spacing=self._registration_constants.ct_spacing,
            detector_spacing=self._registration_constants.fixed_image_spacing, scene_geometry=SceneGeometry(
                source_distance=self._registration_constants.scene_geometry.source_distance,
                fixed_image_offset=self._registration_data.fixed_image_offset),
            output_size=self._registration_data.fixed_image.size())

    def objective_function_drr(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self._registration_data.fixed_image, self.generate_drr(transformation))

    def objective_function_grangeat(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self._registration_data.sinogram2d, self.resample_sinogram3d(transformation))

    def render_drr(self) -> None:
        moved_drr = self.generate_drr(self._transformation_widget.get_current_transformation())
        self._moving_image_layer.data = moved_drr.cpu().numpy()

    def render_moving_sinogram(self) -> None:
        resampled_sinogram = self.resample_sinogram3d(self._transformation_widget.get_current_transformation())
        self._moving_sinogram_layer.data = resampled_sinogram.cpu().numpy()

    def render_moving_images(self) -> None:
        self.render_drr()
        self.render_moving_images()

    def _re_crop_fixed_image(self, cropping: Cropping) -> None:
        self._registration_data.hyperparameters = HyperParameters(
            cropping=cropping, source_offset=self._registration_data.hyperparameters.source_offset)
        self._fixed_image_layer.data = self.registration_data.fixed_image.cpu().numpy()
        self._fixed_image_layer.translate = (cropping.top, cropping.left)
        self._moving_image_layer.translate = (cropping.top, cropping.left)
        self._sinogram2d_layer.data = self.registration_data.sinogram2d.cpu().numpy()
        self.render_drr()
        self.render_moving_sinogram()

    # Event callbacks:
    def _on_alt_down(self, viewer):
        self._key_states["Alt"] = True
        yield
        self._key_states["Alt"] = False

    def _reset(self, layer):
        reset_transformation = self.registration_constants.transformation_ground_truth
        if reset_transformation is None:
            reset_transformation = Transformation.random(device=self.registration_constants.device)
        self._transformation_widget.set_current_transformation(reset_transformation)
        self.render_drr()
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
                self._transformation_widget.set_current_transformation(
                    Transformation(
                        rotation=torch.tensor(
                            rot_combined.as_rotvec(), device=self._transformation_widget.get_current_transformation(

                            ).rotation.device, dtype=self._transformation_widget.get_current_transformation(

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
                self._transformation_widget.set_current_transformation(
                    Transformation(
                        translation=tr, rotation=self._transformation_widget.get_current_transformation().rotation))
                self.render_drr()
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass


def main(*, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool, save_to_cache: bool,
         sinogram_size: int, x_ray: str | None = None, new_drr_size: torch.Size = torch.Size([1000, 1000])) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    registration_constants = RegistrationConstants(
        path, cache_directory, load_cached, regenerate_drr, save_to_cache, sinogram_size, x_ray, device,
        new_drr_size=new_drr_size)

    interface = Interface(registration_constants)

    napari.run()

    return 0


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument(
        "-c", "--cache-directory", type=str, default="cache",
        help="Set the directory where data that is expensive to calculate will be saved. The default "
             "is 'cache'.")
    parser.add_argument(
        "-p", "--ct-nrrd-path", type=str,
        help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
             "synthetic data will be used instead - note that in this case, data will not be saved to "
             "the cache.")
    parser.add_argument(
        "-i", "--no-load", action='store_true', help="Do not load any pre-calculated data from the cache.")
    parser.add_argument(
        "-r", "--regenerate-drr", action='store_true',
        help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument(
        "-s", "--sinogram-size", type=int, default=256,
        help="The number of values of r, theta and phi to calculate plane integrals for, "
             "and the width and height of the grid of samples taken to approximate each integral. The "
             "computational expense of the 3D Radon transform is O(sinogram_size^5).")
    parser.add_argument(
        "-x", "--x-ray", type=str,
        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
             "this is provided, the X-ray will by used instead of any DRR.")
    parser.add_argument(
        "-d", "--drr-size", type=int, default=1000,
        help="The size of the square DRR to generate as the fixed image if no X-ray is given.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    ret = main(
        path=args.ct_nrrd_path, cache_directory=args.cache_directory, load_cached=not args.no_load,
        regenerate_drr=args.regenerate_drr, save_to_cache=not args.no_save, sinogram_size=args.sinogram_size,
        x_ray=args.x_ray if "x_ray" in vars(args) else None, new_drr_size=torch.Size([args.drr_size, args.drr_size]))

    exit(ret)
