import os
import argparse
import logging.config
import time
from typing import NamedTuple

import numpy as np
import torch
import napari
import scipy
from magicgui import magicgui, widgets

from registration.lib.sinogram import *
from registration import drr
from registration import data
from registration import script
from registration.lib import geometry


class SavedTransformation(NamedTuple):
    name: str
    value: Transformation


def main(*, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool, save_to_cache: bool,
         sinogram_size: int, x_ray: str | None = None) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    vol_data, voxel_spacing, sinogram3d = script.get_volume_and_sinogram(path, cache_directory, load_cached=load_cached,
                                                                         save_to_cache=save_to_cache,
                                                                         sinogram_size=sinogram_size, device=device)

    if x_ray is None:
        # Load / generate a DRR through the volume
        drr_spec = None
        if not regenerate_drr and path is not None:
            drr_spec = data.load_cached_drr(cache_directory, path)

        if drr_spec is None:
            drr_spec = drr.generate_new_drr(cache_directory, path, vol_data, voxel_spacing, device=device,
                                            save_to_cache=save_to_cache)

        detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth = (
            drr_spec)
        del drr_spec
    else:
        # Load the given X-ray
        drr_image, detector_spacing, scene_geometry = data.read_dicom(x_ray)
        drr_image = drr_image.to(device=device)
        f_middle = 0.5
        drr_image = drr_image[int(float(drr_image.size()[0]) * .5 * (1. - f_middle)):int(
            float(drr_image.size()[0]) * .5 * (1. + f_middle)), :]

        logger.info("Calculating 2D sinogram (the fixed image)...")

        sinogram2d_counts = 1024
        image_diag: float = (
                detector_spacing * torch.tensor(drr_image.size(), dtype=torch.float32)).square().sum().sqrt().item()
        sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                           LinearRange(-.5 * image_diag, .5 * image_diag))
        sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

        fixed_image = grangeat.calculate_fixed_image(drr_image, source_distance=scene_geometry.source_distance,
                                                     detector_spacing=detector_spacing, output_grid=sinogram2d_grid)

        del sinogram2d_grid
        logger.info("X-ray sinogram calculated.")

    fixed_image_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, fixed_image.size(), device=device)
    transformation = transformation_ground_truth if x_ray is None else Transformation.random(device=device)

    viewer = napari.Viewer()
    fixed_image_layer = viewer.add_image(drr_image.cpu().numpy(), colormap="yellow", interpolation2d="linear")
    moving_image_layer = viewer.add_image(np.zeros((1, 1)), colormap="blue", blending="additive",
                                          interpolation2d="linear")
    # saved_transformations = [SavedTransformation("initial", transformation)]
    key_states = {"Alt": False}

    def refresh():
        nonlocal scene_geometry, transformation, moving_image_layer
        moved_drr = geometry.generate_drr(vol_data, transformation=transformation, voxel_spacing=voxel_spacing,
                                          detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                          output_size=torch.Size([1000, 1000]), samples_per_ray=500)
        moving_image_layer.data = moved_drr.cpu().numpy()

    @viewer.bind_key("Alt")
    def on_alt_down(viewer):
        nonlocal key_states
        key_states["Alt"] = True
        yield
        key_states["Alt"] = False

    @moving_image_layer.bind_key('r')
    def reset(event=None):
        nonlocal transformation, transformation_ground_truth, x_ray
        transformation = transformation_ground_truth if x_ray is None else Transformation.random(device=device)
        refresh()
        logger.info("Reset")

    @moving_image_layer.mouse_drag_callbacks.append
    def mouse_drag(layer, event):
        nonlocal transformation
        if event.button == 1 and key_states["Alt"]:  # Alt-left click drag
            # mouse down
            dragged = False
            drag_start = np.array(event.position)
            rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = 0.002 * (np.array(event.position) - drag_start)
                euler_angles = [delta[0], delta[1], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler("zyx", euler_angles)
                rot_combined = rot_euler * rotation_start
                transformation = Transformation(
                    torch.tensor(rot_combined.as_rotvec(), device=transformation.rotation.device,
                                 dtype=transformation.rotation.dtype), transformation.translation)
                refresh()
                yield
                # on release
                if dragged:
                    # dragged
                    pass
                else:
                    # just clicked
                    pass
        elif event.button == 2 and key_states["Alt"]:  # Alt-right click drag
            # mouse down
            dragged = False
            drag_start = torch.tensor(event.position)
            # rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            translation_start = transformation.translation[0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = 0.02 * (torch.tensor(event.position) - drag_start).flip((0,))
                transformation.translation[0:2] = (translation_start + delta).to(
                    device=transformation.translation.device)
                refresh()
                yield
                # on release
                if dragged:
                    # dragged
                    pass
                else:
                    # just clicked
                    pass

    @magicgui(auto_call=True, z_translation={"widget_type": "FloatSlider", "min": -300.0, "max": 300, "step": 0.2})
    def moving_parameters(z_translation: float = 0.0):
        nonlocal transformation
        transformation.translation[2] = z_translation
        refresh()

    save_button = widgets.PushButton(label="Save transformation")
    saved_transformations_widget = widgets.Select(choices=["initial"])
    saved_transformations_widget.set_choice("initial", transformation)
    set_to_saved_button = widgets.PushButton(label="Set transformation to selected")
    print(saved_transformations_widget.choices)

    # !!! for some reason the 'choices' member of the widget starts correctly containing the transformations, then next
    # time we look at it in the callback, it just contains the names!

    @save_button.changed.connect
    def _():
        nonlocal transformation, saved_transformations_widget
        new_name = "{}".format(time.time())
        saved_transformations_widget.set_choice(new_name, transformation)

    @set_to_saved_button.changed.connect
    def _():
        nonlocal transformation, saved_transformations_widget
        print(saved_transformations_widget.choices)
        current = saved_transformations_widget.value
        if len(current) == 1:
            print(current[0])
            print(saved_transformations_widget.get_choice(current[0]))
            # transformation = saved_transformations_widget.get_choice(current[0])[0].to(device=transformation.device())
            # refresh()
        elif len(current) == 0:
            logger.warning("No transformation selected.")
        else:
            logger.warning("Multiple transformations selected.")

    dock = widgets.Container(
        widgets=[moving_parameters, save_button, saved_transformations_widget, set_to_saved_button])

    viewer.window.add_dock_widget(dock)

    refresh()

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
