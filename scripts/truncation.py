import argparse
import os
from typing import Callable, Tuple, NamedTuple

import pathlib
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm

from notification import logs_setup
from notification import pushover
from registration.lib.structs import Transformation, SceneGeometry
from registration.interface.lib.structs import Target, Cropping, HyperParameters
from registration.lib.sinogram import SinogramClassic
from registration.lib import geometry
from registration import objective_function
from registration.plot_data import LandscapePlotData
from registration import data
from registration import drr

import Extension as reg23

SAVE_DIRECTORY = pathlib.Path("data/temp/truncation")


class RegistrationData:
    class HyperparameterDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path, fixed image and hyperparameters used
        """
        cropped_target: torch.Tensor  # The target image with the cropping applied, but no mask applied
        fixed_image_offset: torch.Tensor
        translation_offset: torch.Tensor

    class MaskTransformationDependent(NamedTuple):
        """
        @brief Struct of data that is dependent on the CT path, fixed image, hyperparameters used and the transformation
        at which the mask was generated
        """
        fixed_image: torch.Tensor  # The target

    def __init__(self, cache_directory: str, ct_path: str, device, downsample_factor: int,
                 truncation_fractions: list[float], hyperparameter_change_callback: Callable[[], None] | None = None,
                 mask_transformation_change_callback: Callable[[], None] | None = None):
        self._truncation_fractions = truncation_fractions

        def truncate(volume: torch.Tensor, fraction: float) -> torch.Tensor:
            top_bottom_chop = int(round(0.5 * fraction * float(volume.size()[0])))
            return volume[top_bottom_chop:max(top_bottom_chop + 1, volume.size()[0] - top_bottom_chop)]

        ct_volume, self._ct_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor=downsample_factor)
        ct_volume = ct_volume.to(device=device, dtype=torch.float32)
        self._ct_volumes = [ct_volume] + [truncate(ct_volume, fraction) for fraction in truncation_fractions]
        self._ct_spacing = self._ct_spacing.to(device=device)

        regenerate_drr = True

        # Load / generate a DRR through the volume
        drr_spec = None
        if not regenerate_drr:
            drr_spec = data.load_cached_drr(cache_directory, ct_path)

        if drr_spec is None:
            drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, self._ct_volumes[0], self._ct_spacing,
                                                  save_to_cache=False, size=None)

        self._fixed_image_spacing, self._scene_geometry, self._image_2d_full, self._transformation_gt = drr_spec
        del drr_spec

        self._hyperparameters = HyperParameters.zero(self.image_2d_full.size())
        self._truncation_level = 0  # this is treated the same as mask transformation
        self._mask_transformation: Transformation | None = None

        self._device = device
        self._source_distance: float = 1000.0
        self._hyperparameter_change_callback = hyperparameter_change_callback
        self._mask_transformation_change_callback = mask_transformation_change_callback

        self._hyperparameters_dirty: bool = True
        self._mask_transformation_dirty: bool = True

        self._suppress_callbacks = True
        self.refresh_hyperparameter_dependent()
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
    def ct_volumes(self) -> list[torch.Tensor]:
        return self._ct_volumes

    @property
    def ct_volume_at_current_truncation(self) -> torch.Tensor:
        if self._mask_transformation_dirty:
            self.refresh_mask_transformation_dependent()
        return self._ct_volumes[self.truncation_level]

    @property
    def ct_spacing(self) -> torch.Tensor:
        return self._ct_spacing

    # -----
    # Target and properties that depend on it, and all those above
    # -----
    @property
    def source_distance(self) -> float:
        return self._source_distance

    @property
    def image_2d_full(self) -> torch.Tensor:
        return self._image_2d_full

    @property
    def fixed_image_spacing(self) -> torch.Tensor:
        return self._fixed_image_spacing

    @property
    def transformation_gt(self) -> Transformation | None:
        return self._transformation_gt

    # -----
    # Hyperparameters and properties that depend on it, and all those above
    # -----

    @property
    def hyperparameters(self) -> HyperParameters:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_value: HyperParameters) -> None:
        self._hyperparameters_dirty = True
        self._mask_transformation_dirty = True
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
        self._mask_transformation = new_value

    @property
    def truncation_level(self) -> int:
        return self._truncation_level

    @truncation_level.setter
    def truncation_level(self, new_value: int) -> None:
        if new_value != self._truncation_level:
            self._mask_transformation_dirty = True
        self._truncation_level = new_value

    @property
    def fixed_image(self) -> torch.Tensor:
        if self._mask_transformation_dirty:
            self.refresh_mask_transformation_dependent()
        return self._mask_transformation_dependent.fixed_image

    def refresh_hyperparameter_dependent(self) -> None:
        # Cropping for the fixed image
        cropped_target = self.hyperparameters.downsampled_crop(self.image_2d_full.size()).apply(self.image_2d_full)

        # The fixed image is offset to adjust for the cropping, and according to the source offset
        # This isn't affected by downsample level
        fixed_image_offset = (self.fixed_image_spacing * self.hyperparameters.cropping.get_centre_offset(
            self.image_2d_full.size()) - self.hyperparameters.source_offset)

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

    def refresh_mask_transformation_dependent(self) -> None:
        self._mask_transformation_dirty = False

        if self.mask_transformation is None:
            fixed_image = self.cropped_target
        else:
            mask = reg23.project_drr_cuboid_mask(  #
                volume_size=torch.tensor(self.ct_volume_at_current_truncation.size(), device=self.device).flip(
                    dims=(0,)),  #
                voxel_spacing=self.ct_spacing.to(device=self.device),  #
                homography_matrix_inverse=self.mask_transformation.inverse().get_h().to(device=self.device),  #
                source_distance=self.source_distance, output_width=self.cropped_target.size()[1],  #
                output_height=self.cropped_target.size()[0],  #
                output_offset=self.fixed_image_offset.to(device=self.device, dtype=torch.float64),  #
                detector_spacing=self.fixed_image_spacing.to(device=self.device)  #
            )
            fixed_image = mask * self.cropped_target
            del mask

        self._mask_transformation_dependent = RegistrationData.MaskTransformationDependent(fixed_image=fixed_image)

        if not self.suppress_callbacks and self._mask_transformation_change_callback is not None:
            self._mask_transformation_change_callback()


class LandscapeTask:
    def __init__(self, registration_data: RegistrationData, *, landscape_size: int, landscape_range: torch.Tensor):
        self._registration_data = registration_data
        self._landscape_size = landscape_size
        assert landscape_range.size() == torch.Size([Transformation.zero().vectorised().numel()])
        self._landscape_range = landscape_range

        self._images_functions = {"drr": self.images_drr,  # "gr_classic": self.images_grangeat_classic,
                                  # "gr_healpix": self.objective_function_grangeat_healpix
                                  }

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def registration_data(self) -> RegistrationData:
        return self._registration_data

    # def resample_sinogram3d(self, transformation: Transformation) -> torch.Tensor:
    #     # Applying the translation offset
    #     translation = transformation.translation.clone()
    #     translation[0:2] += self.registration_data.translation_offset.to(device=transformation.device)
    #     transformation = Transformation(rotation=transformation.rotation, translation=translation)
    #
    #     source_position = torch.tensor([0., 0., self.registration_data.source_distance], device=self.device)
    #     p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    #     ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
    #     return next(iter(self.registration_data.ct_sinograms.values()))[
    #         self.registration_data.hyperparameters.downsample_level].resample_cuda_texture(ph_matrix,
    #                                                                                        self.registration_data.sinogram2d_grid)

    def generate_drr(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = transformation.translation.clone()
        translation[0:2] += self.registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        return geometry.generate_drr(self.registration_data.ct_volume_at_current_truncation,
                                     transformation=transformation, voxel_spacing=self.registration_data.ct_spacing,
                                     detector_spacing=self.registration_data.fixed_image_spacing.to(device=self.device),
                                     #
                                     scene_geometry=SceneGeometry(
                                         source_distance=self.registration_data.source_distance,
                                         fixed_image_offset=self.registration_data.fixed_image_offset),
                                     output_size=self.registration_data.cropped_target.size())

    def images_drr(self, transformation: Transformation) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.registration_data.fixed_image, self.generate_drr(transformation)

    # def images_grangeat_classic(self, transformation: Transformation) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return self.registration_data.sinogram2d, self.resample_sinogram3d(transformation)

    # def objective_function_grangeat_healpix(self, transformation: Transformation) -> torch.Tensor:
    #     return -objective_function.zncc(self.registration_data.sinogram2d, self.resample_sinogram3d(transformation, 1))

    @staticmethod
    def sim_metric_ncc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return -objective_function.ncc(xs, ys)

    def run(self, *, images_function_name: str, sim_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            name: str | None = None, show: bool = False) -> None:
        if images_function_name != "drr":
            old_hyperparameters = self.registration_data.hyperparameters
            zero_crop = Cropping.zero(self.registration_data.image_2d_full.size())
            self.registration_data.hyperparameters = HyperParameters(
                cropping=Cropping(right=zero_crop.right, top=self.registration_data.hyperparameters.cropping.top,
                                  left=zero_crop.left, bottom=self.registration_data.hyperparameters.cropping.bottom),
                source_offset=self.registration_data.hyperparameters.source_offset, downsample_level=0)

        if show:
            plt.figure()
            plt.imshow(self.registration_data.fixed_image.cpu().numpy())
            drr_gt = self.generate_drr(self.registration_data.transformation_gt)
            plt.figure()
            plt.imshow(drr_gt.cpu().numpy())
            logger.info("Sim @ G.T. = {}".format(sim_metric(self.registration_data.fixed_image, drr_gt)))
            plt.show()

        images_func = self._images_functions[images_function_name]

        gt_vectorised = self.registration_data.transformation_gt.vectorised()

        def landscape2(param1: int, param2: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            param1_grid = torch.linspace(gt_vectorised[param1] - 0.5 * self._landscape_range[param1],
                                         gt_vectorised[param1] + 0.5 * self._landscape_range[param1],
                                         self._landscape_size)
            param2_grid = torch.linspace(gt_vectorised[param2] - 0.5 * self._landscape_range[param2],
                                         gt_vectorised[param2] + 0.5 * self._landscape_range[param2],
                                         self._landscape_size)

            def get_transformation(param1_index: int, param2_index: int) -> Transformation:
                params = gt_vectorised.clone()
                params[param1] = param1_grid[param1_index]
                params[param2] = param2_grid[param2_index]
                return Transformation.from_vector(params)

            landscape = torch.zeros(self._landscape_size, self._landscape_size)
            for j in tqdm(range(self._landscape_size)):
                for i in range(self._landscape_size):
                    landscape[j, i] = sim_metric(*images_func(get_transformation(i, j)))

            return param1_grid, param2_grid, landscape

        class LandscapePlot(NamedTuple):
            i: int
            j: int
            fname: str
            xlabel: str
            ylabel: str

        landscapes: list[LandscapePlot] = [LandscapePlot(0, 1, "rxry", "$r_x$", "$r_y$"),
                                           LandscapePlot(1, 2, "ryrz", "$r_y$", "$r_z$"),
                                           LandscapePlot(3, 4, "txty", "$t_x$", "$t_y$"),
                                           LandscapePlot(4, 5, "tytz", "$t_y$", "$t_z$")]

        for lp in landscapes:
            values1, values2, height = landscape2(lp.i, lp.j)
            file_name: str = "{}_{}.pkl".format(images_function_name,
                                                lp.fname) if name is None else "{}_{}_{}.pkl".format(
                images_function_name, name, lp.fname)
            torch.save(  #
                LandscapePlotData(  #
                    xray_path="", param1=lp.i, param2=lp.j, label1=lp.xlabel, label2=lp.ylabel, values1=values1,
                    values2=values2, height=height), SAVE_DIRECTORY / file_name)

        if images_function_name != "drr":
            self.registration_data.hyperparameters = old_hyperparameters


def evaluate_and_save_landscape(*, cache_directory: str, ct_path: str, device, show: bool = False):
    truncation_fractions = [0.1, 0.2, 0.4]

    registration_data = RegistrationData(cache_directory=cache_directory, ct_path=ct_path, device=device,
                                         downsample_factor=1, truncation_fractions=truncation_fractions)

    task = LandscapeTask(registration_data, landscape_size=30,
                         landscape_range=torch.Tensor([1.0, 1.0, 1.0, 30.0, 30.0, 300.0]))

    truncation_fractions = [0.0] + truncation_fractions
    for i in range(len(truncation_fractions)):
        registration_data.truncation_level = i
        task.run(images_function_name="drr", sim_metric=LandscapeTask.sim_metric_ncc, show=show,
                 name="{:.3f}".format(truncation_fractions[i]).split(".")[1])


def main(*, cache_directory: str, ct_path: str | None, show: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_and_save_landscape(cache_directory=cache_directory, ct_path=ct_path, show=show, device=device)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to process. If not "
                             "provided, some simple synthetic data will be used instead - note that in this case, data will not be "
                             "saved to the cache.")
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, ct_path=args.ct_path if "ct_path" in vars(args) else None,
             show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
