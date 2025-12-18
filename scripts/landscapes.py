import argparse
import os
from typing import NamedTuple, Tuple, Callable
import copy

import pathlib
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pyswarms
from tqdm import tqdm

from reg23_experiments.notification import logs_setup
from reg23_experiments.registration.interface.registration_data import RegistrationData
from reg23_experiments.registration.interface.lib.structs import Target, SavedXRayParams, HyperParameters, Cropping
from reg23_experiments.registration.lib import sinogram
from reg23_experiments.registration.lib import geometry
from reg23_experiments.registration.lib.structs import Transformation, SceneGeometry
from reg23_experiments.registration import objective_function
from reg23_experiments.notification import pushover
from reg23_experiments.registration.plot_data import LandscapePlotData

SAVE_DIRECTORY = pathlib.Path("data/temp/landscapes")


class LandscapeTask:
    def __init__(self, registration_data: RegistrationData, *, gt_transformation: Transformation, landscape_size: int,
                 landscape_range: torch.Tensor):
        self._registration_data = registration_data
        self._gt_transformation = gt_transformation
        self._landscape_size = landscape_size
        assert landscape_range.size() == torch.Size([Transformation.zero().vectorised().numel()])
        self._landscape_range = landscape_range

        self._images_functions = {"drr": self.images_drr, "gr_classic": self.images_grangeat_classic,
                                  # "gr_healpix": self.objective_function_grangeat_healpix
                                  }

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def registration_data(self) -> RegistrationData:
        return self._registration_data

    def resample_sinogram3d(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self.registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        source_position = torch.tensor([0., 0., self.registration_data.source_distance], device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        return next(iter(self.registration_data.ct_sinograms.values()))[
            self.registration_data.hyperparameters.downsample_level].resample_cuda_texture(ph_matrix,
                                                                                           self.registration_data.sinogram2d_grid)

    def generate_drr(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self.registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        return geometry.generate_drr(self.registration_data.ct_volume_at_current_level, transformation=transformation,
                                     voxel_spacing=self.registration_data.ct_spacing_at_current_level,
                                     detector_spacing=self.registration_data.fixed_image_spacing_at_current_level.to(
                                         device=self.device),  #
                                     scene_geometry=SceneGeometry(
                                         source_distance=self.registration_data.source_distance,
                                         fixed_image_offset=self.registration_data.fixed_image_offset),
                                     output_size=self.registration_data.cropped_target.size())

    def images_drr(self, transformation: Transformation) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.registration_data.fixed_image, self.generate_drr(transformation)

    def images_grangeat_classic(self, transformation: Transformation) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.registration_data.sinogram2d, self.resample_sinogram3d(transformation)

    # def objective_function_grangeat_healpix(self, transformation: Transformation) -> torch.Tensor:
    #     return -objective_function.zncc(self.registration_data.sinogram2d, self.resample_sinogram3d(transformation, 1))

    @staticmethod
    def sim_metric_ncc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return -objective_function.ncc(xs, ys)

    def run(self, *, images_function_name: str, downsample_level: int,
            sim_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], show: bool = False) -> None:
        if images_function_name != "drr":
            old_hyperparameters = self.registration_data.hyperparameters
            zero_crop = Cropping.zero(self.registration_data.images_2d_full[0].size())
            self.registration_data.hyperparameters = HyperParameters(
                cropping=Cropping(right=zero_crop.right, top=self.registration_data.hyperparameters.cropping.top,
                                  left=zero_crop.left, bottom=self.registration_data.hyperparameters.cropping.bottom),
                source_offset=self.registration_data.hyperparameters.source_offset, downsample_level=downsample_level)

        if show:
            plt.figure()
            plt.imshow(self.registration_data.fixed_image.cpu().numpy())
            drr_gt = self.generate_drr(self._gt_transformation)
            plt.figure()
            plt.imshow(drr_gt.cpu().numpy())
            logger.info("Sim @ G.T. = {}".format(sim_metric(self.registration_data.fixed_image, drr_gt)))
            plt.show()

        images_func = self._images_functions[images_function_name]

        gt_vectorised = self._gt_transformation.vectorised()

        def landscape2(param1: int, param2: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            param1_grid = torch.linspace(gt_vectorised[param1] - 0.5 * self._landscape_range[param1],
                                         gt_vectorised[param1] + 0.5 * self._landscape_range[param1],
                                         self._landscape_size)
            param2_grid = torch.linspace(gt_vectorised[param2] - 0.5 * self._landscape_range[param2],
                                         gt_vectorised[param2] + 0.5 * self._landscape_range[param2],
                                         self._landscape_size)

            def get_transformation(param1_index: int, param2_index: int) -> Transformation:
                params = copy.deepcopy(gt_vectorised)
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
            torch.save(  #
                LandscapePlotData(  #
                    xray_path=self.registration_data.target.xray_path, param1=lp.i, param2=lp.j, label1=lp.xlabel,
                    label2=lp.ylabel, values1=values1, values2=values2, height=height),
                SAVE_DIRECTORY / "{}_{}_{}.pkl".format(  #
                    pathlib.Path(  #
                        self.registration_data.ct_path if self.registration_data.target.xray_path is None else self.registration_data.target.xray_path).stem,
                    images_function_name, lp.fname))

        if images_function_name != "drr":
            self.registration_data.hyperparameters = old_hyperparameters


CT_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct001",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct002",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct003",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct005",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct007",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct014"]
XRAY_DICOM_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray001.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray002.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray003.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray005.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray007.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray014.dcm"]
XRAY_PARAMS_SAVE_PATH = pathlib.Path("saved") / "xray_params_library.pkl"


def evaluate_and_save_landscape(*, cache_directory: str, ct_path: str, xray_path: str | None, show: bool = False):
    flipped: bool = False
    hyperparameters = None

    if xray_path is not None:
        if not XRAY_PARAMS_SAVE_PATH.is_file():
            logger.error("No X-ray parameter save file '{}' found; cannot load parameters."
                         "".format(str(XRAY_PARAMS_SAVE_PATH)))
            return
        saved_parameters = torch.load(XRAY_PARAMS_SAVE_PATH, weights_only=False)
        if not isinstance(saved_parameters, dict):
            logger.error("X-ray parameter save file '{}' has invalid type '{}'; cannot load parameters."
                         "".format(str(XRAY_PARAMS_SAVE_PATH), type(saved_parameters).__name__))
            return
        if xray_path not in saved_parameters:
            logger.error("No parameters saved for X-ray {} in parameter save file '{}'; cannot load parameters."
                         "".format(xray_path, str(XRAY_PARAMS_SAVE_PATH)))
            return
        loaded = saved_parameters[xray_path]
        assert isinstance(loaded, SavedXRayParams)
        flipped = loaded.flipped
        transformation_ground_truth = loaded.transformation
        hyperparameters = loaded.hyperparameters

    try:
        registration_data = RegistrationData(cache_directory=cache_directory, ct_path=ct_path,
                                             target=Target(xray_path=xray_path, flipped=flipped), load_cached=True,
                                             sinogram_types=[sinogram.SinogramClassic], sinogram_size=None,
                                             regenerate_drr=True, save_to_cache=True, new_drr_size=None,
                                             device=torch.device("cuda"))
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise e
        logger.warning("Not enough memory for run; skipping.")
        return  # None

    if xray_path is None:
        transformation_ground_truth = registration_data.transformation_gt

    if hyperparameters is not None:
        registration_data.hyperparameters = hyperparameters

    task = LandscapeTask(registration_data, gt_transformation=transformation_ground_truth, landscape_size=30,
                         landscape_range=torch.Tensor([1.0, 1.0, 1.0, 30.0, 30.0, 300.0]))

    task.run(images_function_name="drr", sim_metric=LandscapeTask.sim_metric_ncc, downsample_level=2, show=show)
    task.run(images_function_name="gr_classic", sim_metric=LandscapeTask.sim_metric_ncc, downsample_level=2, show=show)


def main(cache_directory: str, drr_as_target: bool, show: bool = False):
    if not drr_as_target:
        assert len(XRAY_DICOM_PATHS) == len(CT_PATHS)
    count: int = 1  # len(CT_PATHS)
    for i in range(count):
        evaluate_and_save_landscape(cache_directory=cache_directory, ct_path=CT_PATHS[i],
                                    xray_path=None if drr_as_target else XRAY_DICOM_PATHS[i], show=show)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    # parser.add_argument(
    #     "-p", "--ct-nrrd-path", type=str,
    #     help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
    #          "synthetic data will be used instead - note that in this case, data will not be saved to "
    #          "the cache.")
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-d", "--drr-target", action="store_true",
                        help="Generate a DRR at a random transformation to register to, instead of using an X-ray image.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, drr_as_target=args.drr_target, show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
