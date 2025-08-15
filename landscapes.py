import argparse
import gc
import logging
import os
from typing import Type, Tuple
import time
import math
from datetime import datetime
import copy

import pathlib
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pyswarms
from torch.fx.passes.tools_common import Names
from tqdm import tqdm

import logs_setup
from registration import data
from registration.lib import sinogram
from registration.lib import geometry
from registration.lib.structs import *
from registration.lib import grangeat
from registration import plot_data
from registration import objective_function
from registration import pre_computed
from registration import drr
from notification import pushover


class RegistrationInfo(NamedTuple):
    name: str
    scene_geometry: SceneGeometry
    ct_volume: torch.Tensor
    fixed_image: torch.Tensor
    ct_spacing: torch.Tensor
    sinogram3ds: list[sinogram.Sinogram]
    fixed_image_spacing: torch.Tensor
    sinogram2d_grid: Sinogram2dGrid
    sinogram2d: torch.Tensor


def get_registration_info(*, cache_directory: str, ct_path: str | pathlib.Path, x_ray_path: str | pathlib.Path | None,
                          downsample_factor: int, transformation_ground_truth: Transformation,
                          name: str) -> RegistrationInfo | None:
    device = torch.device("cuda")
    ct_volume, ct_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor=downsample_factor)
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)
    sinogram_size = int(math.ceil(pow(ct_volume.numel(), 1.0 / 3.0)))

    def get_sinogram(sinogram_type: Type[sinogram.SinogramType]) -> sinogram.Sinogram | None:
        sinogram3d = None
        sinogram_hash = data.deterministic_hash_sinogram(ct_path, sinogram_type, sinogram_size, downsample_factor)
        volume_spec = data.load_cached_volume(cache_directory, sinogram_hash)
        if volume_spec is not None:
            _, sinogram3d = volume_spec
        if sinogram3d is None:
            res = pre_computed.calculate_volume_sinogram(cache_directory, ct_volume, voxel_spacing=ct_spacing,
                                                         ct_volume_path=ct_path,
                                                         volume_downsample_factor=downsample_factor, save_to_cache=True,
                                                         sinogram_size=sinogram_size, sinogram_type=sinogram_type)
            if res is None:
                return None
            sinogram3d, _ = res
        return sinogram3d

    sinogram3ds = [get_sinogram(tp) for tp in [sinogram.SinogramClassic, sinogram.SinogramHEALPix]]

    for s in sinogram3ds:
        if s is None:
            return None

    if x_ray_path is None:
        fixed_image_spacing, scene_geometry, fixed_image, _ = drr.generate_drr_as_target(cache_directory, ct_path,
                                                                                         ct_volume, ct_spacing,
                                                                                         save_to_cache=False, size=None,
                                                                                         transformation=transformation_ground_truth.to(
                                                                                             device=ct_volume.device))
    else:
        # Load the given X-ray
        fixed_image, fixed_image_spacing, scene_geometry = data.read_dicom(str(x_ray_path))
        fixed_image = fixed_image.to(device=device)

        f_middle = 0.25
        fixed_image = fixed_image[int(float(fixed_image.size()[0]) * .5 * (1. - f_middle)):int(
            float(fixed_image.size()[0]) * .5 * (1. + f_middle)), :]

    sinogram2d_counts = max(fixed_image.size()[0], fixed_image.size()[1])
    image_diag: float = (fixed_image_spacing.flip(dims=(0,)) * torch.tensor(fixed_image.size(),
                                                                            dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d = grangeat.calculate_fixed_image(fixed_image, source_distance=scene_geometry.source_distance,
                                                detector_spacing=fixed_image_spacing, output_grid=sinogram2d_grid)

    return RegistrationInfo(name=name, scene_geometry=scene_geometry, ct_volume=ct_volume, fixed_image=fixed_image,
                            ct_spacing=ct_spacing, sinogram3ds=sinogram3ds, fixed_image_spacing=fixed_image_spacing,
                            sinogram2d_grid=sinogram2d_grid, sinogram2d=sinogram2d)


SAVE_DIRECTORY = "figures/landscapes"


class LandscapeTask:
    def __init__(self, registration_info: RegistrationInfo, *, gt_transformation: Transformation, landscape_size: int,
                 landscape_range: torch.Tensor):
        self._registration_info = registration_info
        self._gt_transformation = gt_transformation
        self._landscape_size = landscape_size
        assert landscape_range.size() == torch.Size([Transformation.zero().vectorised().numel()])
        self._landscape_range = landscape_range

        self._objective_functions = {"drr": self.objective_function_drr,
                                     "grangeat_classic": self.objective_function_grangeat_classic,
                                     "grangeat_healpix": self.objective_function_grangeat_healpix}

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def registration_info(self) -> RegistrationInfo:
        return self._registration_info

    def resample_sinogram3d(self, transformation: Transformation, sinogram_index: int) -> torch.Tensor:
        source_position = self.registration_info.scene_geometry.source_position(device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        return self.registration_info.sinogram3ds[sinogram_index].resample_cuda_texture(ph_matrix,
                                                                                        self.registration_info.sinogram2d_grid)

    def generate_drr(self, transformation: Transformation) -> torch.Tensor:
        return geometry.generate_drr(self.registration_info.ct_volume,
                                     transformation=transformation.to(device=self.device),
                                     voxel_spacing=self.registration_info.ct_spacing,
                                     detector_spacing=self.registration_info.fixed_image_spacing,
                                     scene_geometry=SceneGeometry(
                                         source_distance=self.registration_info.scene_geometry.source_distance),
                                     output_size=self.registration_info.fixed_image.size())

    def objective_function_drr(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self.registration_info.fixed_image, self.generate_drr(transformation))

    def objective_function_grangeat_classic(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self.registration_info.sinogram2d, self.resample_sinogram3d(transformation, 0))

    def objective_function_grangeat_healpix(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self.registration_info.sinogram2d, self.resample_sinogram3d(transformation, 1))

    def run(self, *, objective_function_name: str):
        #
        plt.figure()
        plt.imshow(self.registration_info.fixed_image.cpu().numpy())
        drr_gt = self.generate_drr(self._gt_transformation)
        plt.figure()
        plt.imshow(drr_gt.cpu().numpy())
        logger.info("Sim @ G.T. = {}".format(-objective_function.zncc(self.registration_info.fixed_image, drr_gt)))
        plt.show()
        #
        obj_func = self._objective_functions[objective_function_name]

        gt_vectorised = self._gt_transformation.vectorised()

        def landscape2(axes, param1: int, param2: int):
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
                    landscape[j, i] = obj_func(get_transformation(i, j))

            param2_grid, param1_grid = torch.meshgrid(param2_grid, param1_grid)
            axes.plot_surface(param1_grid.clone().detach().cpu().numpy(), param2_grid.clone().detach().cpu().numpy(),
                              landscape.clone().detach().cpu().numpy(), cmap=cm.get_cmap("viridis"))

        class LandscapePlot(NamedTuple):
            i: int
            j: int
            fname: str
            xlabel: str
            ylabel: str

        landscapes: list[LandscapePlot] = [LandscapePlot(0, 1, "rxry", "rx", "ry"),
                                           LandscapePlot(1, 2, "ryrz", "ry", "rz"),
                                           LandscapePlot(3, 4, "txty", "tx", "ty"),
                                           LandscapePlot(4, 5, "tytz", "ty", "tz")]

        for lp in landscapes:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
            landscape2(axes, lp.i, lp.j)
            axes.set_xlabel(lp.xlabel)
            axes.set_ylabel(lp.ylabel)
            plt.savefig(SAVE_DIRECTORY + "/landscape_{}_{}.png".format(self.registration_info.name, lp.fname))

        plt.show()


CT_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct001",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct002",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct003",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct005",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct007",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct008",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct010",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct014"]
XRAY_DICOM_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray001.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray002.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray003.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray005.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray007.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray008.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray010.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray014.dcm"]


def evaluate_and_plot_landscape(*, cache_directory: str, ct_path: str, x_ray_path: str | None, downsample_factor: int):
    if x_ray_path is None:
        transformation_ground_truth = Transformation.random()
    else:
        transformation_ground_truth = Transformation(rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0]), translation=torch.tensor([0.0, 0.0, 0.0]))

    try:
        registration_info = get_registration_info(cache_directory=cache_directory, ct_path=ct_path,
                                                  x_ray_path=x_ray_path, downsample_factor=downsample_factor,
                                                  transformation_ground_truth=transformation_ground_truth,
                                                  name=str(pathlib.Path(ct_path).name))
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise e
        logger.warning("Not enough memory for run; skipping.")
        return  # None

    task = LandscapeTask(registration_info, gt_transformation=transformation_ground_truth, landscape_size=30,
                         landscape_range=torch.Tensor([2.0, 2.0, 2.0, 100.0, 100.0, 100.0]))

    task.run(objective_function_name="drr")


def main(cache_directory: str, drr_as_target: bool):
    count: int = len(CT_PATHS)
    if not drr_as_target:
        assert len(XRAY_DICOM_PATHS) == count
    for i in range(count):
        evaluate_and_plot_landscape(cache_directory=cache_directory, ct_path=CT_PATHS[i],
                                    x_ray_path=None if drr_as_target else XRAY_DICOM_PATHS[i], downsample_factor=1)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # for outputting images
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["scatter.marker"] = 'x'
    plt.rcParams["font.size"] = 15  # figures are includes in latex at half size, so 18 is desired size. matplotlib
    # scales up by 1.2 (God only knows why), so setting to 15

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
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, drr_as_target=args.drr_target)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
