import argparse
import gc
import logging
import os
from typing import Type, Callable
import time
import math
from datetime import datetime
import copy

import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import pyswarms

import logs_setup
from registration import script
from registration import data
from registration.lib import sinogram
from registration.lib import sinogram
from registration.lib import geometry
from registration.lib.structs import *
from registration.lib import grangeat
from registration import plot_data
from registration import objective_function

from registration.interface.registration_data import RegistrationData
from registration.interface.registration_constants import RegistrationConstants
from registration.interface.register import OptimisationResult


class RegistrationTask:
    def __init__(self, registration_constants: RegistrationConstants, *, particle_count: int):
        self._registration_constants = registration_constants
        self._particle_count = particle_count

        self._registration_data = RegistrationData(registration_constants=self._registration_constants,
                                                   image_change_callback=None)

        self._objective_functions = {"drr": self.objective_function_drr, "grangeat": self.objective_function_grangeat}

    @property
    def device(self):
        return self._registration_constants.device

    @property
    def registration_constants(self) -> RegistrationConstants:
        return self._registration_constants

    def resample_sinogram3d(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self._registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        source_position = self.registration_constants.scene_geometry.source_position(device=self.device)
        p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
        ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=self.device).to(dtype=torch.float32))
        return self._registration_constants.sinogram3d.resample_cuda_texture(ph_matrix,
                                                                             self._registration_data.sinogram2d_grid)

    def generate_drr(self, transformation: Transformation) -> torch.Tensor:
        # Applying the translation offset
        translation = copy.deepcopy(transformation.translation)
        translation[0:2] += self._registration_data.translation_offset.to(device=transformation.device)
        transformation = Transformation(rotation=transformation.rotation, translation=translation)

        return geometry.generate_drr(self._registration_constants.ct_volume,
                                     transformation=transformation.to(device=self.device),
                                     voxel_spacing=self._registration_constants.ct_spacing,
                                     detector_spacing=self._registration_constants.fixed_image_spacing,
                                     scene_geometry=SceneGeometry(
                                         source_distance=self._registration_constants.scene_geometry.source_distance,
                                         fixed_image_offset=self._registration_data.fixed_image_offset),
                                     output_size=self._registration_data.fixed_image.size())

    def objective_function_drr(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self._registration_data.fixed_image, self.generate_drr(transformation))

    def objective_function_grangeat(self, transformation: Transformation) -> torch.Tensor:
        return -objective_function.zncc(self._registration_data.sinogram2d, self.resample_sinogram3d(transformation))

    def run(self, *, starting_transformation: Transformation, objective_function_name: str, iteration_count: int) -> \
            Tuple[OptimisationResult, float]:
        objective_function = self._objective_functions[objective_function_name]

        def obj_func(params: torch.Tensor) -> torch.Tensor:
            return objective_function(Transformation.from_vector(params))

        starting_parameters = starting_transformation.vectorised()

        n_dimensions = starting_parameters.numel()
        param_history = GrowingTensor([n_dimensions], self._particle_count * iteration_count)
        value_history = GrowingTensor([], self._particle_count * iteration_count)

        param_history.push_back(starting_parameters)
        value_history.push_back(obj_func(starting_parameters))

        def objective_pso(particle_params: np.ndarray) -> np.ndarray:
            ret = np.zeros(particle_params.shape[0])
            for i, row in enumerate(particle_params):
                params = torch.tensor(copy.deepcopy(row))
                param_history.push_back(params)
                value = obj_func(params)
                value_history.push_back(value)
                ret[i] = value.item()
            return ret

        options = {'c1': 2.525, 'c2': 1.225, 'w': 0.28}  # {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        initial_positions = np.random.normal(loc=starting_parameters.cpu().numpy(),
                                             size=(self._particle_count, n_dimensions),
                                             scale=np.array([0.1, 0.1, 0.1, 2.0, 2.0, 2.0]))
        initial_positions[0] = starting_parameters.cpu().numpy()
        optimiser = pyswarms.single.GlobalBestPSO(n_particles=self._particle_count, dimensions=n_dimensions,
                                                  init_pos=initial_positions, options=options)

        torch.cuda.synchronize()
        tic = time.time()
        cost, converged_params = optimiser.optimize(objective_pso, iters=iteration_count)
        torch.cuda.synchronize()
        toc = time.time()
        elapsed = toc - tic
        per_iteration = elapsed / float(iteration_count)
        print("Optimisation finished; took {:.4f}s for {} iterations = {:.4f}s per iteration.".format(elapsed,
                                                                                                      iteration_count,
                                                                                                      per_iteration))

        return OptimisationResult(params=torch.from_numpy(converged_params), param_history=param_history.get(),
                                  value_history=value_history.get()), per_iteration


SAVE_DIRECTORY = pathlib.Path("data/register_plot_data")
SAVE_FILE = SAVE_DIRECTORY / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl")


def save_new(new_data: plot_data.RegisterPlotData):
    if SAVE_FILE.is_file():
        logger.warn("File '{}' already exists; overwriting.".format(str(SAVE_FILE)))
    torch.save(new_data, SAVE_FILE)


def save_append(new_data: plot_data.RegisterPlotData.Dataset | list[plot_data.RegisterPlotData.Dataset]):
    if SAVE_FILE.is_file():
        current_data = torch.load(SAVE_FILE, weights_only=False)
        if not isinstance(current_data, plot_data.RegisterPlotData):
            logger.warn("Invalid data file '{}'. Renaming invalid file and saving new data to old filename.")
            current_data = []
            marked_invalid = SAVE_FILE.with_name(SAVE_FILE.stem + "_invalid.pkl")
            while marked_invalid.is_file():
                marked_invalid = marked_invalid.stem + "_1.pkl"
            SAVE_FILE.rename(marked_invalid)
    else:
        logger.warn("No save file '{}' exists to append datasets to. Saving with placeholder values.")
        current_data = plot_data.RegisterPlotData(iteration_count=-1, particle_count=-1, datasets=[])

    if isinstance(new_data, list):
        current_data.datasets.extend(new_data)
    else:
        current_data.datasets.append(new_data)

    torch.save(current_data, SAVE_FILE)


def run_benchmark(cache_directory: str, ct_path: str | pathlib.Path, xray_dicom_path: str | pathlib.Path,
                  obj_func_names: list[str], load_cached: bool = False, save_to_cache: bool = True,
                  downsample_factor: int = 1, sinogram_type: Type[sinogram.SinogramType] = sinogram.SinogramClassic,
                  plot: bool = False, iteration_count: int = 10, particle_count: int = 2000) -> list[
                                                                                                    plot_data.RegisterPlotData.Dataset] | None:
    device = torch.device("cuda")
    try:
        registration_constants = RegistrationConstants(path=ct_path, cache_directory=cache_directory,
                                                       load_cached=load_cached, regenerate_drr=True,
                                                       save_to_cache=save_to_cache, sinogram_size=None, x_ray=None,
                                                       device=device, sinogram_type=sinogram_type,
                                                       volume_downsample_factor=downsample_factor)
    except MemoryError:
        return None
    starting_params = registration_constants.transformation_ground_truth.vectorised()
    starting_params += (torch.randn(6) * torch.tensor([0.2, 0.2, 0.2, 20.0, 20.0, 50.0])).to(
        device=starting_params.device)
    starting_transformation = Transformation.from_vector(starting_params)

    task = RegistrationTask(registration_constants, particle_count=particle_count)

    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(registration_constants.image_2d_full.cpu())
        axes.axis('square')
        axes.set_title("Ground truth")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        plt.colorbar(mesh)

        _, axes = plt.subplots()
        mesh = axes.pcolormesh(task.generate_drr(starting_transformation).cpu())
        axes.axis('square')
        axes.set_title("Starting transformation")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        plt.colorbar(mesh)

    datasets = []
    for obj_func_name in obj_func_names:

        res, time_per_iteration = task.run(starting_transformation=starting_transformation,
                                           objective_function_name=obj_func_name, iteration_count=iteration_count)
        print(res, time_per_iteration)
        print("Close = {}; converged params = {}; g.t. = {}".format(torch.allclose(res.param_history[-1].cpu(),
                                                                                   registration_constants.transformation_ground_truth.vectorised().cpu()),
                                                                    res.param_history[-1].cpu(),
                                                                    registration_constants.transformation_ground_truth.vectorised().cpu()))

        if plot:
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(task.generate_drr(Transformation.from_vector(res.param_history[-1])).cpu())
            axes.axis('square')
            axes.set_title("Converged transformation")
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            plt.colorbar(mesh)
            plt.show()

        datasets.append(
            plot_data.RegisterPlotData.Dataset(fixed_image_numel=registration_constants.image_2d_full.numel(),
                                               obj_func_name=obj_func_name, sinogram_type=sinogram_type,
                                               time_per_iteration=time_per_iteration,
                                               ground_truth_transformation=registration_constants.transformation_ground_truth,
                                               starting_transformation=starting_transformation.to(
                                                   device=torch.device("cpu")),
                                               converged_transformation=Transformation.from_vector(
                                                   res.param_history[-1].cpu())))
    return datasets


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


def main(*, cache_directory: str, load_cached: bool = False, save_to_cache: bool = True, plot_first: bool = False):
    count: int = len(CT_PATHS)
    assert len(XRAY_DICOM_PATHS) == count

    if not torch.cuda.is_available():
        logger.error("CUDA not available; cannot run any useful benchmarks.")
        exit(1)

    iteration_count: int = 5
    particle_count: int = 500

    downsample_factors = [8, 4, 2, 1]
    sinogram_types = [sinogram.SinogramClassic, sinogram.SinogramHEALPix]
    obj_func_names: list[str] = ["grangeat", "drr"]
    repeats = 1

    save_new(plot_data.RegisterPlotData(iteration_count=iteration_count, particle_count=particle_count, datasets=[]))

    for sinogram_type in sinogram_types:
        first: bool = True
        for i in range(count):
            for downsample_factor in downsample_factors:
                for _ in range(repeats):
                    res = run_benchmark(cache_directory, CT_PATHS[i], XRAY_DICOM_PATHS[i], obj_func_names, load_cached,
                                        save_to_cache, downsample_factor, sinogram_type=sinogram_type,
                                        plot=plot_first and first, iteration_count=iteration_count,
                                        particle_count=particle_count)
                    if res is not None:
                        save_append(res)
                        first = False


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
    parser.add_argument("-i", "--no-load", action='store_true',
                        help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Display/plot the resulting data for the first dataset.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    main(cache_directory=args.cache_directory, load_cached=not args.no_load, save_to_cache=not args.no_save,
         plot_first=args.display)
