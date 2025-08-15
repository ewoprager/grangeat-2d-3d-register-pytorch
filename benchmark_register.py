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

from registration.interface.register import OptimisationResult, mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation


def format_time(seconds: int) -> str:
    days, rem = divmod(seconds, 60 * 60 * 24)
    hours, rem = divmod(rem, 60 * 60)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} minute{'s' if seconds != 1 else ''}")

    return ", ".join(parts)


class RegistrationInfo(NamedTuple):
    scene_geometry: SceneGeometry
    ct_volume: torch.Tensor
    fixed_image: torch.Tensor
    ct_spacing: torch.Tensor
    sinogram3ds: list[sinogram.Sinogram]
    fixed_image_spacing: torch.Tensor
    sinogram2d_grid: Sinogram2dGrid
    sinogram2d: torch.Tensor


def get_registration_info(*, cache_directory: str, ct_path: str | pathlib.Path, downsample_factor: int,
                          transformation_ground_truth: Transformation) -> RegistrationInfo | None:
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

    fixed_image_spacing, scene_geometry, fixed_image, _ = drr.generate_drr_as_target(cache_directory, ct_path,
                                                                                     ct_volume, ct_spacing,
                                                                                     save_to_cache=False, size=None,
                                                                                     transformation=transformation_ground_truth.to(
                                                                                         device=ct_volume.device))

    sinogram2d_counts = max(fixed_image.size()[0], fixed_image.size()[1])
    image_diag: float = (fixed_image_spacing.flip(dims=(0,)) * torch.tensor(fixed_image.size(),
                                                                            dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d = grangeat.calculate_fixed_image(fixed_image, source_distance=scene_geometry.source_distance,
                                                detector_spacing=fixed_image_spacing, output_grid=sinogram2d_grid)

    return RegistrationInfo(scene_geometry=scene_geometry, ct_volume=ct_volume, fixed_image=fixed_image,
                            ct_spacing=ct_spacing, sinogram3ds=sinogram3ds, fixed_image_spacing=fixed_image_spacing,
                            sinogram2d_grid=sinogram2d_grid, sinogram2d=sinogram2d)


class RegistrationTask:
    def __init__(self, registration_info: RegistrationInfo, *, particle_count: int):
        self._registration_info = registration_info
        self._particle_count = particle_count

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

    def run(self, *, starting_transformation: Transformation, objective_function_name: str, iteration_count: int) -> \
            Tuple[OptimisationResult, float]:
        objective_function = self._objective_functions[objective_function_name]

        def obj_func(params: torch.Tensor) -> torch.Tensor:
            return objective_function(mapping_parameters_to_transformation(params))

        starting_parameters = mapping_transformation_to_parameters(starting_transformation)

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
        logger.info("Optimisation finished; took {:.4f}s for {} iterations = {:.4f}s per iteration.".format(elapsed,
                                                                                                            iteration_count,
                                                                                                            per_iteration))

        return OptimisationResult(params=torch.from_numpy(converged_params), param_history=param_history.get(),
                                  value_history=value_history.get()), per_iteration


SAVE_DIRECTORY = pathlib.Path("data/register_plot_data")


def save_new(new_data: plot_data.RegisterPlotData):
    if SAVE_FILE.is_file():
        logger.warning("File '{}' already exists; overwriting.".format(str(SAVE_FILE)))
    torch.save(new_data, SAVE_FILE)


def save_append(new_data: plot_data.RegisterPlotData.Dataset | list[plot_data.RegisterPlotData.Dataset]):
    if SAVE_FILE.is_file():
        current_data = torch.load(SAVE_FILE, weights_only=False)
        if not isinstance(current_data, plot_data.RegisterPlotData):
            logger.warning("Invalid data file '{}'. Renaming invalid file and saving new data to old filename.")
            current_data = []
            marked_invalid = SAVE_FILE.with_name(SAVE_FILE.stem + "_invalid.pkl")
            while marked_invalid.is_file():
                marked_invalid = marked_invalid.stem + "_1.pkl"
            SAVE_FILE.rename(marked_invalid)
    else:
        logger.warning("No save file '{}' exists to append datasets to. Saving with placeholder values.")
        current_data = plot_data.RegisterPlotData(iteration_count=-1, particle_count=-1, datasets=[])

    if isinstance(new_data, list):
        current_data.datasets.extend(new_data)
    else:
        current_data.datasets.append(new_data)

    torch.save(current_data, SAVE_FILE)


def run_benchmark(*, cache_directory: str, ct_path: str | pathlib.Path, iteration_count, particle_count,
                  downsample_factor: int = 1, save_figures: bool = False, show_figures: bool = False) -> list[
                                                                                                             plot_data.RegisterPlotData.Dataset] | None:
    device = torch.device("cuda")

    transformation_ground_truth = Transformation.random()

    starting_params = mapping_transformation_to_parameters(transformation_ground_truth)
    starting_params += ((2.0 * torch.rand(6) - 1.0) * torch.tensor([0.38, 0.38, 0.38, 28.0, 28.0, 28.0])).to(
        device=starting_params.device)
    starting_transformation = mapping_parameters_to_transformation(starting_params)
    # if save_figures or show_figures:
    #     d = transformation_ground_truth.distance(starting_transformation)
    #     while d < 0.4 or d > 0.7:
    #         starting_params = transformation_ground_truth.vectorised()
    #         starting_params += ((2.0 * torch.rand(6) - 1.0) * torch.tensor([0.4, 0.4, 0.4, 30.0, 30.0, 30.0])).to(
    #             device=starting_params.device)
    #         starting_transformation = Transformation.from_vector(starting_params)

    try:
        registration_info = get_registration_info(cache_directory=cache_directory, ct_path=ct_path,
                                                  downsample_factor=downsample_factor,
                                                  transformation_ground_truth=transformation_ground_truth)
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise
        return None
    if registration_info is None:
        return None

    task = RegistrationTask(registration_info, particle_count=particle_count)

    if save_figures or show_figures:
        plt.imshow(registration_info.fixed_image.cpu().numpy(), cmap="Greys_r")
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_drr_at_gt.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.registration_info.sinogram2d.cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_grangeat_fixed.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.resample_sinogram3d(transformation_ground_truth, 0).cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_resampling_classic_at_gt.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.resample_sinogram3d(transformation_ground_truth, 1).cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_resampling_healpix_at_gt.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.generate_drr(starting_transformation).cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_drr_at_start.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.resample_sinogram3d(starting_transformation, 0).cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_resampling_classic_at_start.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()

        plt.imshow(task.resample_sinogram3d(starting_transformation, 1).cpu().numpy(), cmap='Greys_r')
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_figures:
            plt.savefig("data/temp/img_resampling_healpix_at_start.png", bbox_inches='tight', pad_inches=0)
        if show_figures:
            plt.show()
        else:
            plt.close()

    datasets = []
    distance_string = "Distance initialised to G.T. = {:.4f}".format(
        transformation_ground_truth.distance(starting_transformation))
    if show_figures:
        logger.info(distance_string)
    if save_figures:
        with open("data/temp/img_stats.txt", "w") as file:
            file.write(distance_string)
    for obj_func_name in ["drr", "grangeat_classic", "grangeat_healpix"]:
        res, time_per_iteration = task.run(starting_transformation=starting_transformation,
                                           objective_function_name=obj_func_name, iteration_count=iteration_count)
        converged_params = res.params.cpu().to(dtype=torch.float32)
        if save_figures or show_figures:
            # Converged DRR
            plt.imshow(task.generate_drr(mapping_parameters_to_transformation(converged_params)).cpu().numpy(),
                       cmap='Greys_r')
            plt.axis("off")
            plt.tight_layout(pad=0)
            if save_figures:
                plt.savefig("data/temp/img_drr_at_converged_{}.png".format(obj_func_name), bbox_inches='tight',
                            pad_inches=0)
            if show_figures:
                plt.show()

            if obj_func_name == "grangeat_classic":
                # Converged resampling classic
                plt.imshow(
                    task.resample_sinogram3d(mapping_parameters_to_transformation(converged_params), 0).cpu().numpy(),
                    cmap='Greys_r')
                plt.axis("off")
                plt.tight_layout(pad=0)
                if save_figures:
                    plt.savefig("data/temp/img_resampling_at_converged_{}.png".format(obj_func_name),
                                bbox_inches='tight', pad_inches=0)
                if show_figures:
                    plt.show()

            if obj_func_name == "grangeat_healpix":
                # Converged resampling healpix
                plt.imshow(
                    task.resample_sinogram3d(mapping_parameters_to_transformation(converged_params), 1).cpu().numpy(),
                    cmap='Greys_r')
                plt.axis("off")
                plt.tight_layout(pad=0)
                if save_figures:
                    plt.savefig("data/temp/img_resampling_at_converged_{}.png".format(obj_func_name),
                                bbox_inches='tight', pad_inches=0)
                if show_figures:
                    plt.show()

            distance_converged_to_gt = transformation_ground_truth.distance(
                mapping_parameters_to_transformation(converged_params))
            distance_string = "Distance converged to GT for {} = {:.4f}".format(obj_func_name, distance_converged_to_gt)
            if show_figures:
                logger.info(distance_string)
            else:
                plt.close()

            if save_figures:
                with open("data/temp/img_stats.txt", "a") as file:
                    file.write("\n{}".format(distance_string))

        datasets.append(plot_data.RegisterPlotData.Dataset(fixed_image_numel=registration_info.fixed_image.numel(),
                                                           obj_func_name=obj_func_name,
                                                           time_per_iteration=time_per_iteration,
                                                           ground_truth_transformation=transformation_ground_truth,
                                                           starting_transformation=starting_transformation.to(
                                                               device=torch.device("cpu")),
                                                           converged_transformation=mapping_parameters_to_transformation(
                                                               converged_params)))

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


def main(*, cache_directory: str, save_first: bool = False, show_first: bool = False, append_to_last: bool = False):
    count: int = len(CT_PATHS)
    assert len(XRAY_DICOM_PATHS) == count

    if not torch.cuda.is_available():
        logger.error("CUDA not available; cannot run any useful benchmarks.")
        exit(1)

    iteration_count: int = 15
    particle_count: int = 2000

    downsample_factors = [1, 2, 4, 8]  # [1, 2, 4, 8]
    data_indices = range(count)  # range(count)
    repeats = 6
    estimated_runtime = int(np.round(float(repeats * len(data_indices) * iteration_count * particle_count) * float(
        len(downsample_factors)) * 13.0 / 2000.0))
    logger.info("Estimated runtime = {}".format(format_time(estimated_runtime)))

    if not append_to_last:
        save_new(
            plot_data.RegisterPlotData(iteration_count=iteration_count, particle_count=particle_count, datasets=[]))

    first: bool = True
    for _ in range(repeats):
        for i in data_indices:
            for downsample_factor in downsample_factors:
                try:
                    res = run_benchmark(cache_directory=cache_directory, ct_path=CT_PATHS[i],
                                        downsample_factor=downsample_factor, save_figures=save_first and first,
                                        show_figures=show_first and first, iteration_count=iteration_count,
                                        particle_count=particle_count)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise
                    logger.warning("Not enough memory for run; skipping.")
                    continue
                if res is not None:
                    save_append(res)
                    first = False


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
    parser.add_argument("-f", "--save-figures", action='store_true', help="Save images for the first dataset.")
    parser.add_argument("-s", "--show-figures", action='store_true', help="Show images for the first dataset.")
    parser.add_argument("-a", "--append-to-last", action='store_true',
                        help="Append the data to the last saved data file.")
    parser.add_argument("-n", "--notify", action='store_true', help="Send notification on completion.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    if args.append_to_last:
        files = list(SAVE_DIRECTORY.glob("*.pkl"))
        SAVE_FILE = max(files, key=lambda f: f.stem)
    else:
        SAVE_FILE = SAVE_DIRECTORY / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl")

    try:
        main(cache_directory=args.cache_directory, save_first=args.save_figures, show_first=args.show_figures,
             append_to_last=args.append_to_last)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
