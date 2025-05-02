import copy
from typing import Tuple
import time
from enum import Enum
import os
import argparse
import logging.config

import matplotlib.pyplot as plt
import numpy as np
import torch
import nrrd
import scipy
from tqdm import tqdm

import Extension

# from diffdrr.drr import DRR
# from diffdrr.data import read
# from sympy.solvers.solvers import det_perm
# from torchio.data.image import ScalarImage
# from diffdrr.visualization import plot_drr
# from diffdrr.pose import RigidTransform, make_matrix

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration import drr
from registration.lib import geometry
from registration import data
from registration import pre_computed
from registration import objective_function
from registration import script
import registration.lib.plot as myplt


class SinogramStructure(Enum):
    CLASSIC = 1
    FIBONACCI = 2


def main(*, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool, save_to_cache: bool,
         sinogram_size: int, sinogram_structure: SinogramStructure = SinogramStructure.CLASSIC,
         x_ray: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))
    # device = "cpu"

    # cal_image = torch.zeros((10, 10))
    # cal_image[0, 0] = 1.
    # cal_image[-1, 0] = .5
    # cal_image[0, -1] = .2
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(cal_image)
    # axes.axis('square')
    # plt.colorbar(mesh)
    #
    # grid = torch.tensor([[-.9, -.9]])
    # logger.info(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])
    # grid = torch.tensor([[.9, -.9]])
    # logger.info(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])
    # grid = torch.tensor([[-.9, .9]])
    # logger.info(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])

    # vol_size = torch.Size([10, 10, 10])

    # vol_data = torch.rand(vol_size, device=device)

    # vol_data = torch.zeros(vol_size, device=device)
    # vol_data[:, 0, 0] = torch.linspace(0., 1., vol_data.size()[0])
    # vol_data[0, 0, :] = torch.linspace(0., 1., vol_data.size()[2])
    # vol_data[0, :, 0] = torch.linspace(1., 0., vol_data.size()[1])
    # vol_data[-1, 0, 0] = 0.5
    # vol_data[-1, -1, -1] = 0.2
    # vol_data[0, 0, 0] = 1.
    # voxel_spacing = torch.Tensor([10., 10., 10.])  # [mm]

    # Load the volume and get its sinogram
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
        drr_image = drr_image[int(float(drr_image.size()[0]) * .5 * (1. - f_middle)):int(float(drr_image.size()[0]) * .5 * (1. + f_middle)), :]

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

    logger.info("Plotting DRR/X-ray and fixed image...")
    # Plotting DRR/X-ray
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(drr_image.cpu())
    axes.axis('square')
    axes.set_title("g")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.colorbar(mesh)

    # nrrd.write("/home/eprager/Desktop/projection_image.nrrd", drr_image.cpu().unsqueeze(0).numpy())

    # Plotting fixed image (d/ds R_2 [g^tilde])
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(fixed_image.cpu())
    axes.axis('square')
    axes.set_title("d/ds R2 [g^tilde]")
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)
    # Getting the limits to use for other colour mesh plots:
    colour_limits: Tuple[float, float] = mesh.get_clim()
    logger.info("DRR/X-ray and fixed image plotted.")

    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, fixed_image.size(), device=device)

    tr = transformation_ground_truth if x_ray is None else Transformation.random()
    logger.info("Evaluating at ground truth..." if x_ray is None else "Evaluating at random transformation")
    zncc, resampled = objective_function.evaluate(fixed_image, sinogram3d, transformation=tr.to(device=device),
                                                  scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                                                  plot=colour_limits)
    logger.info("Evaluation: -ZNCC = -{:.4e}".format(zncc.item()  # evaluate_direct(fixed_image, vol_data,
                                                     # transformation=tr,
                                                     #                 scene_geometry=scene_geometry,
                                                     #                 fixed_image_grid=sinogram2d_grid,
                                                     #                 voxel_spacing=voxel_spacing,
                                                     #                 plot=True)
                                                     ))

    # plt.show()
    # low = torch.max(fixed_image.min(), resampled.min())
    # high = torch.min(fixed_image.max(), resampled.max())
    # overlaid = torch.stack((((fixed_image - low) / (high - low)).cpu(), ((resampled - low) / (high - low)).cpu(),
    #                         torch.zeros_like(fixed_image, device='cpu')), dim=-1)
    # plt.imshow(overlaid)

    # Landscape of similarity metric over two transformation parameters
    if False:
        n = 80
        angle0s = torch.linspace(transformation_ground_truth.rotation[0] - torch.pi,
                                 transformation_ground_truth.rotation[0] + torch.pi, n)
        angle1s = torch.linspace(transformation_ground_truth.rotation[1] - torch.pi,
                                 transformation_ground_truth.rotation[1] + torch.pi, n)
        nznccs = torch.zeros((n, n))
        for i in tqdm(range(nznccs.numel())):
            i0 = i % n
            i1 = i // n
            nznccs[i1, i0] = -objective_function.evaluate(fixed_image, sinogram3d, transformation=Transformation(
                torch.tensor([angle0s[i0], angle1s[i1], transformation_ground_truth.rotation[2]], device=device),
                transformation_ground_truth.translation), scene_geometry=scene_geometry,
                                                          fixed_image_grid=sinogram2d_grid)[0]
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(nznccs)
        axes.set_title("landscape over two angle components")
        axes.set_xlabel("x-component of rotation vector")
        axes.set_ylabel("y-component of rotation vector")
        axes.axis('square')
        plt.colorbar(mesh)
        plt.savefig("data/temp/landscape_no_sample_smoothing.pgf")

        # Landscape with sample smoothing
        if False:
            nznccs = torch.zeros((n, n))
            for i in tqdm(range(nznccs.numel())):
                i0 = i % n
                i1 = i // n
                nznccs[i1, i0] = -objective_function.evaluate(fixed_image, sinogram3d, transformation=Transformation(
                    torch.tensor([angle0s[i0], angle1s[i1], transformation_ground_truth.rotation[2]], device=device),
                    transformation_ground_truth.translation), scene_geometry=scene_geometry,
                                                              fixed_image_grid=sinogram2d_grid, smooth=True)[0]
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(nznccs)
            axes.set_title("landscape over two angle components, with sample smoothing")
            axes.set_xlabel("x-component of rotation vector")
            axes.set_ylabel("y-component of rotation vector")
            axes.axis('square')
            plt.colorbar(mesh)
            plt.savefig("data/temp/landscape_with_sample_smoothing.pgf")

    # Scatter of similarity vs. distance in SE3
    if False:
        n = 1000
        nznccs = torch.zeros(n)
        distances = torch.zeros(n)
        for i in tqdm(range(n)):
            transformation = Transformation.random(device=device)
            distances[i] = transformation_ground_truth.distance(transformation)
            nznccs[i] = -objective_function.evaluate(fixed_image, sinogram3d, transformation=transformation,
                                                     scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid)[0]

        _, axes = plt.subplots()
        axes.scatter(distances, nznccs)
        axes.set_xlabel("distance in SE3")
        axes.set_ylabel("-ZNCC")
        axes.set_title("Relationship between similarity measure and distance in SE3")

    # Registration
    if True:
        def objective(params: torch.Tensor) -> torch.Tensor:
            return -objective_function.evaluate(fixed_image, sinogram3d,
                                                transformation=Transformation(params[0:3], params[3:6]).to(
                                                    device=device), scene_geometry=scene_geometry,
                                                fixed_image_grid=sinogram2d_grid)[0]

        logger.info("Optimising...")
        param_history = []
        value_history = []
        transformation_start = Transformation(-.75 * torch.pi * torch.tensor([1., -1., -1]), torch.tensor([0., 0., 80.]))
        start_params: torch.Tensor = transformation_start.vectorised()

        initial_image = geometry.generate_drr(vol_data, transformation=transformation_start.to(device=device),
                                              voxel_spacing=voxel_spacing, detector_spacing=detector_spacing,
                                              scene_geometry=scene_geometry, output_size=drr_image.size(),
                                              samples_per_ray=512)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(initial_image.cpu())
        axes.axis('square')
        axes.set_title("DRR at initial transformation")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        plt.colorbar(mesh)
        plt.show()
        del initial_image

        converged_params: torch.Tensor | None = None
        if False:  # Using a PyTorch optimiser
            n = 1000
            iterated_params = start_params.clone().to(device=sinogram3d.device)
            iterated_params.requires_grad_(True)
            torch.autograd.set_detect_anomaly(True)
            optimiser = torch.optim.SGD([iterated_params], lr=0.01)
            optimiser.zero_grad()
            tic = time.time()

            def closure():
                param_history.append(iterated_params.clone().cpu())
                value = objective(iterated_params)
                value.backward(torch.zeros_like(value))
                value_history.append(value.clone().cpu())
                return value

            for i in range(n):
                iterated_params = optimiser.step(closure)

            toc = time.time()
            logger.info("Done. Took {:.3f}s.".format(toc - tic))
            logger.info("Final value = {:.3f} at params = {}".format(value_history[-1], param_history[-1]))
            converged_params = param_history[-1]
        else:  # Using a scipy optimiser
            def objective_scipy(params: np.ndarray) -> float:
                params = torch.tensor(copy.deepcopy(params))
                param_history.append(params)
                value = objective(params)
                value_history.append(value)
                return value.item()

            tic = time.time()
            # res = scipy.optimize.minimize(objective_scipy, start_params.numpy(), method='Powell')
            res = scipy.optimize.basinhopping(objective_scipy, start_params.numpy(), T=5.0,
                                              minimizer_kwargs={"method": 'Nelder-Mead'})
            toc = time.time()
            logger.info("Done. Took {:.3f}s.".format(toc - tic))
            logger.info(res)
            converged_params = torch.from_numpy(res.x)

        objective_function.evaluate(fixed_image, sinogram3d,
                                    transformation=Transformation(converged_params[0:3], converged_params[3:6]).to(
                                        device=device), scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                                    plot=colour_limits)

        final_image = geometry.generate_drr(vol_data, transformation=Transformation(
            torch.tensor(converged_params[0:3], device=device), torch.tensor(converged_params[3:6], device=device)),
                                            voxel_spacing=voxel_spacing, detector_spacing=detector_spacing,
                                            scene_geometry=scene_geometry, output_size=drr_image.size(),
                                            samples_per_ray=512)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(final_image.cpu())
        axes.axis('square')
        axes.set_title("DRR at final transformation")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        plt.colorbar(mesh)

        del final_image

        param_history = torch.stack(param_history, dim=0)
        value_history = torch.tensor(value_history)

        its = np.arange(param_history.size()[0])
        its2 = np.array([its[0], its[-1]])

        if x_ray is None:
            param_history[:, 0:3] -= transformation_ground_truth.rotation
            param_history[:, 3:6] -= transformation_ground_truth.translation

        # rotations
        _, axes = plt.subplots()
        axes.plot(its2, np.full(2, 0.), ls='dashed')
        axes.plot(its, param_history[:, 0], label="r0 - r0*" if x_ray is None else "r0")
        axes.plot(its, param_history[:, 1], label="r1 - r1*" if x_ray is None else "r1")
        axes.plot(its, param_history[:, 2], label="r2 - r2*" if x_ray is None else "r2")
        axes.legend()
        axes.set_xlabel("iteration")
        axes.set_ylabel("param value [rad]")
        axes.set_title("rotation parameter values over optimisation iterations")
        # translations
        _, axes = plt.subplots()
        axes.plot(its2, np.full(2, 0.), ls='dashed')
        axes.plot(its, param_history[:, 3], label="t0 - t0*" if x_ray is None else "t0")
        axes.plot(its, param_history[:, 4], label="t1 - t1*" if x_ray is None else "t1")
        axes.plot(its, param_history[:, 5], label="t2 - t2*" if x_ray is None else "t2")
        axes.legend()
        axes.set_xlabel("iteration")
        axes.set_ylabel("param value [mm]")
        axes.set_title("translation parameter values over optimisation iterations")

        _, axes = plt.subplots()
        axes.plot(value_history)
        axes.set_xlabel("iteration")
        axes.set_ylabel("-zncc")
        axes.set_title("loss over optimisation iterations")

    plt.show()


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

    main(path=args.ct_nrrd_path, cache_directory=args.cache_directory, load_cached=not args.no_load,
         regenerate_drr=args.regenerate_drr, save_to_cache=not args.no_save, sinogram_size=args.sinogram_size,
         sinogram_structure=SinogramStructure.CLASSIC, x_ray=args.x_ray if "x_ray" in vars(args) else None)
