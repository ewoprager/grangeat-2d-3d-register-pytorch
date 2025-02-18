import copy
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import nrrd
from tqdm import tqdm
import scipy

import Extension
# from diffdrr.drr import DRR
# from diffdrr.data import read
# from sympy.solvers.solvers import det_perm
# from torchio.data.image import ScalarImage
# from diffdrr.visualization import plot_drr
# from diffdrr.pose import RigidTransform, make_matrix

from registration.common import *
import registration.grangeat as grangeat
import registration.geometry as geometry


def read_nrrd(path: str, downsample_factor=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Loading CT data file {}...".format(path))
    data, header = nrrd.read(path)
    print("Done.")
    print("Processing CT data...")
    sizes = header['sizes']
    print("\tVolume size = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    data = torch.tensor(data, device="cpu")
    image = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool3d(downsample_factor)
        image = down_sampler(image[None, :, :, :])[0]
    sizes = image.size()
    print("\tVolume size after down-sampling = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    bounds = torch.Tensor([image.min().item(), image.max().item()])
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    bounds[1] *= 10000.
    directions = torch.tensor(header['space directions'])
    spacing = float(downsample_factor) * directions.norm(dim=1)
    print("\tCT voxel spacing = [{} x {} x {}] mm".format(spacing[0], spacing[1], spacing[2]))
    print("Done.")

    return image, spacing, bounds


def zncc(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    n = xs.numel()
    assert (ys.size() == xs.size())
    n = float(n)
    sum_x = xs.sum()
    sum_y = ys.sum()
    sum_x2 = xs.square().sum()
    sum_y2 = ys.square().sum()
    sum_prod = (xs * ys).sum()
    num = n * sum_prod - sum_x * sum_y
    den = (n * sum_x2 - sum_x.square()).sqrt() * (n * sum_y2 - sum_y.square()).sqrt()
    return num / den


def evaluate(fixed_image: torch.Tensor, sinogram3d: torch.Tensor, *, transformation: Transformation,
             scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, sinogram3d_range: Sinogram3dRange,
             plot: bool = False) -> torch.Tensor:
    # resampled = grangeat.resample_slice(sinogram3d, transformation=transformation, scene_geometry=scene_geometry,
    #                                     output_grid=fixed_image_grid, input_range=sinogram3d_range)
    device = sinogram3d.device
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)
    sinogram_range_low = torch.tensor([sinogram3d_range.r.low, sinogram3d_range.theta.low, sinogram3d_range.phi.low])
    sinogram_range_high = torch.tensor(
        [sinogram3d_range.r.high, sinogram3d_range.theta.high, sinogram3d_range.phi.high])
    sinogram_spacing = (sinogram_range_high - sinogram_range_low) / (
            torch.tensor(sinogram3d.size(), dtype=torch.float32) - 1.)
    sinogram_range_centres = .5 * (sinogram_range_low + sinogram_range_high)
    resampled = Extension.resample_sinogram3d(sinogram3d, sinogram_spacing, sinogram_range_centres, ph_matrix,
                                              fixed_image_grid.phi, fixed_image_grid.r)

    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampled.clone().cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] resampled")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return zncc(fixed_image, resampled)


def evaluate_direct(fixed_image: torch.Tensor, volume_data: torch.Tensor, *, transformation: Transformation,
                    scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, voxel_spacing: torch.Tensor,
                    plot: bool = False) -> torch.Tensor:
    direct = grangeat.directly_calculate_radon_slice(volume_data, transformation=transformation,
                                                     scene_geometry=scene_geometry, output_grid=fixed_image_grid,
                                                     voxel_spacing=voxel_spacing)
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(direct.cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] calculated directly")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return zncc(fixed_image, direct)


def load_cached_volume(cache_directory: str):
    file: str = cache_directory + "/volume_spec.pt"
    try:
        volume_spec = torch.load(file)
    except:
        print("No cache file '{}' found.".format(file))
        return None
    if not isinstance(volume_spec, VolumeSpec):
        print("Cache file '{}' invalid.".format(file))
        return None
    path = volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    sinogram3d_range = volume_spec.sinogram_range
    print("Loaded cached volume spec from '{}'".format(file))
    return path, volume_downsample_factor, sinogram3d, sinogram3d_range


def calculate_volume_sinogram(cache_directory: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                              ct_volume_path: str, volume_downsample_factor: int, *, device, save_to_cache=True,
                              vol_counts=256):
    print("Calculating 3D sinogram (the volume to resample)...")

    vol_diag: float = (
            voxel_spacing * torch.tensor(volume_data.size(), dtype=torch.float32)).square().sum().sqrt().item()
    sinogram3d_range = Sinogram3dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * vol_diag, .5 * vol_diag))

    sinogram3d_grid = sinogram3d_range.generate_linear_grid(vol_counts, device=device)
    sinogram3d = grangeat.calculate_radon_volume(volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid,
                                                 samples_per_direction=vol_counts)

    if save_to_cache:
        torch.save(VolumeSpec(ct_volume_path, volume_downsample_factor, sinogram3d, sinogram3d_range),
                   cache_directory + "/volume_spec.pt")

    print("Done and saved.")

    # X, Y, Z = torch.meshgrid(  #     [torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1)])  # fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=sinogram3d.cpu().flatten(),  #                                  isomin=sinogram3d.min().item(), isomax=sinogram3d.max().item(), opacity=.2, surface_count=21))  # fig.show()

    # vol_image = ScalarImage(tensor=vol_data[None, :, :, :])  # vol_subject = read(vol_image, spacing=voxel_spacing)  # I believe that the detector array lies on the x-z plane, with x down, and z to the left (and so y outward)  # drr_generator = DRR(vol_subject,  # An object storing the CT volume, origin, and voxel spacing  #                     sdd=source_distance,  # Source-to-detector distance (i.e., focal length)  #                     height=int(torch.ceil(  #                         1.1 * voxel_spacing.mean() * torch.tensor(vol_size).max() / detector_spacing).item()),  #                     # Image height (if width is not provided, the generated DRR is square)  #                     delx=detector_spacing,  # Pixel spacing (in mm)  #                     ).to(device)  #

    return sinogram3d, sinogram3d_range


def load_cached_drr(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/drr_spec.pt"
    try:
        drr_spec = torch.load(file)
    except:
        print("No cache file '{}' found.".format(file))
        return None
    if not isinstance(drr_spec, DrrSpec):
        print("Cache file '{}' invalid.".format(file))
        return None
    if drr_spec.ct_volume_path != ct_volume_path:
        print("Cached drr '{}' is from different volume = '{}'; required volume = {}.".format(file,
                                                                                              drr_spec.ct_volume_path,
                                                                                              ct_volume_path))
        return None
    detector_spacing = drr_spec.detector_spacing
    scene_geometry = drr_spec.scene_geometry
    drr_image = drr_spec.image
    fixed_image = drr_spec.sinogram
    sinogram2d_range = drr_spec.sinogram_range
    transformation_ground_truth = drr_spec.transformation
    print("Loaded cached drr spec from '{}'".format(file))
    return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth


def generate_new_drr(cache_directory: str, ct_volume_path: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                     *, device, save_to_cache=True):
    # transformation = Transformation(torch.tensor([0., 0., 0.]),
    #                                 torch.tensor([10., 0., 0.]) + Transformation.zero().translation).to(device=device)
    # transformation = Transformation.zero(device=volume_data.device)
    transformation = Transformation.random(device=volume_data.device)
    print("Generating DRR at transformation:\n\tr = {}\n\tt = {}...".format(transformation.rotation,
                                                                            transformation.translation))

    #
    # drr_image = drr_generator(rotations, translations, parameterization="euler_angles", convention="ZXY")
    # # plot_drr(drr_image, ticks=False)
    # drr_image = drr_image[0, 0]
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_image.cpu())
    # axes.axis('square')
    # plt.colorbar(mesh)

    detector_spacing = torch.tensor([.25, .25])
    scene_geometry = SceneGeometry(source_distance=1000.)

    drr_image = geometry.generate_drr(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
                                      detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                      output_size=torch.Size([1000, 1000]), samples_per_ray=500)

    print("Done.")

    print("Calculating 2D sinogram (the fixed image)...")

    sinogram2d_counts = 1024
    image_diag: float = (
            detector_spacing * torch.tensor(drr_image.size(), dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid = sinogram2d_range.generate_linear_grid(sinogram2d_counts, device=device)

    fixed_image = grangeat.calculate_fixed_image(drr_image, source_distance=scene_geometry.source_distance,
                                                 detector_spacing=detector_spacing, output_grid=sinogram2d_grid)

    if save_to_cache:
        torch.save(DrrSpec(ct_volume_path, detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range,
                           transformation), cache_directory + "/drr_spec.pt")

    print("Done and saved.")

    return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation


def register(path: str | None, *, cache_directory: str, load_cached: bool = True, regenerate_drr: bool = False,
             save_to_cache=True):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

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
    # print(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])
    # grid = torch.tensor([[.9, -.9]])
    # print(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])
    # grid = torch.tensor([[-.9, .9]])
    # print(torch.nn.functional.grid_sample(cal_image[None, None, :, :], grid[None, None, :, :])[0, 0])

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

    volume_spec = None
    sinogram3d = None
    sinogram3d_range = None
    if load_cached:
        volume_spec = load_cached_volume(cache_directory)

    if volume_spec is None:
        volume_downsample_factor: int = 4
    else:
        path, volume_downsample_factor, sinogram3d, sinogram3d_range = volume_spec

    if path is None:
        save_to_cache = False
        vol_data = torch.zeros((3, 3, 3), device=device)
        vol_data[1, 1, 1] = 1.
        voxel_spacing = torch.tensor([10., 10., 10.])
    else:
        vol_data, voxel_spacing, bounds = read_nrrd(path, downsample_factor=volume_downsample_factor)
        vol_data = vol_data.to(device=device, dtype=torch.float32)

    if sinogram3d is None or sinogram3d_range is None:
        sinogram3d, sinogram3d_range = calculate_volume_sinogram(cache_directory, vol_data, voxel_spacing, path,
                                                                 volume_downsample_factor, device=device,
                                                                 save_to_cache=save_to_cache, vol_counts=64)

    voxel_spacing = voxel_spacing.to(device=device)

    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, path)

    if drr_spec is None:
        drr_spec = generate_new_drr(cache_directory, path, vol_data, voxel_spacing, device=device,
                                    save_to_cache=save_to_cache)

    detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth = drr_spec

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(drr_image.cpu())
    axes.axis('square')
    axes.set_title("g")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.colorbar(mesh)

    # nrrd.write("/home/eprager/Desktop/projection_image.nrrd", drr_image.cpu().unsqueeze(0).numpy())

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(fixed_image.cpu())
    axes.axis('square')
    axes.set_title("d/ds R2 [g^tilde]")
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)

    sinogram2d_grid = sinogram2d_range.generate_linear_grid(fixed_image.size(), device=device)

    print("{:.4e}".format(
        evaluate(fixed_image, sinogram3d, transformation=transformation_ground_truth.to(device=device),
                 scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid, sinogram3d_range=sinogram3d_range,
                 plot=True)  # evaluate_direct(fixed_image, vol_data, transformation=transformation_ground_truth,
        #                 scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid, voxel_spacing=voxel_spacing,
        #                 plot=True)
    ))

    if False:
        n = 100
        angle0s = torch.linspace(transformation_ground_truth.rotation[0] - torch.pi,
                                 transformation_ground_truth.rotation[0] + torch.pi, n)
        angle1s = torch.linspace(transformation_ground_truth.rotation[1] - torch.pi,
                                 transformation_ground_truth.rotation[1] + torch.pi, n)
        nznccs = torch.zeros((n, n))
        for i in tqdm(range(nznccs.numel())):
            i0 = i % n
            i1 = i // n
            nznccs[i1, i0] = -evaluate(fixed_image, sinogram3d, transformation=Transformation(
                torch.tensor([angle0s[i0], angle1s[i1], transformation_ground_truth.rotation[2]], device=device),
                transformation_ground_truth.translation), scene_geometry=scene_geometry,
                                       fixed_image_grid=sinogram2d_grid, sinogram3d_range=sinogram3d_range)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(nznccs)
        axes.set_title("landscape over two angle components")
        axes.set_xlabel("x-component of rotation vector")
        axes.set_ylabel("y-component of rotation vector")
        axes.axis('square')
        plt.colorbar(mesh)

    if False:
        n = 1000
        nznccs = torch.zeros(n)
        distances = torch.zeros(n)
        for i in tqdm(range(n)):
            transformation = Transformation.random(device=device)
            distances[i] = transformation_ground_truth.distance(transformation)
            nznccs[i] = -evaluate(fixed_image, sinogram3d, transformation=transformation, scene_geometry=scene_geometry,
                                  fixed_image_grid=sinogram2d_grid, sinogram3d_range=sinogram3d_range)

        _, axes = plt.subplots()
        axes.scatter(distances, nznccs)
        axes.set_xlabel("distance in SE3")
        axes.set_ylabel("-ZNCC")
        axes.set_title("Relationship between similarity measure and distance in SE3")

    if False:
        def objective(params: torch.Tensor) -> torch.Tensor:
            return -evaluate(fixed_image, sinogram3d,
                             transformation=Transformation(params[0:3], params[3:6]).to(device=device),
                             scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                             sinogram3d_range=sinogram3d_range)

        print("Optimising...")
        param_history = []
        value_history = []
        transformation_start = Transformation.random()
        start_params: torch.Tensor = transformation_start.vectorised()

        converged_params: torch.Tensor | None = None
        if False:
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
            print("Done. Took {:.3f}s.".format(toc - tic))
            print("Final value = {:.3f} at params = {}".format(value_history[-1], param_history[-1]))
            converged_params = param_history[-1]
        else:
            def objective_scipy(params: np.ndarray) -> float:
                params = torch.tensor(copy.deepcopy(params))
                param_history.append(params)
                value = objective(params)
                value_history.append(value)
                return value.item()

            tic = time.time()
            res = scipy.optimize.minimize(objective_scipy, start_params.numpy(), method='Nelder-Mead')
            toc = time.time()
            print("Done. Took {:.3f}s.".format(toc - tic))
            print(res)
            converged_params = torch.from_numpy(res.x)

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

        param_history = torch.stack(param_history, dim=0)
        value_history = torch.tensor(value_history)

        its = np.arange(param_history.size()[0])
        its2 = np.array([its[0], its[-1]])

        # rotations
        _, axes = plt.subplots()
        axes.plot(its2, np.full(2, 0.), ls='dashed')
        axes.plot(its, param_history[:, 0] - transformation_ground_truth.rotation[0].item(), label="r0 - r0*")
        axes.plot(its, param_history[:, 1] - transformation_ground_truth.rotation[1].item(), label="r1 - r1*")
        axes.plot(its, param_history[:, 2] - transformation_ground_truth.rotation[2].item(), label="r2 - r2*")
        axes.legend()
        axes.set_xlabel("iteration")
        axes.set_ylabel("param value [rad]")
        axes.set_title("rotation parameter values over optimisation iterations")
        # translations
        _, axes = plt.subplots()
        axes.plot(its2, np.full(2, 0.), ls='dashed')
        axes.plot(its, param_history[:, 3] - transformation_ground_truth.translation[0].item(), label="t0 - t0*")
        axes.plot(its, param_history[:, 4] - transformation_ground_truth.translation[1].item(), label="t1 - t1*")
        axes.plot(its, param_history[:, 5] - transformation_ground_truth.translation[2].item(), label="t2 - t2*")
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
