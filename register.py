import copy
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as pgo
import time
import torch
import nrrd
from scipy.spatial.transform import Rotation
import kornia
from torch.nn.parameter import Parameter
from tqdm import tqdm
from typing import NamedTuple
import scipy

from diffdrr.drr import DRR
from diffdrr.data import read
from sympy.solvers.solvers import det_perm
from torchio.data.image import ScalarImage
from diffdrr.visualization import plot_drr
from diffdrr.pose import RigidTransform, make_matrix

import Extension as ExtensionTest


class LinearMapping:
    def __init__(self, a: float | torch.Tensor, b: float | torch.Tensor):
        self.a = a
        self.b = b

    def __call__(self, x: float | torch.Tensor) -> float | torch.Tensor:
        return self.a + self.b * x


class LinearRange:
    def __init__(self, low: float, high: float, ):
        self.low = low
        self.high = high

    def generate_range(self, count: int, *, device) -> torch.Tensor:
        return torch.linspace(self.low, self.high, count, device=device)

    def get_mapping_from(self, other: 'LinearRange') -> LinearMapping:
        frac: float = (self.high - self.low) / (other.high - other.low)
        return LinearMapping(self.low - frac * other.low, frac)


class Transformation(NamedTuple):
    rotation: torch.Tensor
    translation: torch.Tensor

    def inverse(self) -> 'Transformation':
        return Transformation(-self.rotation, -self.translation)


class SceneGeometry(NamedTuple):
    source_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array
    ct_origin_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array)


class Sinogram2dGrid(NamedTuple):
    phi: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.r.device


class Sinogram3dGrid(NamedTuple):
    phi: torch.Tensor
    theta: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.theta.device and self.theta.device == self.r.device


class Sinogram2dRange(NamedTuple):
    phi: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts)
        return Sinogram2dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[1], device=device))


class Sinogram3dRange(NamedTuple):
    phi: LinearRange
    theta: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        return Sinogram3dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.theta.low, self.theta.high, counts[1], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[2], device=device))


class VolumeSpec(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: torch.Tensor
    sinogram_range: Sinogram3dRange


class DrrSpec(NamedTuple):
    ct_volume_path: str
    detector_spacing: torch.Tensor  # [mm] distances between the detectors: (vertical, horizontal)
    scene_geometry: SceneGeometry
    image: torch.Tensor
    sinogram: torch.Tensor
    sinogram_range: Sinogram2dRange
    transformation: Transformation


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


def calculate_radon_volume(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, samples_per_direction: int = 128,
                           output_grid: Sinogram3dGrid):
    assert (output_grid.device_consistent())
    assert (volume_data.device == output_grid.phi.device)
    return ExtensionTest.dRadon3dDR(volume_data, voxel_spacing[2].item(), voxel_spacing[1].item(),
                                    voxel_spacing[0].item(), output_grid.phi.to(device=volume_data.device),
                                    output_grid.theta.to(device=volume_data.device),
                                    output_grid.r.to(device=volume_data.device), samples_per_direction)


def transform(positions_cartesian: torch.Tensor, transformation: Transformation,
              exclude_translation: bool = False) -> torch.Tensor:
    r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(transformation.rotation[None, :])[0].to(
        device=positions_cartesian.device, dtype=torch.float32)
    positions_cartesian = torch.einsum('kl,...l->...k', r, positions_cartesian.to(dtype=torch.float32))
    if not exclude_translation:
        positions_cartesian = positions_cartesian + transformation.translation.to(device=positions_cartesian.device,
                                                                                  dtype=torch.float32)
    return positions_cartesian


def generate_drr(volume_data: torch.Tensor, *, transformation: Transformation, voxel_spacing: torch.Tensor,
                 detector_spacing: torch.Tensor, scene_geometry: SceneGeometry, output_size: torch.Size,
                 samples_per_ray: int = 64) -> torch.Tensor:
    img_width: int = output_size[1]
    img_height: int = output_size[0]
    source_position: torch.Tensor = torch.tensor([0., 0., scene_geometry.source_distance])
    detector_xs: torch.Tensor = detector_spacing[1] * (
            torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    detector_ys: torch.Tensor = detector_spacing[0] * (
            torch.arange(img_height, 0, -1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    detector_ys, detector_xs = torch.meshgrid(detector_ys, detector_xs)
    directions: torch.Tensor = torch.nn.functional.normalize(
        torch.stack((detector_xs, detector_ys, torch.zeros_like(detector_xs)), dim=-1) - source_position, dim=-1)
    volume_depth: float = volume_data.size()[2] * voxel_spacing[2].item()
    volume_diag: torch.Tensor = (torch.tensor(volume_data.size(), dtype=torch.float32) * voxel_spacing).flip(dims=(0,))
    lambda_start: float = scene_geometry.source_distance - scene_geometry.ct_origin_distance - .5 * volume_depth
    lambda_end: float = torch.linalg.vector_norm(torch.tensor(
        [0., 0., scene_geometry.ct_origin_distance - scene_geometry.source_distance]) - .5 * volume_diag).item()
    step_size: float = (lambda_end - lambda_start) / float(samples_per_ray)
    deltas = directions * step_size
    deltas = transform(deltas, transformation, exclude_translation=True)
    starts = source_position + lambda_start * directions - torch.tensor([0., 0., scene_geometry.ct_origin_distance])
    starts = transform(starts, transformation)

    deltas_texture = (2. * deltas / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    grid = (2. * starts / volume_diag).to(device=volume_data.device, dtype=torch.float32)
    ret = torch.zeros(output_size, device=volume_data.device)
    for i in range(samples_per_ray):
        ret += torch.nn.functional.grid_sample(volume_data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]
        grid += deltas_texture
    return step_size * ret


def calculate_fixed_image(drr_image: torch.Tensor, *, source_distance: float, detector_spacing: torch.Tensor,
                          output_grid: Sinogram2dGrid, samples_per_line: int = 128) -> torch.Tensor:
    assert (output_grid.device_consistent())
    assert (output_grid.phi.device == drr_image.device)

    img_width = drr_image.size()[1]
    img_height = drr_image.size()[0]

    xs = detector_spacing[1] * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing[0] * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    ys, xs = torch.meshgrid(ys, xs)
    cos_gamma = source_distance / torch.sqrt(xs.square() + ys.square() + source_distance * source_distance)
    g_tilde = cos_gamma.to('cuda') * drr_image

    fixed_scaling = (output_grid.r / source_distance).square() + 1.

    ##
    # no_derivative = ExtensionTest.radon2d(g_tilde, detector_spacing, detector_spacing, phi_values, r_values,
    #                                       samples_per_line)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(no_derivative.cpu())
    # axes.axis('square')
    # axes.set_title("R2 [g^tilde]")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # post_derivative = no_derivative.diff(dim=-1) / torch.abs(r_values[1] - r_values[0])
    # post_derivative[:, :(post_derivative.size()[1] // 2)] *= -1.
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(post_derivative.cpu())
    # axes.axis('square')
    # axes.set_title("diff/ds R2 [g^tilde]")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    ##

    return fixed_scaling * ExtensionTest.dRadon2dDR(g_tilde, detector_spacing[1].item(), detector_spacing[0].item(),
                                                    output_grid.phi, output_grid.r, samples_per_line)


def fixed_polar_to_moving_cartesian(input_grid: Sinogram2dGrid, *, scene_geometry: SceneGeometry) -> torch.Tensor:
    hypotenuses = torch.sqrt(input_grid.r.square() + scene_geometry.source_distance * scene_geometry.source_distance)
    sin_alphas = input_grid.r / hypotenuses
    zs = sin_alphas.square() * (scene_geometry.source_distance - scene_geometry.ct_origin_distance)
    scaled_rs = ((
                         scene_geometry.source_distance - scene_geometry.ct_origin_distance - zs) * input_grid.r / scene_geometry.source_distance)
    xs = scaled_rs * torch.cos(input_grid.phi)
    ys = scaled_rs * torch.sin(input_grid.phi)
    return torch.stack((xs, ys, zs), dim=-1)


def moving_cartesian_to_moving_spherical(positions_cartesian: torch.Tensor) -> Sinogram3dGrid:
    xs = positions_cartesian[..., 0]
    ys = positions_cartesian[..., 1]
    zs = positions_cartesian[..., 2]
    phis = torch.atan2(ys, xs)
    phis_over = phis > .5 * torch.pi
    phis_under = phis < -.5 * torch.pi
    phis[phis_over] -= torch.pi
    phis[phis_under] += torch.pi
    thetas = torch.atan2(zs, torch.sqrt(xs.square() + ys.square()))
    rs = torch.sqrt(xs.square() + ys.square() + zs.square())
    rs[torch.logical_or(phis_over, phis_under)] *= -1.
    return Sinogram3dGrid(phis, thetas, rs)


def resample_slice(sinogram3d: torch.Tensor, *, input_range: Sinogram3dRange, transformation: Transformation,
                   scene_geometry: SceneGeometry, output_grid: Sinogram2dGrid):
    assert (output_grid.device_consistent())
    assert (output_grid.phi.device == sinogram3d.device)

    output_grid_2d = Sinogram2dGrid(*torch.meshgrid(output_grid.phi, output_grid.r))

    output_grid_cartesian_2d = fixed_polar_to_moving_cartesian(output_grid_2d, scene_geometry=scene_geometry)

    output_grid_cartesian_2d = transform(output_grid_cartesian_2d, transformation)

    output_grid_sph_2d = moving_cartesian_to_moving_spherical(output_grid_cartesian_2d)

    ##
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(phi_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("phi_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(theta_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("theta_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(r_values_sph.cpu())
    # axes.axis('square')
    # axes.set_title("r_sph resampling values")
    # axes.set_xlabel("r_pol")
    # axes.set_ylabel("phi_pol")
    # plt.colorbar(mesh)
    ##

    grid_range = LinearRange(-1., 1.)

    i_mapping: LinearMapping = grid_range.get_mapping_from(input_range.r)
    j_mapping: LinearMapping = grid_range.get_mapping_from(input_range.theta)
    k_mapping: LinearMapping = grid_range.get_mapping_from(input_range.phi)

    grid = torch.stack(
        (i_mapping(output_grid_sph_2d.r), j_mapping(output_grid_sph_2d.theta), k_mapping(output_grid_sph_2d.phi)),
        dim=-1)
    return torch.nn.functional.grid_sample(sinogram3d[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]


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
    resampled = resample_slice(sinogram3d, transformation=transformation, scene_geometry=scene_geometry,
                               output_grid=fixed_image_grid, input_range=sinogram3d_range)
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampled.cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] resampled")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return zncc(fixed_image, resampled)


def load_cached_volume(cache_directory: str):
    print("Loading cached volume spec...")
    volume_spec = torch.load(cache_directory + "/volume_spec.pt")
    assert (isinstance(volume_spec, VolumeSpec))
    path = volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    sinogram3d_range = volume_spec.sinogram_range
    print("Done.")
    return path, volume_downsample_factor, sinogram3d, sinogram3d_range


def calculate_volume_sinogram(cache_directory: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                              ct_volume_path: str, volume_downsample_factor: int, *, device):
    print("Calculating 3D sinogram (the volume to resample)...")

    vol_diag: float = (
            voxel_spacing * torch.tensor(volume_data.size(), dtype=torch.float32)).square().sum().sqrt().item()
    sinogram3d_range = Sinogram3dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * vol_diag, .5 * vol_diag))

    vol_counts = 256
    sinogram3d_grid = sinogram3d_range.generate_linear_grid(vol_counts, device=device)
    sinogram3d = calculate_radon_volume(volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid,
                                        samples_per_direction=vol_counts)

    torch.save(VolumeSpec(ct_volume_path, volume_downsample_factor, sinogram3d, sinogram3d_range),
               cache_directory + "/volume_spec.pt")

    print("Done and saved.")

    # X, Y, Z = torch.meshgrid(  #     [torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1)])  # fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=sinogram3d.cpu().flatten(),  #                                  isomin=sinogram3d.min().item(), isomax=sinogram3d.max().item(), opacity=.2, surface_count=21))  # fig.show()

    # vol_image = ScalarImage(tensor=vol_data[None, :, :, :])  # vol_subject = read(vol_image, spacing=voxel_spacing)  # I believe that the detector array lies on the x-z plane, with x down, and z to the left (and so y outward)  # drr_generator = DRR(vol_subject,  # An object storing the CT volume, origin, and voxel spacing  #                     sdd=source_distance,  # Source-to-detector distance (i.e., focal length)  #                     height=int(torch.ceil(  #                         1.1 * voxel_spacing.mean() * torch.tensor(vol_size).max() / detector_spacing).item()),  #                     # Image height (if width is not provided, the generated DRR is square)  #                     delx=detector_spacing,  # Pixel spacing (in mm)  #                     ).to(device)  #

    return sinogram3d, sinogram3d_range


def load_cached_drr(cache_directory: str, ct_volume_path: str):
    print("Loading cached drr spec...")
    drr_spec = torch.load(cache_directory + "/drr_spec.pt")
    assert (isinstance(drr_spec, DrrSpec))
    if drr_spec.ct_volume_path == ct_volume_path:
        detector_spacing = drr_spec.detector_spacing
        scene_geometry = drr_spec.scene_geometry
        drr_image = drr_spec.image
        fixed_image = drr_spec.sinogram
        sinogram2d_range = drr_spec.sinogram_range
        transformation_ground_truth = drr_spec.transformation
        print("Done.")
        return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth
    else:
        print("Cached drr is from different volume, so generating a new one.")
        return None


def generate_new_drr(cache_directory: str, ct_volume_path: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                     *, device):
    rotation = torch.rand(3)
    translation = torch.rand(3)
    print("Generating DRR at transformation:\n\tr = {}\n\tt = {}...".format(rotation, translation))
    transformation = Transformation(rotation, translation)

    #
    # drr_image = drr_generator(rotations, translations, parameterization="euler_angles", convention="ZXY")
    # # plot_drr(drr_image, ticks=False)
    # drr_image = drr_image[0, 0]
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_image.cpu())
    # axes.axis('square')
    # plt.colorbar(mesh)

    detector_spacing = torch.tensor([.25, .25])
    scene_geometry = SceneGeometry(source_distance=1000., ct_origin_distance=100.)

    drr_image = generate_drr(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
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

    fixed_image = calculate_fixed_image(drr_image, source_distance=scene_geometry.source_distance,
                                        detector_spacing=detector_spacing, output_grid=sinogram2d_grid)

    torch.save(DrrSpec(ct_volume_path, detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range,
                       transformation), cache_directory + "/drr_spec.pt")

    print("Done and saved.")

    return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation


def register(path: str, *, cache_directory: str, load_cached: bool = True, regenerate_drr: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    sinogram3d = None
    sinogram3d_range = None
    if load_cached:
        path, volume_downsample_factor, sinogram3d, sinogram3d_range = load_cached_volume(cache_directory)
    else:
        volume_downsample_factor: int = 4

    vol_data, voxel_spacing, bounds = read_nrrd(path, downsample_factor=volume_downsample_factor)
    vol_data = vol_data.to(device=device, dtype=torch.float32)

    if sinogram3d is None or sinogram3d_range is None:
        sinogram3d, sinogram3d_range = calculate_volume_sinogram(cache_directory, vol_data, voxel_spacing, path,
                                                                 volume_downsample_factor, device=device)

    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, path)

    if drr_spec is None:
        drr_spec = generate_new_drr(cache_directory, path, vol_data, voxel_spacing, device=device)

    detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth = drr_spec

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(drr_image.cpu())
    axes.axis('square')
    axes.set_title("g")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.colorbar(mesh)

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(fixed_image.cpu())
    axes.axis('square')
    axes.set_title("d/ds R2 [g^tilde]")
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)

    sinogram2d_grid = sinogram2d_range.generate_linear_grid(fixed_image.size(), device=device)

    print("{:.4e}".format(
        evaluate(fixed_image, sinogram3d, transformation=transformation_ground_truth, scene_geometry=scene_geometry,
                 fixed_image_grid=sinogram2d_grid, sinogram3d_range=sinogram3d_range, plot=True)))

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
                torch.tensor([angle0s[i0], angle1s[i1], transformation_ground_truth.rotation[2]]),
                transformation_ground_truth.translation), scene_geometry=scene_geometry,
                                       fixed_image_grid=sinogram2d_grid, sinogram3d_range=sinogram3d_range)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(nznccs)
        axes.set_title("landscape over angle about x axis")
        axes.set_xlabel("angle0")
        axes.set_ylabel("angle1")
        axes.axis('square')
        plt.colorbar(mesh)

    if True:
        def objective(params: torch.Tensor) -> torch.Tensor:
            return -evaluate(fixed_image, sinogram3d, transformation=Transformation(params[0:3], params[3:6]),
                             scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                             sinogram3d_range=sinogram3d_range)

        print("Optimising...")
        n = 1000
        param_history = []
        value_history = []
        params_: torch.Tensor = torch.rand(6)

        # optimiser = torch.optim.SGD([params_], lr=0.01)

        def objective_scipy(params: np.ndarray) -> float:
            params = torch.tensor(copy.deepcopy(params))
            param_history.append(params)
            value = objective(params)
            value_history.append(value)
            return value.item()

        tic = time.time()

        # for i in range(n):
        #     def closure():
        #         param_history[i] = copy.deepcopy(params_)
        #         value = objective(params_)
        #         value_history[i] = value
        #         return value
        #
        #     optimiser.step(closure)
        res = scipy.optimize.minimize(objective_scipy, params_.numpy(), method='Nelder-Mead')

        toc = time.time()
        print("Done. Took {:.3f}s.".format(toc - tic))

        print(res)

        # print("Final value = {:.3f} at params = {}".format(value_history[-1], param_history[-1]))

        final_image = generate_drr(vol_data,
                                   transformation=Transformation(torch.tensor(res.x[0:3]), torch.tensor(res.x[3:6])),
                                   voxel_spacing=voxel_spacing, detector_spacing=detector_spacing,
                                   scene_geometry=scene_geometry, output_size=drr_image.size(), samples_per_ray=512)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(final_image.cpu())
        axes.axis('square')
        axes.set_title("drr final")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        plt.colorbar(mesh)

        param_history = torch.stack(param_history, dim=0)
        value_history = torch.tensor(value_history)

        _, axes = plt.subplots()
        axes.plot(param_history[:, 0])
        axes.plot(param_history[:, 1])
        axes.plot(param_history[:, 2])
        axes.plot(param_history[:, 3])
        axes.plot(param_history[:, 4])
        axes.plot(param_history[:, 5])
        axes.set_xlabel("iteration")
        axes.set_ylabel("param value")

        _, axes = plt.subplots()
        axes.plot(value_history)
        axes.set_xlabel("iteration")
        axes.set_ylabel("-zncc")

        plt.show()
