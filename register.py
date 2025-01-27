from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as pgo
import time
import torch
import nrrd
from scipy.spatial.transform import Rotation

from diffdrr.drr import DRR
from diffdrr.data import read
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


# class VolumeLinearRange:
#     def __init__(self, x_range: LinearRange, y_range: LinearRange, z_range: LinearRange):
#         self.x_range = x_range
#         self.y_range = y_range
#         self.z_range = z_range


def calculate_radon_volume(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, samples_per_direction: int = 128,
                           phi_range: LinearRange, theta_range: LinearRange, r_range: LinearRange, device):
    phi_values = phi_range.generate_range(64, device=device)
    theta_values = theta_range.generate_range(64, device=device)
    r_values = r_range.generate_range(64, device=device)
    return ExtensionTest.dRadon3dDR(volume_data, voxel_spacing[2].item(), voxel_spacing[1].item(),
                                    voxel_spacing[0].item(), phi_values, theta_values, r_values, samples_per_direction)


def calculate_fixed_image(drr_image: torch.Tensor, *, source_distance: float, detector_spacing: float,
                          phi_values: torch.Tensor, r_values: torch.Tensor,
                          samples_per_line: int = 128) -> torch.Tensor:
    img_width = drr_image.size()[1]
    img_height = drr_image.size()[0]

    xs = detector_spacing * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    ys, xs = torch.meshgrid(ys, xs)
    cos_gamma = source_distance / torch.sqrt(xs.square() + ys.square() + source_distance * source_distance)
    g_tilde = cos_gamma.to('cuda') * drr_image

    fixed_scaling = (r_values / source_distance).square() + 1.

    return fixed_scaling * ExtensionTest.dRadon2dDR(g_tilde, detector_spacing, detector_spacing, phi_values, r_values,
                                                    samples_per_line)


def deb_brief(name: str, tensor: torch.Tensor):
    print(
        "tensor '{}': occupies range ({:.3e}, {:.3e}), with median {:.3e}, mean {:.3e} and std dev {:.3e}".format(name,
                                                                                                                  tensor.min().item(),
                                                                                                                  tensor.max().item(),
                                                                                                                  tensor.median().item(),
                                                                                                                  tensor.mean().item(),
                                                                                                                  torch.sqrt(
                                                                                                                      tensor.var()).item()))


def fixed_polar_to_moving_cartesian(phis: torch.Tensor, rs: torch.tensor, *, source_distance: float,
                                    ct_origin_distance: float):
    hypotenuses = torch.sqrt(rs.square() + source_distance * source_distance)
    sin_alphas = rs / hypotenuses
    ys = sin_alphas.square() * (source_distance - ct_origin_distance)
    scaled_rs = (source_distance - ct_origin_distance - ys) * rs / source_distance
    xs = -scaled_rs * torch.sin(phis)
    zs = -scaled_rs * torch.cos(phis)
    return xs, ys, zs


def moving_cartesian_to_moving_spherical(xs: torch.Tensor, ys: torch.Tensor, zs: torch.Tensor):
    phis = torch.atan2(ys, xs)
    phis_over = phis > .5 * torch.pi
    phis_under = phis < -.5 * torch.pi
    phis[phis_over] -= torch.pi
    phis[phis_under] += torch.pi
    thetas = torch.atan2(zs, torch.sqrt(xs.square() + ys.square()))
    rs = torch.sqrt(xs.square() + ys.square() + zs.square())
    rs[torch.logical_or(phis_over, phis_under)] *= -1.
    return phis, thetas, rs


def resample_slice(volume: torch.Tensor, *, rotation: torch.Tensor, translation: torch.Tensor, source_distance: float,
                   ct_origin_distance: float, phi_values_po: torch.Tensor, r_values_po: torch.Tensor,
                   phi_range_sph: LinearRange, theta_range_sph: LinearRange, r_range_sph: LinearRange):
    phi_values_po, r_values_po = torch.meshgrid(phi_values_po, r_values_po)

    phi_values_sph, theta_values_sph, r_values_sph = moving_cartesian_to_moving_spherical(
        *fixed_polar_to_moving_cartesian(phi_values_po, r_values_po, source_distance=source_distance,
                                         ct_origin_distance=ct_origin_distance))

    grid_range = LinearRange(-1., 1.)

    i_mapping: LinearMapping = grid_range.get_mapping_from(r_range_sph)
    j_mapping: LinearMapping = grid_range.get_mapping_from(theta_range_sph)
    k_mapping: LinearMapping = grid_range.get_mapping_from(phi_range_sph)

    grid = torch.stack((i_mapping(r_values_sph), j_mapping(theta_values_sph), k_mapping(phi_values_sph)), dim=-1)
    deb_brief("grid rs", grid[:, :, 0])
    deb_brief("grid thetas", grid[:, :, 1])
    deb_brief("grid phis", grid[:, :, 2])
    return torch.nn.functional.grid_sample(volume[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]


def register():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_distance: float = 1000.  # [mm]; distance in the positive y-direction from the centre of the detector array
    detector_spacing: float = 10.  # [mm]
    voxel_spacing = torch.Tensor([10., 10., 10.])  # [mm]
    ct_origin_distance: float = 100.  ## [mm]; distance in the positive y-direction from the centre of the detector array

    vol_size = torch.Size([10, 10, 10])
    # vol_data = torch.rand(vol_size, device=device)
    vol_data = torch.zeros(vol_size, device=device)
    # vol_data[:, 0, 0] = torch.linspace(0., 1., vol_data.size()[0])
    # vol_data[0, :, 0] = torch.linspace(0., 1., vol_data.size()[1])
    # vol_data[0, 0, :] = torch.linspace(0., 1., vol_data.size()[2])
    # vol_data[0, :, 0] = torch.linspace(1., 0., vol_data.size()[1])
    # vol_data[-1, 0, 0] = 0.5
    # vol_data[-1, -1, -1] = 0.2
    vol_data[0, 0, 0] = 1.
    vol_image = ScalarImage(tensor=vol_data[None, :, :, :])
    vol_subject = read(vol_image, spacing=voxel_spacing)

    phi_range_sph = LinearRange(-.5 * torch.pi, .5 * torch.pi)
    theta_range_sph = LinearRange(-.5 * torch.pi, .5 * torch.pi)
    vol_size_world = voxel_spacing * torch.tensor(vol_size, dtype=torch.float32)
    vol_diag: float = vol_size_world.square().sum().sqrt().item()
    r_range_sph = LinearRange(-.5 * vol_diag, .5 * vol_diag)

    r_mu = calculate_radon_volume(vol_data, voxel_spacing=voxel_spacing, phi_range=phi_range_sph,
                                  theta_range=theta_range_sph, r_range=r_range_sph, device=device)

    X, Y, Z = torch.meshgrid([torch.arange(0, 64, 1), torch.arange(0, 64, 1), torch.arange(0, 64, 1)])
    fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=r_mu.cpu().flatten(),
                                     isomin=r_mu.min().item(), isomax=r_mu.max().item(), opacity=.2, surface_count=21))
    fig.show()

    # I believe that the detector array lies on the x-z plane, with x down, and z to the left (and so y outward)
    drr_generator = DRR(vol_subject,  # An object storing the CT volume, origin, and voxel spacing
                        sdd=source_distance,  # Source-to-detector distance (i.e., focal length)
                        height=int(torch.ceil(
                            1.1 * voxel_spacing.mean() * torch.tensor(vol_size).max() / detector_spacing).item()),
                        # Image height (if width is not provided, the generated DRR is square)
                        delx=detector_spacing,  # Pixel spacing (in mm)
                        ).to(device)

    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, ct_origin_distance, 0.0]], device=device)

    drr_image = drr_generator(rotations, translations, parameterization="euler_angles", convention="ZXY")
    # plot_drr(drr_image, ticks=False)
    drr_image = drr_image[0, 0]
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(drr_image.cpu())
    axes.axis('square')
    plt.colorbar(mesh)

    rhs_size = 64

    phi_values_po = torch.linspace(-.5 * torch.pi, .5 * torch.pi, rhs_size, device=device)
    image_diag: float = (
            detector_spacing * torch.tensor(drr_image.size(), dtype=torch.float32)).square().sum().sqrt().item()
    r_values_po = torch.linspace(-.5 * image_diag, .5 * image_diag, rhs_size, device=device)

    rhs = calculate_fixed_image(drr_image, source_distance=source_distance, detector_spacing=detector_spacing,
                                phi_values=phi_values_po, r_values=r_values_po)

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(rhs.cpu())
    axes.axis('square')
    plt.colorbar(mesh)

    _, axes = plt.subplots()
    resampled = resample_slice(r_mu, rotation=rotations[0].cpu(), translation=translations[0].cpu(),
                               source_distance=source_distance, ct_origin_distance=ct_origin_distance,
                               phi_values_po=phi_values_po, r_values_po=r_values_po, phi_range_sph=phi_range_sph,
                               theta_range_sph=theta_range_sph, r_range_sph=r_range_sph)
    mesh = axes.pcolormesh(resampled.cpu())
    axes.axis('square')
    plt.colorbar(mesh)

    plt.show()
