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


def calculate_radon_volume(volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor, device):
    phi_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, 64, device=device)
    theta_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, 64, device=device)
    image_depth: torch.Tensor = 1. * torch.tensor(volume_data.size()[0], dtype=torch.float32)
    image_height: torch.Tensor = 1. * torch.tensor(volume_data.size()[1], dtype=torch.float32)
    image_width: torch.Tensor = 1. * torch.tensor(volume_data.size()[2], dtype=torch.float32)
    image_diag = torch.sqrt(image_depth.square() + image_height.square() + image_width.square()).item()
    r_values = torch.linspace(-.5 * image_diag, .5 * image_diag, 64, device=device)

    return ExtensionTest.dRadon3dDR(volume_data, voxel_spacing[2].item(), voxel_spacing[1].item(),
                                    voxel_spacing[0].item(), phi_values, theta_values, r_values, 64)


def calculate_fixed_image(drr_image: torch.Tensor, *, source_distance: float, detector_spacing: float,
                          phi_values: torch.Tensor, r_values: torch.Tensor) -> torch.Tensor:
    img_width = drr_image.size()[1]
    img_height = drr_image.size()[0]

    xs = detector_spacing * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    ys, xs = torch.meshgrid(ys, xs)
    sq_mags = xs * xs + ys * ys
    cos_gamma = source_distance / torch.sqrt(source_distance * source_distance + sq_mags)
    g_tilde = cos_gamma.to('cuda') * drr_image

    fixed_scaling = (r_values * r_values / (source_distance * source_distance)) + 1.

    return fixed_scaling * ExtensionTest.dRadon2dDR(g_tilde, detector_spacing, detector_spacing, phi_values, r_values,
                                                    64)


def deb_brief(name: str, tensor: torch.Tensor):
    print(
        "tensor '{}': occupies range ({:.3e}, {:.3e}), with median {:.3e}, mean {:.3e} and std dev {:.3e}".format(name,
                                                                                                                  tensor.min().item(),
                                                                                                                  tensor.max().item(),
                                                                                                                  tensor.median().item(),
                                                                                                                  tensor.mean().item(),
                                                                                                                  torch.sqrt(
                                                                                                                      tensor.var()).item()))


def resample_slice(volume: torch.Tensor, *, volume_size: torch.Tensor, voxel_spacing: torch.Tensor,
                   rotation: torch.Tensor, translation: torch.Tensor, source_distance: float, ct_origin_distance: float,
                   phi_values: torch.Tensor, r_values: torch.Tensor):
    # R = torch.from_numpy(Rotation.from_rotvec(rotation).as_matrix()).to(torch.float32)
    # transform = RigidTransform(make_matrix(R, translation))
    deb_brief("phi_values", phi_values)
    deb_brief("r_values", r_values)
    phi_values, r_values = torch.meshgrid(phi_values, r_values)
    hypoteneuses = torch.sqrt(r_values.square() + source_distance * source_distance)
    deb_brief("hypoteneuses", hypoteneuses)
    sin_alphas = r_values / hypoteneuses
    deb_brief("sin_alphas", sin_alphas)
    ys = sin_alphas.square() * (source_distance - ct_origin_distance)
    deb_brief("ys", ys)
    scaled_rs = (source_distance - ct_origin_distance - ys) * r_values / source_distance
    deb_brief("scaled_rs", scaled_rs)
    xs = scaled_rs * torch.cos(phi_values)
    deb_brief("xs", xs)
    zs = -scaled_rs * torch.sin(phi_values)
    deb_brief("zs", zs)
    # cartesian_positions = torch.stack((xs, ys, zs), dim=-1)

    spherical_phis = torch.atan2(xs, ys)
    spherical_thetas = torch.atan2(-zs, torch.sqrt(xs.square() + ys.square()))
    spherical_rs = torch.sqrt(xs.square() + ys.square() + zs.square())
    a = torch.tensor([-1., 0., 0.], device=volume.device)
    b = torch.tensor([2. / (volume_size * voxel_spacing).mean().item(), 2. / torch.pi, 2. / torch.pi], device=volume.device)
    grid = a + b * torch.stack((spherical_rs, spherical_thetas, spherical_phis), dim=-1)
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

    vol_size = torch.Size([30, 30, 30])
    # vol_data = torch.rand(vol_size, device=device)
    vol_data = torch.zeros(vol_size, device=device)
    vol_data[5, 12, 15] = 2.
    vol_data[0, 0, 0] = 1.
    vol_data[15, 15, 15] = .5
    vol_image = ScalarImage(tensor=vol_data[None, :, :, :])
    vol_subject = read(vol_image, spacing=voxel_spacing)

    r_mu = calculate_radon_volume(vol_data, voxel_spacing=voxel_spacing, device=device)

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
    plot_drr(drr_image, ticks=False)
    drr_image = drr_image[0, 0]

    phi_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, 32, device=device)
    image_height: torch.Tensor = detector_spacing * torch.tensor(drr_image.size()[0], dtype=torch.float32)
    image_width: torch.Tensor = detector_spacing * torch.tensor(drr_image.size()[1], dtype=torch.float32)
    image_diag = torch.sqrt(image_height.square() + image_width.square()).item()
    r_values = torch.linspace(-.5 * image_diag, .5 * image_diag, 32, device=device)

    rhs = calculate_fixed_image(drr_image, source_distance=source_distance, detector_spacing=detector_spacing,
                                phi_values=phi_values, r_values=r_values)

    _, axes = plt.subplots()
    axes.pcolormesh(rhs.cpu())
    axes.axis('square')

    _, axes = plt.subplots()
    resampled = resample_slice(r_mu, volume_size=torch.tensor(vol_size, dtype=torch.float32),
                               voxel_spacing=voxel_spacing, rotation=rotations[0].cpu(),
                               translation=translations[0].cpu(), source_distance=source_distance,
                               ct_origin_distance=ct_origin_distance, phi_values=phi_values, r_values=r_values)
    axes.pcolormesh(resampled.cpu())
    axes.axis('square')

    plt.show()
