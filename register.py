from typing import Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import time
import torch
import nrrd

from diffdrr.drr import DRR
from diffdrr.data import read
from torchio.data.image import ScalarImage
from diffdrr.visualization import plot_drr

import Extension as ExtensionTest


def register():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_position: float = 1000.  # in mm
    detector_spacing: float = 2.  # in mm

    vol_size = torch.Size([3, 3, 5])
    vol_data = torch.rand(vol_size)
    vol_image = ScalarImage(tensor=vol_data[None, :, :, :])
    vol_subject = read(vol_image)

    drr_generator = DRR(vol_subject,  # An object storing the CT volume, origin, and voxel spacing
                        sdd=source_position,  # Source-to-detector distance (i.e., focal length)
                        height=int(torch.ceil(1.1 * torch.tensor(vol_size).max()).item()),
                        # Image height (if width is not provided, the generated DRR is square)
                        delx=detector_spacing,  # Pixel spacing (in mm)
                        ).to(device)

    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, source_position * 0.8, 0.0]], device=device)

    img = drr_generator(rotations, translations, parameterization="euler_angles", convention="ZXY")
    plot_drr(img, ticks=False)
    plt.show()

    lhs = ExtensionTest.dRadon3dDR(vol_data, 1., 1., 1., 64, 64, 64, 10)

    img_width = img.size()[3]
    img_height = img.size()[2]

    xs = detector_spacing * (torch.arange(0, img_width, 1, dtype=torch.float32) - 0.5 * float(img_width - 1))
    ys = detector_spacing * (torch.arange(0, img_height, 1, dtype=torch.float32) - 0.5 * float(img_height - 1))
    ys, xs = torch.meshgrid(ys, xs)
    sq_mags = xs * xs + ys * ys
    cos_gamma = source_position / torch.sqrt(source_position * source_position + sq_mags)
    g_tilde = cos_gamma.to('cuda') * img

    phi_values = torch.pi * (-.5 + torch.arange(0, 32, 1, dtype=torch.float32, device=device) / float(32 - 1))
    r_values = torch.linspace(0., .5 * torch.sqrt(
        torch.tensor([float(img_height) * float(img_height) + float(img_width) * float(img_width)])).item(), 32,
                              device=device)

    fixed_scaling = (r_values * r_values / (source_position * source_position)) + 1.

    rhs = fixed_scaling * ExtensionTest.dRadon2dDR(g_tilde[0, 0], detector_spacing, detector_spacing, phi_values,
                                                   r_values, 32)

    plt.pcolormesh(rhs.cpu())
    plt.axis('square')
    plt.show()
