import argparse
from typing import Tuple

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from notification import logs_setup
from registration.lib.structs import LinearRange
from registration.lib.sinogram import SinogramClassic, SinogramHEALPix, Sinogram3dGrid


def spherical_to_cartesian(grid: Sinogram3dGrid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cos_theta = grid.theta.cos()
    sin_theta = grid.theta.sin()
    cos_phi = grid.phi.cos()
    sin_phi = grid.phi.sin()
    x = grid.r * cos_theta * cos_phi
    y = grid.r * cos_theta * sin_phi
    z = grid.r * sin_theta
    return x, y, z


def main():
    print("Hello, World")
    sinogram_size: int = 14
    grid: Sinogram3dGrid = SinogramClassic.build_grid(
        counts=(int(torch.ceil(float(sinogram_size) * torch.tensor(0.25 * torch.pi).sqrt())),
                int(torch.ceil(float(sinogram_size) * torch.tensor(0.25 * torch.pi).sqrt())), 2),
        r_range=LinearRange(-100.0, 100.0))
    grid_healpix: Sinogram3dGrid = SinogramHEALPix.build_grid(
        n_side=int(torch.ceil(torch.tensor(float(sinogram_size)) / torch.tensor(12.).sqrt()).item()),
        r_range=LinearRange(-100.0, 100.0), r_count=2)

    fig = plt.figure()
    fig.patch.set_facecolor('none')  # transparent figure background
    axes = fig.add_subplot(121, projection='3d')
    x, y, z = spherical_to_cartesian(grid)
    axes.scatter(x, y, z, s=30)
    axes.set_aspect('equal')
    axes.set_axis_off()
    axes.set_facecolor('none')
    axes = fig.add_subplot(122, projection='3d')
    x, y, z = spherical_to_cartesian(grid_healpix)
    axes.scatter(x, y, z, s=30)
    axes.set_aspect('equal')
    axes.set_axis_off()
    axes.set_facecolor('none')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=-0.3, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    _parser = argparse.ArgumentParser(description="", epilog="")
    # _parser.add_argument("-c", "--cache-directory", type=str, default="cache",
    #                      help="Set the directory where data that is expensive to calculate will be saved. The default "
    #                           "is 'cache'.")
    # _parser.add_argument("-p", "--ct-path", type=str,
    #                      help="Give a path to a .nrrd file containing CT data, or a directory containing .dcm CT slices"
    #                           " to process. If not provided, some simple synthetic data will be used instead - note "
    #                           "that in this case, data will not be saved to the cache.")
    # _parser.add_argument("-i", "--no-load", action='store_true',
    #                      help="Do not load any pre-calculated data from the cache.")
    # _parser.add_argument("-r", "--regenerate-drr", action='store_true',
    #                      help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # _parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    # _parser.add_argument("-s", "--sinogram-size", type=int, default=None,
    #                      help="The number of values of r, theta and phi to calculate plane integrals for, "
    #                           "and the width and height of the grid of samples taken to approximate each integral. The "
    #                           "computational expense of the 3D Radon transform is O(sinogram_size^5). If not given, the "
    #                           "default is determined by the size of the CT volume.")
    # _parser.add_argument("-x", "--xray-path", type=str,
    #                      help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
    #                           "this is provided, the X-ray will by used instead of any DRR.")
    # _parser.add_argument("-d", "--drr-size", type=int, default=1000,
    #                      help="The size of the square DRR to generate as the fixed image if no X-ray is given.")
    # _parser.add_argument("-t", "--sinogram-type", type=str, default="classic",
    #                      help="String name of the storage method for the 3D sinogram. Must be one of: 'classic', "
    #                           "'healpix'.")
    _args = _parser.parse_args()

    main()
