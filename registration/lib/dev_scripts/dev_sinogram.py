import argparse
import logging.config

import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from registration.lib.structs import *
from registration.lib import sinogram
from registration import pre_computed


def map_back_and_forth():
    n_side: int = 5

    phi = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi).generate_tex_coord_grid(40)
    theta = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi).generate_tex_coord_grid(40)
    phi, theta = torch.meshgrid(phi, theta)
    _grid = Sinogram3dGrid(phi=phi, theta=theta, r=torch.zeros_like(phi))

    u, v, r = sinogram.SinogramHEALPix.spherical_to_tex_coord(_grid, n_side)

    logger.info("n_side = {}".format(n_side))
    logger.info("desired u range: 0.5 to {}".format(3.0 * n_side + 2.5))
    logger.info("u range: {} to {}".format(u.min().item(), u.max().item()))
    logger.info("desired v range: 0.5 to {}".format(2.0 * n_side + 2.5))
    logger.info("v range: {} to {}".format(v.min().item(), v.max().item()))

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(phi.numpy(), theta.numpy(), u.numpy())
    plt.colorbar(mesh)
    axes.set_xlabel("phi")
    axes.set_ylabel("theta")
    axes.set_aspect('equal', 'box')
    axes.set_title("u")

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(phi.numpy(), theta.numpy(), v.numpy())
    plt.colorbar(mesh)
    axes.set_xlabel("phi")
    axes.set_ylabel("theta")
    axes.set_aspect('equal', 'box')
    axes.set_title("v")

    phi2, theta2 = sinogram.SinogramHEALPix.tex_coord_to_spherical(u, v, r, n_side)

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(phi.numpy(), theta.numpy(), phi2.numpy())
    plt.colorbar(mesh)
    axes.set_xlabel("phi")
    axes.set_ylabel("theta")
    axes.set_aspect('equal', 'box')
    axes.set_title("phi")

    _, axes = plt.subplots()
    mesh = axes.pcolormesh(phi.numpy(), theta.numpy(), theta2.numpy())
    plt.colorbar(mesh)
    axes.set_xlabel("phi")
    axes.set_ylabel("theta")
    axes.set_aspect('equal', 'box')
    axes.set_title("theta")

    plt.show()


def show_padding(output_directory: str | None, no_show: bool):
    n_side: int = 4

    u = torch.arange(3 * n_side)
    v = torch.arange(2 * n_side)
    v, u = torch.meshgrid(v, u)

    u_sinogram = sinogram.SinogramHEALPix(u.unsqueeze(0) + 1.0, LinearRange(0.0, 1.0))
    u_padding = u_sinogram.data[0] < 0.5
    u_sinogram._data -= 1.0

    if not no_show:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(
            torch.arange(3 * n_side + 4).numpy(), torch.arange(2 * n_side + 4).numpy(), u_sinogram.data[0].numpy())
        plt.colorbar(mesh)
        axes.set_xlabel("u")
        axes.set_ylabel("v")
        axes.invert_yaxis()
        axes.set_aspect('equal', 'box')
        axes.set_title("u")

    v_sinogram = sinogram.SinogramHEALPix(v.unsqueeze(0) + 1.0, LinearRange(0.0, 1.0))
    v_padding = v_sinogram.data[0] < 0.5
    v_sinogram._data -= 1.0

    if not no_show:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(
            torch.arange(3 * n_side + 4).numpy(), torch.arange(2 * n_side + 4).numpy(), v_sinogram.data[0].numpy())
        plt.colorbar(mesh)
        axes.set_xlabel("u")
        axes.set_ylabel("v")
        axes.invert_yaxis()
        axes.set_aspect('equal', 'box')
        axes.set_title("v")

        plt.show()

    if output_directory is not None:
        output_directory = pathlib.Path(output_directory)
        if not output_directory.is_dir():
            logger.error("Output directory '{}' does not exist.".format(str(output_directory)))
            exit(1)

        i = torch.arange(3 * n_side + 4)
        j = torch.arange(2 * n_side + 4)
        j, i = torch.meshgrid(j, i)
        u = u_sinogram.data[0].to(dtype=torch.int32)
        v = v_sinogram.data[0].to(dtype=torch.int32)

        # saving the scalar data in PGF plot table format
        u_plot_path = output_directory / "u_padding_plot.dat"
        with open(u_plot_path, "w") as file:
            for row in range(i.size()[0]):
                for col in range(i.size()[1]):
                    file.write("{} {} {}\n".format(i[row, col], j[row, col], u[i.size()[0] - row - 1, col]))
                file.write("\n")
        logger.info("`u` coord padding plot saved to '{}'".format(str(u_plot_path)))

        v_plot_path = output_directory / "v_padding_plot.dat"
        with open(v_plot_path, "w") as file:
            for row in range(i.size()[0]):
                for col in range(i.size()[1]):
                    file.write("{} {} {}\n".format(i[row, col], j[row, col], v[i.size()[0] - row - 1, col]))
                file.write("\n")
        logger.info("`v` coord padding plot saved to '{}'".format(str(u_plot_path)))

        u_labels_path = output_directory / "u_padding_labels.tex"
        with open(u_labels_path, "w") as file:
            for row in range(i.size()[0]):
                for col in range(i.size()[1]):
                    file.write(
                        "\\node at (axis cs:{},{}) {{\\scriptsize {}}};\n".format(
                            i[row, col], j[row, col],
                            "-" if u_padding[i.size()[0] - row - 1, col] else u[i.size()[0] - row - 1, col]))
                file.write("\n")
        logger.info("`u` coord labels saved to '{}'".format(str(u_labels_path)))

        v_labels_path = output_directory / "v_padding_labels.tex"
        with open(v_labels_path, "w") as file:
            for row in range(i.size()[0]):
                for col in range(i.size()[1]):
                    file.write(
                        "\\node at (axis cs:{},{}) {{\\scriptsize {}}};\n".format(
                            i[row, col], j[row, col],
                            "-" if v_padding[i.size()[0] - row - 1, col] else v[i.size()[0] - row - 1, col]))
                file.write("\n")
        logger.info("`v` coord labels saved to '{}'".format(str(v_labels_path)))


def plot_sphere():
    n_side: int = 4

    sphere = pv.Sphere(theta_resolution=360, phi_resolution=180)  # high resolution

    points = torch.tensor(sphere.points)  # shape (N, 3)

    # Compute spherical UV coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    phi = torch.atan2(y, x)
    theta = (z / points.norm(dim=1)).asin()

    logger.info("theta in ({}, {})".format(theta.min(), theta.max()))
    logger.info("phi in ({}, {})".format(phi.min(), phi.max()))

    grid = Sinogram3dGrid(phi=phi, theta=theta, r=20.0 * torch.ones_like(phi)).unflip()

    vol_data = torch.zeros((7, 7, 7))
    vol_data[1, 1, 1] = 1.
    vol_data[0, 3, :] = 0.7
    vol_data[6, :, :] = 0.2
    vol_data[3:6, 2, 3] = 0.8
    voxel_spacing = torch.tensor([10., 10., 10.])
    sinogram3d, _ = pre_computed.calculate_volume_sinogram(
        None, vol_data, voxel_spacing=voxel_spacing, ct_volume_path=None, volume_downsample_factor=1,
        save_to_cache=False, sinogram_size=24, sinogram_type=sinogram.SinogramHEALPix)

    sampled = sinogram3d.sample(grid)
    _min = sampled.min()
    _max = sampled.max()
    sampled_mapped = 255.0 * (sampled - _min) / (_max - _min)
    colors = torch.stack((sampled_mapped, sampled_mapped, sampled_mapped), dim=-1).cpu().numpy()
    colors = colors.astype(np.uint8)
    # Assign colors to vertices
    sphere.point_data["colors"] = colors
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, scalars="colors", rgb=True)
    plotter.show()


if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    # parse arguments
    _parser = argparse.ArgumentParser()
    _subparsers = _parser.add_subparsers(dest="command")

    _parser_map = _subparsers.add_parser("map-back-and-forth", help="")

    _parser_pad = _subparsers.add_parser("show-padding", help="")
    _parser_pad.add_argument("-o", "--output-dir", type=str, help="")
    _parser_pad.add_argument("-n", "--no-show", action="store_true", help="")

    _parser_run = _subparsers.add_parser("plot-sphere", help="")

    _args = _parser.parse_args()

    if _args.command == "map-back-and-forth":
        map_back_and_forth()
    elif _args.command == "show-padding":
        show_padding(output_directory=_args.output_dir, no_show=_args.no_show)
    else:  # _args.command == "plot-sphere"
        plot_sphere()
