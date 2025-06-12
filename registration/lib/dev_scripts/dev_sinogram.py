import argparse
import logging.config

import torch
import matplotlib.pyplot as plt
import pyvista as pv

from registration.lib.structs import *
from registration.lib import sinogram


def map_back_and_forth():
    n_side: int = 5

    _phi = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi).generate_tex_coord_grid(40)
    _theta = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi).generate_tex_coord_grid(40)
    _phi, _theta = torch.meshgrid(_phi, _theta)
    _grid = Sinogram3dGrid(phi=_phi, theta=_theta, r=torch.zeros_like(_phi))

    _u, _v = sinogram.SinogramHEALPix.spherical_to_tex_coord(_grid, n_side)

    print("n_side = {}".format(n_side))
    print("desired u range: 0.5 to {}".format(3.0 * n_side + 2.5))
    print("u range: {} to {}".format(_u.min().item(), _u.max().item()))
    print("desired v range: 0.5 to {}".format(2.0 * n_side + 2.5))
    print("v range: {} to {}".format(_v.min().item(), _v.max().item()))

    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(_phi.numpy(), _theta.numpy(), _u.numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("phi")
    _axes.set_ylabel("theta")
    _axes.set_aspect('equal', 'box')
    _axes.set_title("u")

    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(_phi.numpy(), _theta.numpy(), _v.numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("phi")
    _axes.set_ylabel("theta")
    _axes.set_aspect('equal', 'box')
    _axes.set_title("v")

    _phi2, _theta2 = sinogram.SinogramHEALPix.tex_coord_to_spherical(_u, _v, n_side)

    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(_phi.numpy(), _theta.numpy(), _phi2.numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("phi")
    _axes.set_ylabel("theta")
    _axes.set_aspect('equal', 'box')
    _axes.set_title("phi")

    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(_phi.numpy(), _theta.numpy(), _theta2.numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("phi")
    _axes.set_ylabel("theta")
    _axes.set_aspect('equal', 'box')
    _axes.set_title("theta")

    plt.show()


def show_padding():
    n_side: int = 5

    _u = torch.arange(3 * n_side)
    _v = torch.arange(2 * n_side)
    _v, _u = torch.meshgrid(_v, _u)

    _s = sinogram.SinogramHEALPix(_u.unsqueeze(0), LinearRange(0.0, 1.0))
    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(torch.arange(3 * n_side + 4).numpy(), torch.arange(2 * n_side + 4).numpy(),
                             _s.data[0].numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("u")
    _axes.set_ylabel("v")
    _axes.invert_yaxis()
    _axes.set_aspect('equal', 'box')
    _axes.set_title("u")

    _s = sinogram.SinogramHEALPix(_v.unsqueeze(0), LinearRange(0.0, 1.0))
    _, _axes = plt.subplots()
    _mesh = _axes.pcolormesh(torch.arange(3 * n_side + 4).numpy(), torch.arange(2 * n_side + 4).numpy(),
                             _s.data[0].numpy())
    plt.colorbar(_mesh)
    _axes.set_xlabel("u")
    _axes.set_ylabel("v")
    _axes.invert_yaxis()
    _axes.set_aspect('equal', 'box')
    _axes.set_title("v")

    plt.show()


def plot_sphere():
    pass


if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("../../../logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    # parse arguments
    _parser = argparse.ArgumentParser()
    _subparsers = _parser.add_subparsers(dest="command")
    _parser_map = _subparsers.add_parser("map-back-and-forth", help="")
    _parser_pad = _subparsers.add_parser("show-padding", help="")
    _parser_run = _subparsers.add_parser("plot-sphere", help="")
    _args = _parser.parse_args()

    if _args.command == "map-back-and-forth":
        map_back_and_forth()
    elif _args.command == "show-padding":
        show_padding()
    else:  # _args.command == "plot-sphere"
        plot_sphere()
