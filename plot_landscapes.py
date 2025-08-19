import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
import pathlib

import logs_setup
from registration.plot_data import LandscapePlotData


def to_latex_scientific(x: float, precision: int = 2, include_plus: bool = False):
    if x == 0:
        return f"{0:.{precision}f}"
    exponent: int = int(f"{x:e}".split("e")[1])
    mantissa: float = x / (10.0 ** exponent)
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    if include_plus:
        return fr"{mantissa:+.{precision}f} \times 10^{{{exponent}}}"
    return fr"{mantissa:.{precision}f} \times 10^{{{exponent}}}"


DATA_DIRECTORY = pathlib.Path("data/temp/landscapes")
SAVE_DIRECTORY = pathlib.Path("figures/landscapes")


def main():
    assert DATA_DIRECTORY.is_dir()

    for file in DATA_DIRECTORY.iterdir():
        if not file.is_file() or file.suffix != ".pkl":
            continue

        pdata = torch.load(file, weights_only=False)
        if not isinstance(pdata, LandscapePlotData):
            logger.error("File '{}' is of unrecognized type '{}'.".format(str(file), type(pdata).__name__))
            continue

        fig = plt.figure(figsize=(6.5, 6))
        axes = fig.add_subplot(1, 1, 1, projection="3d")
        param2_grid, param1_grid = torch.meshgrid(pdata.values2, pdata.values1)
        axes.plot_surface(param1_grid.clone().detach().cpu().numpy(), param2_grid.clone().detach().cpu().numpy(),
                          pdata.height.clone().detach().cpu().numpy(), cmap=cm.get_cmap("viridis"))
        axes.set_xlabel(pdata.label1)
        axes.set_ylabel(pdata.label2)
        axes.set_zlabel("$-\mathrm{ZNCC}$")
        plt.tight_layout()
        plt.savefig(SAVE_DIRECTORY / "landscape_{}.png".format(file.stem))
        plt.close()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # for outputting PGFs
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["scatter.marker"] = 'x'
    plt.rcParams["font.size"] = 22  # figures are includes in latex at quarte size, so 36 is desired size. matplotlib
    # scales up by 1.2 (God only knows why). 36 is tool big, however, so going a bit smaller than 30

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    # parser.add_argument(
    #     "-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    main()
