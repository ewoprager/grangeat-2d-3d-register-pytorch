import argparse

import torch
import pathlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd

from reg23_experiments.notification import logs_setup


def convert_to_dataframe(directory: pathlib.Path) -> pd.DataFrame:
    config = torch.load(directory / "config.pkl")
    assert isinstance(config, dict)
    nominal_distances = config.pop("nominal_distances")
    if "distance_distribution" in config:
        config.pop("distance_distribution")
    if "iteration_count" in config:
        config.pop("iteration_count")
    if "notes" in config:
        config.pop("notes")
    shared_parameters = torch.load(directory / "shared_parameters.pkl")
    assert isinstance(shared_parameters, dict)
    row_global = config | shared_parameters
    rows_out = []
    for element in directory.iterdir():
        if not element.is_dir():
            continue
        parameters = torch.load(element / "parameters.pkl")
        rows_here = row_global | parameters
        convergence_series = torch.load(element / "convergence_series.pkl")  # size = (n.d. count, it. count)
        nominal_distance_count = convergence_series.size(0)
        iteration_count = convergence_series.size(1)
        for j in range(nominal_distance_count):
            for i in range(iteration_count):
                rows_out.append(rows_here | {"starting_distance": nominal_distances[j].item(), "iteration": i,
                                             "distance": convergence_series[j, i].item()})
    return pd.DataFrame(rows_out)


def dataframe_to_tensor(df: pd.DataFrame, *, ordered_axes: list[str], value_column: str) -> tuple[
    torch.Tensor, dict[str, torch.Tensor]]:
    s = df.set_index(ordered_axes)[value_column].sort_index()
    axis_levels = [  #
        s.index.levels[s.index.names.index(name)]  #
        for name in ordered_axes  #
    ]
    full_index = pd.MultiIndex.from_product(axis_levels, names=ordered_axes)
    s = s.reindex(full_index)
    if s.isna().any():
        logger.warn("Grid is incomplete — missing coordinate combinations.")
    axis_values = {  #
        name: torch.from_numpy(level.to_numpy())  #
        for name, level in zip(ordered_axes, axis_levels)  #
    }
    axis_lengths = [len(level) for level in axis_levels]
    tensor = torch.from_numpy(s.to_numpy()).view(*axis_lengths)
    return tensor, axis_values


def main(*, load_dir: str | pathlib.Path, which_dataset: str, display: bool) -> None:
    assert load_dir.is_dir()
    # Getting the latest data instance if desired
    if which_dataset == "latest":
        subdirs = []
        for element in load_dir.iterdir():
            if not element.is_dir():
                continue
            subdirs.append(str(element.stem))
        subdirs.sort()
        which_dataset = subdirs[-1]
    instance_dir: pathlib.Path = load_dir / which_dataset
    assert instance_dir.is_dir()
    # Reading in parquet data and concatenating
    df = pd.concat([  #
        pd.read_parquet(element)  #
        for element in instance_dir.iterdir()  #
        if element.stem.startswith("data") and element.suffix == ".parquet"  #
    ])

    # data over downsample level, truncation fraction and starting distance, stratified by masking
    if False:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_to_tensor(  #
            df.loc[(df["mask"] == "Every evaluation")],  #
            ordered_axes=["downsample_level", "truncation_percent", "starting_distance", "iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots(distances.size(0), distances.size(1))
        fig.subplots_adjust(left=0.05,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.95,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
        for k, dl in enumerate(axis_values["downsample_level"]):
            for j, tf in enumerate(axis_values["truncation_percent"]):
                for i, sd in enumerate(axis_values["starting_distance"]):
                    axes[k, j].plot(axis_values["iteration"], distances[k, j, i, :], label="s.d. {:.3f}".format(sd))
                    axes[k, j].set_title("d.l. {}; t.f. {:.3f}".format(dl, tf))
                    axes[k, j].set_xlabel("iteration")
                    axes[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[k, j].set_ylabel("distance from G.T.")
                    axes[k, j].set_ylim((0.0, None))
                    axes[k, j].legend()
        plt.show()

    # data over truncation fraction, stratified by masking
    if True:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_to_tensor(  #
            df.loc[(df["mask"] == "Every evaluation weighting zncc")],  #
            ordered_axes=["truncation_percent", "iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots()
        fig.subplots_adjust(left=0.05,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.95,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
        for j, tf in enumerate(axis_values["truncation_percent"]):
            axes.plot(axis_values["iteration"], distances[j, :], label="t.f. {:.3f}".format(tf))
            axes.set_xlabel("iteration")
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_ylabel("distance from G.T.")
            # axes.set_ylim((0.0, None))
            axes.legend()
        plt.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # for outputting PGFs
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["scatter.marker"] = 'x'
    # plt.rcParams["font.size"] = 22  # figures are includes in latex at quarte size, so 36 is desired size. matplotlib
    # scales up by 1.2 (God only knows why). 36 is tool big, however, so going a bit smaller than 30

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-l", "--load-dir", type=str, default="data/temp/program_truncation",
                        help="Directory in which to find the data files.")
    parser.add_argument("-w", "--which-dataset", type=str, default="latest",
                        help="Which dataset to plot. Either 'latest', or a timestamp in the format 'YYYY-MM-DD_hh-mm-ss'.")
    # parser.add_argument("-s", "--save-path", type=str, default="figures/truncation/measurement",
    #                     help="Set a directory in which to save the resulting figures..")
    parser.add_argument("-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    main(load_dir=pathlib.Path(args.load_dir), which_dataset=args.which_dataset, display=args.display)
