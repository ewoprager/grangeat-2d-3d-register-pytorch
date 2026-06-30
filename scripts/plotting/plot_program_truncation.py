import argparse
import itertools
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

from reg23_experiments.analysis.helpers import dataframe_rectangular_columns_to_tensor
from reg23_experiments.analysis.plot import separate_subplots
from reg23_experiments.utils import logs_setup

MPL_COLOURS = rcParams['axes.prop_cycle'].by_key()['color']


def get_color(i):
    return MPL_COLOURS[i % len(MPL_COLOURS)]


def var_to_string(variable_name: str, value: Any) -> str:
    if variable_name == "mask" or variable_name == "cropping" or variable_name == "sim_metric":
        return value
    elif variable_name == "xray_path":
        return pathlib.Path(value).name
    elif variable_name == "truncation_percent" or variable_name == "downsample_level":
        return f"{value}"
    elif variable_name == "starting_distance":
        return f"{value:.3f}"
    return f"<unknown variable '{variable_name}'>"


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


def main(  #
        *,  #
        load_dir: pathlib.Path,  #
        which_dataset: str,  #
        display: bool,  #
        save_figures: bool,  #
        save_directory: pathlib.Path,  #
        analysis_format: bool,  #
) -> None:
    assert load_dir.is_dir()

    # -----
    if analysis_format:
        plt.rcParams["font.size"] = 6
    else:
        # for outputting PGFs
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["scatter.marker"] = 'x'
        plt.rcParams[
            "font.size"] = 15  # figures are includes in latex at quarte size, so 36 is desired size. matplotlib    #
        # scales up by 1.2 (God only knows why). 36 is tool big, however, so going a bit smaller than 30

    # -----
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

    # -----
    # Reading in parquet data and concatenating
    df = pd.concat([  #
        pd.read_parquet(element)  #
        for element in instance_dir.iterdir()  #
        if element.stem.startswith("data") and element.suffix == ".parquet"  #
    ], ignore_index=True)
    distance_std_available = "distance_std" in df
    crop_size_available = "crop_width" in df and "crop_height" in df

    # -----
    # Reading in the variables
    variables_path = instance_dir / "variables.txt"
    assert variables_path.is_file()
    with open(variables_path, 'r') as file:
        variables_config = yaml.safe_load(file)
    assert "variables" in variables_config
    variables: list[str] = list(variables_config["variables"].keys())

    variable_hierachy: list[str] = ["mask", "truncation_percent", "cropping", "xray_path"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierachy)}
    variables = sorted(variables, key=lambda name: variable_importances[name], reverse=True)

    dense = not analysis_format

    if len(variables) == 1:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=variables + ["iteration"],  #
            value_column="distance"  #
        )
        if distance_std_available:
            distance_stds, _ = dataframe_rectangular_columns_to_tensor(  #
                df,  #
                ordered_axes=variables + ["iteration"],  #
                value_column="distance_std"  #
            )
        if crop_size_available:
            crop_widths, _ = dataframe_rectangular_columns_to_tensor(  #
                df.loc[df["iteration"] == 0],  #
                ordered_axes=variables,  #
                value_column="crop_width"  #
            )
            crop_heights, _ = dataframe_rectangular_columns_to_tensor(  #
                df.loc[df["iteration"] == 0],  #
                ordered_axes=variables,  #
                value_column="crop_height"  #
            )

        fig, axes = plt.subplots()
        if dense:
            fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                right=0.98,  # right margin
                                bottom=0.08,  # bottom margin
                                top=0.95,  # top margin
                                wspace=0.2,  # width space between columns
                                hspace=0.3  # height space between rows
                                )
        for i0, v0 in enumerate(axis_values[variables[0]]):
            axes.plot(axis_values["iteration"], distances[i0, :], label=f"{var_to_string(variables[0], v0)}",
                      color=get_color(i0))
            if distance_std_available:
                axes.errorbar(axis_values["iteration"], distances[i0, :], yerr=distance_stds[i0, :], fmt='x-',
                              capsize=4, color=get_color(i0))
            if crop_size_available:
                axes2 = axes.twinx()
                axes2.plot(axis_values["iteration"], crop_widths[i0, :], color=get_color(i0))

        axes.set_title(f"Dist. vs. iteration for different {variables[0]}")
        axes.set_xlabel("iteration")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.set_ylabel("distance from G.T.")
        axes.set_ylim((0.0, None))
        axes.legend()
        plt.show()

    if len(variables) == 2:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=variables + ["iteration"],  #
            value_column="distance"  #
        )

        fig, axes = plt.subplots(distances.size(0))
        if dense:
            fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                right=0.98,  # right margin
                                bottom=0.08,  # bottom margin
                                top=0.95,  # top margin
                                wspace=0.2,  # width space between columns
                                hspace=0.3  # height space between rows
                                )
        for i0, v0 in enumerate(axis_values[variables[0]]):
            for i1, v1 in enumerate(axis_values[variables[1]]):
                axes[i0].plot(axis_values["iteration"], distances[i0, i1, :],
                              label=f"{variables[1]}={var_to_string(variables[1], v1)}")
            axes[i0].set_title(f"{variables[0]}={var_to_string(variables[0], v0)}")
            axes[i0].set_xlabel("iteration")
            axes[i0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[i0].set_ylabel("distance from G.T.")
            axes[i0].set_ylim((0.0, None))
            axes[i0].legend()
        plt.show()

    if len(variables) == 3:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=variables + ["iteration"],  #
            value_column="distance"  #
        )

        fig, axes = plt.subplots(distances.size(0), distances.size(1))
        if dense:
            fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                right=0.98,  # right margin
                                bottom=0.08,  # bottom margin
                                top=0.95,  # top margin
                                wspace=0.2,  # width space between columns
                                hspace=0.3  # height space between rows
                                )
        for i0, v0 in enumerate(axis_values[variables[0]]):
            for i1, v1 in enumerate(axis_values[variables[1]]):
                for i2, v2 in enumerate(axis_values[variables[2]]):
                    axes[i0, i1].plot(axis_values["iteration"], distances[i0, i1, i2, :],
                                      label=f"{variables[2]}={var_to_string(variables[2], v2)}")
                axes[i0, i1].set_title(f"{variables[0]}={var_to_string(variables[0], v0)};{variables[1]}="
                                       f"{var_to_string(variables[1], v1)}")
                axes[i0, i1].set_xlabel("iteration")
                axes[i0, i1].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[i0, i1].set_ylabel("distance from G.T.")
                axes[i0, i1].set_ylim((0.0, None))
                axes[i0, i1].legend()
        plt.show()
        return

    if len(variables) == 4:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=variables + ["iteration"],  #
            value_column="distance"  #
        )

        # getting the median largest distance value
        ylim_upper = distances.amax(dim=-1).quantile(q=0.75)

        for i0, v0 in enumerate(axis_values[variables[0]]):
            fig, axes = plt.subplots(distances.size(1), distances.size(2), figsize=(13, 8))
            fig.suptitle(f"{variables[0]}={var_to_string(variables[0], v0)}")
            if dense:
                fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                    right=0.98,  # right margin
                                    bottom=0.08,  # bottom margin
                                    top=0.95,  # top margin
                                    wspace=0.2,  # width space between columns
                                    hspace=0.3  # height space between rows
                                    )
            for i1, v1 in enumerate(axis_values[variables[1]]):
                for i2, v2 in enumerate(axis_values[variables[2]]):
                    for i3, v3 in enumerate(axis_values[variables[3]]):
                        axes[i1, i2].plot(axis_values["iteration"], distances[i0, i1, i2, i3, :],
                                          label=f"{variables[3]}={var_to_string(variables[3], v3)}")
                    axes[i1, i2].set_title(f"{variables[1]}={var_to_string(variables[1], v1)};{variables[2]}="
                                           f"{var_to_string(variables[2], v2)}")
                    axes[i1, i2].set_xlabel("iteration")
                    axes[i1, i2].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[i1, i2].set_ylabel("distance from G.T.")
                    axes[i1, i2].set_ylim((0.0, ylim_upper))
                    axes[i1, i2].legend()
        plt.show()
        return

    # -----
    # data over downsample level, truncation fraction and starting distance, stratified by masking
    if len(variables) == 4 and variables[0] == "mask":
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df.loc[(df[variables[3]] == "Every evaluation")],  #
            ordered_axes=variables[:3] + ["iteration"],  #
            value_column="distance"  #
        )

        fig, axes = plt.subplots(distances.size(0), distances.size(1))
        fig.subplots_adjust(left=0.05,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.95,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
        for k, dl in enumerate(axis_values[variables[0]]):
            for j, tf in enumerate(axis_values[variables[1]]):
                for i, sd in enumerate(axis_values[variables[2]]):
                    axes[k, j].plot(axis_values["iteration"], distances[k, j, i, :], label="s.d. {:.3f}".format(sd))
                    axes[k, j].set_title("d.l. {}; t.f. {:.3f}".format(dl, tf))
                    axes[k, j].set_xlabel("iteration")
                    axes[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[k, j].set_ylabel("distance from G.T.")
                    axes[k, j].set_ylim((0.0, None))
                    axes[k, j].legend()
        plt.show()

    # data over truncation percent, masking and cropping
    if len(variables) == 3:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=variables + ["iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots(distances.size(0), distances.size(1))
        fig.subplots_adjust(left=0.05,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.95,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
        for k, cp in enumerate(axis_values[variables[0]]):
            for j, mk in enumerate(axis_values[variables[1]]):
                for i, tp in enumerate(axis_values[variables[2]]):
                    axes[k, j].plot(axis_values["iteration"], distances[k, j, i, :], label=f"t.p. {tp}")
                    axes[k, j].set_title("cp. {}; mk. {}".format(cp, mk))
                    axes[k, j].set_xlabel("iteration")
                    axes[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[k, j].set_ylabel("distance from G.T.")
                    axes[k, j].set_ylim((0.0, None))
                    axes[k, j].legend()
        plt.show()

    # data over truncation fraction, stratified by cropping
    if len(variables) == 2 and variables[1] == "cropping":
        for crop in ["None", "full_depth_drr", "nonzero_drr"]:
            # for mask in ["None", "Every evaluation", "Every evaluation weighting zncc"]:
            # converting to a tensor, with an axis per variable
            distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
                df.loc[(df["cropping"] == crop)],  #
                # df.loc[(df["mask"] == mask)],  #
                ordered_axes=[variables[0], "iteration"],  #
                value_column="distance")

            fig, axes = plt.subplots()
            fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                right=0.98,  # right margin
                                bottom=0.08,  # bottom margin
                                top=0.95,  # top margin
                                wspace=0.2,  # width space between columns
                                hspace=0.3  # height space between rows
                                )
            for j, tf in enumerate(axis_values[variables[0]]):
                axes.plot(axis_values["iteration"], distances[j, :], label="t.f. {:.3f}".format(tf))
                axes.set_xlabel("iteration")
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                axes.set_ylabel("distance from G.T.")
                # axes.set_ylim((0.0, None))
                axes.legend()
                axes.set_title(crop)

    # data over downsample level and truncation fraction, stratified by masking
    if False:
        starting_distance = df["starting_distance"].values[0]
        iteration_count = df["iteration_count"].values[0]
        distances, axis_values = dataframe_to_tensor(  #
            df.loc[(df["mask"] == "None")],  #
            ordered_axes=["downsample_level", "truncation_percent", "iteration"],  #
            value_column="distance")
        fig, axes = (separate_subplots if save_figures else plt.subplots)(3, distances.size(0))
        if not save_figures:
            fig.subplots_adjust(left=0.05,  # margin on left side of figure
                                right=0.98,  # right margin
                                bottom=0.08,  # bottom margin
                                top=0.95,  # top margin
                                wspace=0.2,  # width space between columns
                                hspace=0.3)  # height space between rows
        for k, mask in enumerate(["None", "Every evaluation", "Every evaluation weighting zncc"]):
            # converting to a tensor, with an axis per variable
            distances, axis_values = dataframe_to_tensor(  #
                df.loc[(df["mask"] == mask)],  #
                ordered_axes=["downsample_level", "truncation_percent", "iteration"],  #
                value_column="distance")
            max_truncation = axis_values["truncation_percent"].max()
            for j, dl in enumerate(axis_values["downsample_level"]):
                for i, tf in enumerate(axis_values["truncation_percent"]):
                    axes[k, j].plot(  #
                        axis_values["iteration"] + 1,  #
                        distances[j, i, :],  #
                        label="t.f. {:.3f}".format(tf.item()),  #
                        color=((tf / max_truncation).item(), 1.0 - (tf / max_truncation).item(), 0.0))
                axes[k, j].set_xlabel("iteration")
                axes[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[k, j].set_ylabel("distance from G.T.")
                axes[k, j].set_ylim((0.0, starting_distance))
                if not save_figures:
                    axes[k, j].set_title(f"Mask: {mask}, d.l. {dl}")
                axes[k, j].legend()
        if save_figures:
            for k, j in itertools.product(range(3), range(len(axis_values["downsample_level"]))):
                fig[k, j].tight_layout()
                fig[k, j].savefig(save_directory / f"convergence_{k}_{j}.pgf")

        fig, axes = plt.subplots()
        for j, mask in enumerate(["None", "Every evaluation", "Every evaluation weighting zncc"]):
            distances = df.loc[  #
                (df["mask"] == mask) &  #
                (df["downsample_level"] == 1) &  #
                (df["iteration"] == iteration_count - 1)  #
                ].set_index("truncation_percent")["distance"].sort_index()
            axes.plot(distances, label=mask)
        axes.set_xlabel("Truncation percent")
        axes.set_ylabel(f"Converged distance after {iteration_count} iterations")
        axes.set_ylim((0.0, starting_distance))
        axes.legend()
        if save_figures:
            fig.tight_layout()
            fig.savefig(save_directory / "converged_against_truncation.pgf")
        if display:
            plt.show()

    # data over similarity metric only
    if False:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_to_tensor(  #
            df,  #
            ordered_axes=["sim_metric", "iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots()
        for i, sm in enumerate(axis_values["sim_metric"]):
            axes.plot(axis_values["iteration"], distances[i, :], label=sm)
            axes.set_xlabel("iteration")
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_ylabel("distance from G.T.")
            # axes.set_ylim((0.0, None))
            axes.legend()
        if display:
            plt.show()

    # data over cropping only
    if False:
        # converting to a tensor, with an axis per variable
        distances, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df,  #
            ordered_axes=["cropping", "iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots()
        for i, cropping_desc in enumerate(axis_values["cropping"]):
            axes.plot(axis_values["iteration"], distances[i, :], label=cropping_desc)
            axes.set_xlabel("iteration")
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_ylabel("distance from G.T.")
            # axes.set_ylim((0.0, None))
            axes.legend()

    if display:
        plt.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-l", "--load-dir", type=str, default="experimental_results/program_truncation",
                        help="Directory in which to find the data files.")
    parser.add_argument("-w", "--which-dataset", type=str, default="latest",
                        help="Which dataset to plot. Either 'latest', or a timestamp in the format "
                             "'YYYY-MM-DD_hh-mm-ss'.")
    parser.add_argument("-s", "--save-dir", type=str, default="figures/truncation/program_truncation",
                        help="Set a directory in which to save the resulting figures.")
    parser.add_argument("-r", "--save-figures", action="store_true",
                        help="Format plots appropriately for using in a report and save them in the save directory.")
    parser.add_argument("-d", "--display", action="store_true", help="Display/plot the resulting data.")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="Format the plots for analysis, rather than PGF plot generation.")
    args = parser.parse_args()

    main(load_dir=pathlib.Path(args.load_dir), which_dataset=args.which_dataset, display=args.display,
         save_figures=args.save_figures, save_directory=pathlib.Path(args.save_dir), analysis_format=args.analysis)
