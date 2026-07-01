import argparse
import itertools
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import rcParams
from matplotlib.figure import Figure
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
    elif variable_name == "crop_expand":
        return f"{value:.1f}"
    try:
        return str(value)
    except Exception:
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


def grid_of_plots_figure(  #
        *,  #
        independent_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        dense: bool = False,  #
        ylim: tuple[float, float] | None = None,  #
) -> tuple[Figure, np.ndarray]:
    # check arguments
    assert 2 <= len(independent_values) <= 4
    assert dependent_values.size() == torch.Size([len(v) for _, v in independent_values])
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()
    # figure and axes
    fig, axes = plt.subplots(*dependent_values.size()[:-2], figsize=(13, 8))
    axes = np.array(axes)
    if dense:
        fig.subplots_adjust(left=0.05,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.95,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in independent_values[:-2]]):
        axis_index = () if index_value_pairs == () else tuple(i for i, _ in index_value_pairs)
        for j, v in enumerate(independent_values[-2][1]):
            dependent_index = axis_index + (j,)
            axes[axis_index].plot(  #
                independent_values[-1][1],  #
                dependent_values[*dependent_index, :],  #
                label=f"{independent_values[-2][0]}={var_to_string(independent_values[-2][0], v)}",  #
                color=get_color(j),  #
            )
            if dependent_errors is not None:
                axes[axis_index].errorbar(  #
                    independent_values[-1][1],  #
                    dependent_values[*dependent_index, :],  #
                    yerr=dependent_errors[*dependent_index, :],  #
                    fmt='x-',  #
                    capsize=4,  #
                    color=get_color(j)  #
                )
        axes[axis_index].set_xlabel(independent_values[-1][0])
        axes[axis_index].set_title(  #
            ";".join([  #
                f"{independent_values[i][0]}={var_to_string(independent_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ])  #
        )
        axes[axis_index].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[axis_index].set_ylabel(dependent_variable)
        if ylim is not None:
            axes[axis_index].set_ylim(ylim)
        axes[axis_index].legend()
    return fig, axes


def plot_grid_figures(  #
        *,  #
        independent_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        dense: bool = False,  #
) -> None:
    # check arguments
    assert 2 <= len(independent_values)
    assert dependent_values.size() == torch.Size([len(v) for _, v in independent_values])
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()
    # getting the median largest distance value
    ylim: tuple[float, float] | None = (0.0, dependent_values.amax(dim=-1).quantile(q=0.75).item()) if len(
        independent_values) > 2 else None

    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in independent_values[:-4]]):
        dependent_index = () if index_value_pairs == () else tuple(i for i, _ in index_value_pairs)
        fig, axes = grid_of_plots_figure(  #
            independent_values=independent_values[-4:],  #
            dependent_variable=dependent_variable,  #
            dependent_values=dependent_values[*dependent_index],  #
            dependent_errors=None if dependent_errors is None else dependent_errors[*dependent_index],  #
            dense=dense,  #
            ylim=ylim,  #
        )
        fig.suptitle(  #
            ";".join([  #
                f"{independent_values[i][0]}={var_to_string(independent_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ])  #
        )
    plt.show()


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

    variable_hierarchy: list[str] = ["crop_expand", "mask", "cropping", "xray_path",
                                     "truncation_percent"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierarchy)}
    variables = sorted(  #
        variables,  #
        key=lambda name: variable_importances[name] if name in variable_importances else len(variable_hierarchy),  #
        reverse=True  #
    )

    dense = not analysis_format

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
    plot_grid_figures(  #
        independent_values=axis_values,  #
        dependent_variable="distance from G.T.",  #
        dependent_values=distances,  #
        dependent_errors=distance_stds if distance_std_available else None,  #
        dense=dense,  #
    )

    if "xray_path" in variables and crop_size_available:
        # crop_widths, axis_values = dataframe_rectangular_columns_to_tensor(  #
        #     df.loc[df["iteration"] == 0],  #
        #     ordered_axes=variables,  #
        #     value_column="crop_width"  #
        # )
        crop_heights, axis_values = dataframe_rectangular_columns_to_tensor(  #
            df.loc[df["iteration"] == 0],  #
            ordered_axes=variables,  #
            value_column="crop_height"  #
        )

        invariant_variables = [  #
            "crop_expand",  #
            "mask"  #
        ]  # crop expand is applied after measuring, so it is truly invariant

        for name in invariant_variables:
            try:
                i = variables.index(name)
            except ValueError:
                continue
            axis_values = [e for e in axis_values if e[0] != name]
            # crop_widths = crop_widths.mean(dim=i)
            crop_heights = crop_heights.mean(dim=i)

        # crop_values = torch.stack((crop_widths, crop_heights), dim=-2)
        # axis_values.insert(-1, ("crop dir", np.array(["width", "height"])))

        plot_grid_figures(  #
            independent_values=axis_values,  #
            dependent_variable="crop height [mm]",  #
            dependent_values=crop_heights,  #
            dense=dense,  #
        )

    return

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
