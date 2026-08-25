import argparse
import itertools
import pathlib

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from reg23_experiments.analysis.format import get_colour, latex_escape, set_mpl_latex_options, var_to_string
from reg23_experiments.analysis.manipulation import CartesianZippedTensors, dataframe_rectangular_columns_to_tensor, \
    dataframe_to_cartesian_zipped_tensors
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.image import read_dicom
from reg23_experiments.io.sitk import load_ct_series
from reg23_experiments.ops import geometry
from reg23_experiments.utils import logs_setup

l_cache = dict()
theta_cache = dict()


def ct_xray_to_h_linear(  #
        *,  #
        saved_transformations: pd.DataFrame,  #
        xray_path: str | pathlib.Path,  #
        ct_path: str | pathlib.Path,  #
        ct_series_uid: str,  #
) -> float:
    # CT
    ct_key = str(ct_path) + ct_series_uid
    if ct_key in l_cache:
        l = l_cache[ct_key]
    else:
        ct: sitk.Image | Error = load_ct_series(ct_path, ct_series_uid)
        if isinstance(ct, Error):
            raise Exception(f"Failed to open CT from path '{ct_path}': {ct.description}")
        l = float(ct.GetSize()[1]) * ct.GetSpacing()[1]
        l_cache[ct_key] = l
    # X-ray
    if xray_path in theta_cache:
        theta = theta_cache[xray_path]
    else:
        xray_sop_instance_uid = read_dicom(xray_path)["uid"]
        idx = (xray_sop_instance_uid, "gold_standard")
        row = saved_transformations.loc[idx]
        t = Transformation.from_vector(  #
            torch.tensor([row[f"x{i}"] for i in range(6)], dtype=torch.float64)  #
        )  #
        theta = abs(geometry.axis_angle_extract_axis(t.rotation, torch.tensor([1.0, 0.0, 0.0])) - 0.5 * np.pi)
        theta_cache[xray_path] = theta
    return l * np.sin(theta)


def save_legend_figure(axes, path: pathlib.Path) -> None:
    handles, labels = axes.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(2, 2))
    legend = legend_fig.legend(handles, labels, loc="center", ncol=1,  # or however many columns you want
                               frameon=False)
    legend_fig.canvas.draw()
    legend_fig.savefig(path, bbox_inches="tight", bbox_extra_artists=[legend], )


def grid_of_plots_figure(  #
        *,  #
        cartesian_axes_values: list[tuple[str, np.ndarray]],  #
        zipped_axis_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        dense: bool = False,  #
        ylim: tuple[float, float] | None = None,  #
        legend: bool = True,  #
) -> tuple[Figure, np.ndarray]:
    axes_threshold = -1 if zipped_axis_values else -2
    # check arguments
    assert 1 <= len(cartesian_axes_values) <= 2 + abs(axes_threshold)
    axes_lengths = [len(v) for _, v in cartesian_axes_values]
    if zipped_axis_values:
        zipped_length = len(zipped_axis_values[0][1])
        assert all(len(t[1]) == zipped_length for t in zipped_axis_values)
        axes_lengths += [zipped_length]
    assert dependent_values.size() == torch.Size(axes_lengths)
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()
    # figure and axes
    fig, axes = plt.subplots(*dependent_values.size()[:-2], figsize=(6, 6) if dense else (13, 8))
    axes = np.array(axes)
    if dense:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9, wspace=0.2, hspace=0.3)

    x_label = cartesian_axes_values[-1][0]
    x_values = cartesian_axes_values[-1][1]

    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in cartesian_axes_values[:axes_threshold]]):
        axis_index = tuple(i for i, _ in index_value_pairs)

        title = ";".join([  #
            f"{cartesian_axes_values[i][0]}={var_to_string(cartesian_axes_values[i][0], w)}"  #
            for i, w in enumerate([v for _, v in index_value_pairs])  #
        ])  #

        if zipped_axis_values:
            zipped_variables = [t[0] for t in reversed(zipped_axis_values)]
            for i, zipped_values in enumerate(zip(*[t[1] for t in reversed(zipped_axis_values)])):
                dependent_index = axis_index + (slice(None), i)
                line_label = ";".join(  #
                    f"{var}={var_to_string(var, val)}"  #
                    for var, val in zip(zipped_variables, zipped_values)  #
                )
                axes[axis_index].plot(  #
                    x_values,  #
                    dependent_values[dependent_index],  #
                    label=line_label,  #
                    color=get_colour(i),  #
                )
                if dependent_errors is not None:
                    axes[axis_index].errorbar(  #
                        x_values,  #
                        dependent_values[dependent_index],  #
                        yerr=dependent_errors[dependent_index],  #
                        fmt='x-',  #
                        capsize=4,  #
                        color=get_colour(i),  #
                    )
        else:
            line_variable = cartesian_axes_values[-2][0]
            line_values = cartesian_axes_values[-2][1]
            for i, line_value in enumerate(line_values):
                dependent_index = axis_index + (i, slice(None))
                line_label = f"{line_variable}={var_to_string(line_variable, line_value)}"
                axes[axis_index].plot(  #
                    x_values,  #
                    dependent_values[dependent_index],  #
                    label=line_label,  #
                    color=get_colour(i),  #
                )
                if dependent_errors is not None:
                    axes[axis_index].errorbar(  #
                        x_values,  #
                        dependent_values[dependent_index],  #
                        yerr=dependent_errors[dependent_index],  #
                        fmt='x-',  #
                        capsize=4,  #
                        color=get_colour(i),  #
                    )

        axes[axis_index].set_xlabel(latex_escape(x_label))
        axes[axis_index].set_title(latex_escape(title))
        axes[axis_index].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[axis_index].set_ylabel(latex_escape(dependent_variable))
        if ylim is not None:
            axes[axis_index].set_ylim(ylim)
        if legend:
            axes[axis_index].legend()

        if False:
            if isinstance(cartesian_axes_values[-2][1][0], float):
                v_min = np.min(cartesian_axes_values[-2][1])
                v_max = np.max(cartesian_axes_values[-2][1])
            for j, v in enumerate(independent_values[-2][1]):
                dependent_index = axis_index + (j,)
                if False and isinstance(v, float):
                    r = (v - v_min) / (v_max - v_min)
                    colour = (r, 1.0 - r, 0.0)
                else:
                    colour = get_colour(j)
                axes[axis_index].plot(  #
                    independent_values[-1][1],  #
                    dependent_values[*dependent_index, :],  #
                    label=latex_escape(f"{independent_values[-2][0]}={var_to_string(independent_values[-2][0], v)}"),  #
                    color=colour,  #
                )
                if dependent_errors is not None:
                    axes[axis_index].errorbar(  #
                        independent_values[-1][1],  #
                        dependent_values[*dependent_index, :],  #
                        yerr=dependent_errors[*dependent_index, :],  #
                        fmt='x-',  #
                        capsize=4,  #
                        color=colour  #
                    )
            axes[axis_index].set_xlabel(latex_escape(independent_values[-1][0]))
            axes[axis_index].set_title(latex_escape(  #
                ";".join([  #
                    f"{independent_values[i][0]}={var_to_string(independent_values[i][0], w)}"  #
                    for i, w in enumerate([v for _, v in index_value_pairs])  #
                ])  #
            ))
            axes[axis_index].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[axis_index].set_ylabel(latex_escape(dependent_variable))
            if ylim is not None:
                axes[axis_index].set_ylim(ylim)
            if legend:
                axes[axis_index].legend()
    return fig, axes


def plot_grid_figures(  #
        *,  #
        cartesian_axes_values: list[tuple[str, np.ndarray]],  #
        zipped_axis_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        dense: bool = False,  #
        save_to: pathlib.Path | None = None,  #
        legend_separate: bool = False,  #
) -> None:
    axes_threshold = -3 if zipped_axis_values else -4

    # getting the median largest distance value
    ylim: tuple[float, float] | None = (0.0, dependent_values.amax(dim=-1).quantile(q=0.5).item()) if len(
        cartesian_axes_values) > 2 else None

    # check arguments
    if len(cartesian_axes_values) <= abs(axes_threshold):
        fig, axes = grid_of_plots_figure(  #
            cartesian_axes_values=cartesian_axes_values[axes_threshold:],  #
            zipped_axis_values=zipped_axis_values,  #
            dependent_variable=dependent_variable,  #
            dependent_values=dependent_values,  #
            dependent_errors=dependent_errors,  #
            dense=dense,  #
            ylim=ylim,  #
            legend=not legend_separate,  #
        )
        plt.show()
        return

    axes_lengths = [len(v) for _, v in cartesian_axes_values]
    if zipped_axis_values:
        zipped_length = len(zipped_axis_values[0][1])
        assert all(len(t[1]) == zipped_length for t in zipped_axis_values)
        axes_lengths += [zipped_length]
    assert dependent_values.size() == torch.Size(axes_lengths)
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()

    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in cartesian_axes_values[:axes_threshold]]):
        dependent_index = tuple(i for i, _ in index_value_pairs)
        fig, axes = grid_of_plots_figure(  #
            cartesian_axes_values=cartesian_axes_values[axes_threshold:],  #
            zipped_axis_values=zipped_axis_values,  #
            dependent_variable=dependent_variable,  #
            dependent_values=dependent_values[*dependent_index],  #
            dependent_errors=None if dependent_errors is None else dependent_errors[*dependent_index],  #
            dense=dense,  #
            ylim=ylim,  #
            legend=not legend_separate,  #
        )
        fig.suptitle(latex_escape(  #
            ";".join([  #
                f"{cartesian_axes_values[i][0]}={var_to_string(cartesian_axes_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ])  #
        ))
        if save_to is not None:
            fig.savefig(save_to / ("_".join(  #
                f"{cartesian_axes_values[i][0]}-{j}"  #
                for i, j in enumerate([k for k, _ in index_value_pairs])  #
            ) + ".pgf"))
    if legend_separate:
        save_legend_figure(axes.flatten()[0], save_to / "legend.pgf")
    plt.show()


def convergence_curve_to_accuracy(  #
        distances: torch.Tensor,  #
        distance_stds: torch.Tensor,  #
        iteration_dim: int,  #
) -> torch.Tensor:
    index = [slice(None)] * distances.ndim
    index[iteration_dim] = -1
    return (distances + distance_stds)[tuple(index)]


def main(  #
        *,  #
        load_dir: pathlib.Path,  #
        which_datasets: list[str],  #
        display: bool,  #
        save_to: pathlib.Path | None,  #
        analysis_format: bool,  #
) -> None:
    assert load_dir.is_dir()
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)

    # -----
    if analysis_format:
        plt.rcParams["font.size"] = 6
    else:
        set_mpl_latex_options()
    dense = not analysis_format

    # -----
    # Getting the latest data instance if desired
    if not which_datasets:
        subdirs = []
        for element in load_dir.iterdir():
            if not element.is_dir():
                continue
            subdirs.append(str(element.stem))
        subdirs.sort()
        which_datasets = [subdirs[-1]]
    instance_dirs: list[pathlib.Path] = [load_dir / name for name in which_datasets]
    for d in instance_dirs:
        assert d.is_dir()

    # -----
    # Reading in parquet data and concatenating
    df = pd.concat([  #
        pd.read_parquet(element)  #
        for element in itertools.chain.from_iterable([d.iterdir() for d in instance_dirs])  #
        if element.stem.startswith("data") and element.suffix == ".parquet"  #
    ], ignore_index=True)
    distance_std_available = "distance_std" in df
    crop_size_available = "crop_width" in df and "crop_height" in df

    if False:
        # -----
        # Reading in parquet data and concatenating
        df_extra = pd.concat([  #
            pd.read_parquet(element)  #
            for element in pathlib.Path(
                "/home/eprager/Projects/grangeat-2d-3d-register-pytorch/experimental_results/program_truncation/2026"
                "-08-25_00-42-02_capture_range").iterdir()
            #
            if element.stem.startswith("data") and element.suffix == ".parquet"  #
        ], ignore_index=True)
        df_extra = df_extra[df_extra["downsample_level"] == 2]
        df = pd.concat([df, df_extra], ignore_index=True)
        df["xray_path"] = df["xray_path"].map(lambda p: pathlib.Path(p).name)
        df["ct_path"] = df["ct_path"].map(lambda p: pathlib.Path(p).name)

    # -----
    # Reading in the variables
    variables_path = instance_dirs[0] / "variables.txt"
    assert variables_path.is_file()
    with open(variables_path, 'r') as file:
        variables_config = yaml.safe_load(file)
    # assert "variables" in variables_config
    # variables: list[str] = list(variables_config["variables"].keys())
    assert "cartesian" in variables_config
    cartesian_variables: list[str] = list(variables_config["cartesian"].keys())

    ## !!!
    # cartesian_variables.append("sim_metric")
    ## !!!

    variable_hierarchy: list[str] = ["starting_distance", "sim_metric", "weight_alpha", "apply_weighting",
                                     "iterations_per_crop_update", "cropping", "cropping_method", "truncation_percent",
                                     "apply_scaling", "iterations_per_weight_update", "crop_expand", "mask",
                                     "desired_h_valid", "downsample_level", "xray_path"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierarchy)}
    cartesian_variables = sorted(  #
        cartesian_variables,  #
        key=lambda name: variable_importances[name] if name in variable_importances else len(variable_hierarchy),  #
        reverse=True  #
    )

    dependent_variables = ["distance"]
    if distance_std_available:
        dependent_variables.append("distance_std")

    if crop_size_available:
        df = df.drop(columns=["crop_width", "crop_height"])

    czt: CartesianZippedTensors = dataframe_to_cartesian_zipped_tensors(  #
        df,  #
        cartesian_variables=cartesian_variables + ["iteration"],  #
        dependent_variables=dependent_variables,  #
    )

    if False:
        plot_grid_figures(  #
            cartesian_axes_values=czt.cartesian_axes_values,  #
            zipped_axis_values=czt.zipped_axis_values,  #
            dependent_variable="distance from gold standard",  #
            dependent_values=czt.dependent_variable_tensors["distance"],  #
            dependent_errors=czt.dependent_variable_tensors["distance_std"] if distance_std_available else None,  #
            dense=dense,  #
        )
    else:
        czt = czt.reduce("iteration", method="take_last")
        plot_grid_figures(  #
            cartesian_axes_values=czt.cartesian_axes_values,  #
            zipped_axis_values=czt.zipped_axis_values,  #
            dependent_variable="distance_at_last_iteration",  #
            dependent_values=czt.dependent_variable_tensors["distance_at_last_iteration"],  #
            # dependent_errors=czt.dependent_variable_tensors["distance_std"] if distance_std_available else None,  #
            dense=dense,  #
        )

    return

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
    if "crop_expand" not in variables or True:
        if True:
            independent_variables = axis_values
            dependent_variable = "distance from gold-standard"
            dependent_values = distances
            dependent_errors = distance_stds if distance_std_available else None
        else:
            assert (distance_std_available, "Distance standard deviations are required for accuracy metric.")
            independent_variables = axis_values[:-1]
            dependent_variable = "accuracy"
            dependent_values = convergence_curve_to_accuracy(distances, distance_stds, -1)
            dependent_errors = None
        if len(independent_variables) == 1:
            plt.plot(independent_variables[0][1], dependent_values)
            plt.xlabel(f"{independent_variables[0][0]}")
            plt.ylabel(f"{dependent_variable}")
            plt.show()
        else:
            plot_grid_figures(  #
                independent_values=independent_variables,  #
                dependent_variable=dependent_variable,  #
                dependent_values=dependent_values,  #
                dependent_errors=dependent_errors,  #
                dense=dense,  #
                save_to=save_to,  #
                legend_separate=False,  #
            )
    else:
        dimension = variables.index("crop_expand")
        best_crop_expand_indices = distances[..., -1].argmin(dim=dimension, keepdim=True)
        new_size = distances.amin(dim=dimension, keepdim=True).size()
        distances_chosen = distances.gather(  #
            dim=dimension,  #
            index=best_crop_expand_indices.unsqueeze(-1).expand(new_size)  #
        ).squeeze(dimension)
        if distance_std_available:
            distance_stds_chosen = distance_stds.gather(  #
                dim=dimension,  #
                index=best_crop_expand_indices.unsqueeze(-1).expand(new_size)  #
            ).squeeze(dimension)
        new_axis_values = [(name, array) for name, array in axis_values if name != "crop_expand"]

        ylim: tuple[float, float] | None = (0.0, distances_chosen.amax(dim=-1).quantile(q=0.75).item()) if len(
            new_axis_values) > 2 else None

        for index_value_pairs in itertools.product(*[enumerate(v) for _, v in new_axis_values[:-3]]):
            dependent_index = () if index_value_pairs == () else tuple(i for i, _ in index_value_pairs)
            fig, axes = grid_of_plots_figure(  #
                independent_values=new_axis_values[-3:],  #
                dependent_variable="distance from gold-standard",  #
                dependent_values=distances_chosen[*dependent_index],  #
                dependent_errors=distance_stds_chosen[*dependent_index] if distance_std_available else None,  #
                dense=dense,  #
                ylim=ylim,  #
            )
            fig.suptitle(latex_escape(  #
                ";".join([  #
                    f"{new_axis_values[i][0]}={var_to_string(new_axis_values[i][0], w)}"  #
                    for i, w in enumerate([v for _, v in index_value_pairs])  #
                ])  #
            ))
            if save_to is not None:
                fig.savefig(save_to / ("_".join(  #
                    f"{new_axis_values[i][0]}-{j}"  #
                    for i, j in enumerate([k for k, _ in index_value_pairs])  #
                ) + ".pgf"))
        plt.show()

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


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-l", "--load-dir", type=str, default="experimental_results/program_truncation",
                        help="Directory in which to find the data files.")
    parser.add_argument("-w", "--which-datasets", type=str, nargs='+',
                        help="Which datasets to plot, given as timestamps in the format "
                             "'YYYY-MM-DD_hh-mm-ss'. If not provided, the latest dataset will be used.")
    parser.add_argument("-s", "--save-to", type=str, default=None,
                        help="Set a directory in which to save the resulting figures.")
    parser.add_argument("-d", "--display", action="store_true", help="Display/plot the resulting data.")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="Format the plots for analysis, rather than PGF plot generation.")
    args = parser.parse_args()

    main(  #
        load_dir=pathlib.Path(args.load_dir),  #
        which_datasets=args.which_datasets,  #
        display=args.display,  #
        save_to=None if args.save_to is None else pathlib.Path(args.save_to),  #
        analysis_format=args.analysis,  #
    )
