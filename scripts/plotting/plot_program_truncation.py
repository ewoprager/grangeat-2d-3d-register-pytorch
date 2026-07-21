import argparse
import itertools
import pathlib
from typing import Any

import matplotlib

matplotlib.use("QtAgg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn
import torch
import yaml
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from reg23_experiments.analysis.helpers import dataframe_rectangular_columns_to_tensor
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.data.transformation_save_data import TransformationSaveData
from reg23_experiments.io.image import read_dicom
from reg23_experiments.io.save_data import load_latest_save
from reg23_experiments.io.sitk import load_ct_series
from reg23_experiments.ops import geometry
from reg23_experiments.utils import logs_setup
from reg23_experiments.utils.console_logging import tqdm

MPL_COLOURS = rcParams['axes.prop_cycle'].by_key()['color']


def get_colour(i):
    return MPL_COLOURS[i % len(MPL_COLOURS)]


def latex_escape(s: str) -> str:
    return (  #
        s.replace("\\", r"\textbackslash{}")  #
        .replace("_", r"\_")  #
        .replace("%", r"\%")  #
        .replace("&", r"\&")  #
        .replace("#", r"\#")  #
        .replace("{", r"\{")  #
        .replace("}", r"\}")  #
    )


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


def var_to_string(variable_name: str, value: Any) -> str:
    if variable_name == "cropping" or variable_name == "sim_metric":
        return value
    elif variable_name == "mask":
        if value == "None":
            return "no"
        elif value == "Every evaluation weighting zncc":
            return "yes"
        else:
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
        legend: bool = True,  #
) -> tuple[Figure, np.ndarray]:
    # check arguments
    assert 2 <= len(independent_values) <= 4
    assert dependent_values.size() == torch.Size([len(v) for _, v in independent_values])
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()
    # figure and axes
    fig, axes = plt.subplots(*dependent_values.size()[:-2], figsize=(6, 6) if dense else (13, 8))
    axes = np.array(axes)
    if dense:
        fig.subplots_adjust(left=0.08,  # margin on left side of figure
                            right=0.98,  # right margin
                            bottom=0.08,  # bottom margin
                            top=0.9,  # top margin
                            wspace=0.2,  # width space between columns
                            hspace=0.3  # height space between rows
                            )
    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in independent_values[:-2]]):
        axis_index = () if index_value_pairs == () else tuple(i for i, _ in index_value_pairs)
        if isinstance(independent_values[-2][1][0], float):
            v_min = np.min(independent_values[-2][1])
            v_max = np.max(independent_values[-2][1])
        for j, v in enumerate(independent_values[-2][1]):
            dependent_index = axis_index + (j,)
            if isinstance(v, float):
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
        independent_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        dense: bool = False,  #
        save_to: pathlib.Path | None = None,  #
        legend_separate: bool = False,  #
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
            legend=not legend_separate,  #
        )
        fig.suptitle(latex_escape(  #
            ";".join([  #
                f"{independent_values[i][0]}={var_to_string(independent_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ])  #
        ))
        if save_to is not None:
            fig.savefig(save_to / ("_".join(  #
                f"{independent_values[i][0]}-{j}"  #
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
        fit: bool = False,  #
) -> None:
    assert load_dir.is_dir()
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)

    # -----
    if analysis_format:
        plt.rcParams["font.size"] = 6
    else:
        # for outputting PGFs
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["scatter.marker"] = 'x'
        plt.rcParams[
            "font.size"] = 11  # figures are includes in latex at quarte size, so 36 is desired size. matplotlib    #
        # scales up by 1.2 (God only knows why). 36 is tool big, however, so going a bit smaller than 30
        rcParams["pgf.texsystem"] = "pdflatex"

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

    # -----
    # Reading in the variables
    variables_path = instance_dirs[0] / "variables.txt"
    assert variables_path.is_file()
    with open(variables_path, 'r') as file:
        variables_config = yaml.safe_load(file)
    assert "variables" in variables_config
    variables: list[str] = list(variables_config["variables"].keys())

    variable_hierarchy: list[str] = ["crop_expand", "mask", "cropping", "truncation_percent", "desired_h_valid",
                                     "xray_path"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierarchy)}
    variables = sorted(  #
        variables,  #
        key=lambda name: variable_importances[name] if name in variable_importances else len(variable_hierarchy),  #
        reverse=True  #
    )

    dense = not analysis_format

    if not fit:
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
        if "crop_expand" not in variables or False:
            if False:
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
    else:
        # Fit a model to the data

        # -----
        # Load all saved transformations; these are searched through for ground truth alignments
        res: tuple[pathlib.Path, TransformationSaveData, int] | Error = load_latest_save(  #
            TransformationSaveData,  #
            save_directory=pathlib.Path("data/app_transformation_save_data")  #
        )
        if isinstance(res, Error):
            raise RuntimeError(f"Failed to load saved transformation: {res.description}")
        _, transformation_save_data, _ = res
        saved_transformations: pd.DataFrame = transformation_save_data.get_data()
        logger.info(f"Saved transformation data:\n{saved_transformations.to_string()}")

        assert (distance_std_available, "Distance standard deviations are required for accuracy metric.")
        # Collapse to just last iteration
        accuracy_df = df[df["iteration"] == df["iteration"].max()].drop(columns=["iteration"])
        # Remove unnecessary dependent variable columns
        accuracy_df.drop(columns=["crop_width", "crop_height"], inplace=True)
        # CT and X-ray paths to h_linear
        accuracy_df["h_linear"] = [  #
            ct_xray_to_h_linear(  #
                saved_transformations=saved_transformations,  #
                xray_path=xray_path,  #
                ct_path=ct_path,  #
                ct_series_uid=ct_series_uid,  #
            )  #
            for xray_path, ct_path, ct_series_uid in tqdm(  #
                zip(accuracy_df["xray_path"], accuracy_df["ct_path"], accuracy_df["ct_series_uid"]),  #
                desc="Calculating h_linear"  #
            )  #
        ]
        accuracy_df.drop(columns=["xray_path", "ct_path", "ct_series_uid"], inplace=True)
        # Drop columns for constant variables
        accuracy_df = accuracy_df.drop(columns=[  #
            col for col in  #
            accuracy_df.columns[accuracy_df.nunique() == 1]  #
        ])

        print(accuracy_df.to_string())

        if "sample_count_per_distance" in accuracy_df.columns:
            accuracy_df.drop(columns=["sample_count_per_distance"], inplace=True)

        # -----
        # Gaussian Process Regression
        # Get the dependent value vector
        y: np.ndarray = accuracy_df["distance"].to_numpy()
        y_sigma: np.ndarray = accuracy_df["distance_std"].to_numpy()
        # Get the independent value vectors as a matrix
        independent_variables: list[str] = ["desired_h_valid", "h_linear", "crop_expand"]
        X: np.ndarray = accuracy_df[independent_variables].to_numpy()
        indices_in_slice = (accuracy_df["crop_expand"] == 0.0).to_numpy()
        gpr = sklearn.gaussian_process.GaussianProcessRegressor(alpha=np.square(y_sigma)).fit(X, y)

        h_valids = np.linspace(20.0, 35.0, 50)
        h_linears = np.linspace(15.0, 80.0, 50)
        h_valids, h_linears = np.meshgrid(h_valids, h_linears)
        values = {  #
            "desired_h_valid": h_valids.flatten(),  #
            "h_linear": h_linears.flatten(),  #
            "crop_expand": np.zeros_like(h_valids.flatten()),  #
        }
        model_values, model_stds = gpr.predict(np.stack([values[name] for name in independent_variables], axis=1), return_std=True)
        model_values = model_values.reshape(h_valids.shape)
        model_stds = model_stds.reshape(h_valids.shape)
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
        axes.plot_surface(h_linears, h_valids, model_values)
        axes.plot_surface(h_linears, h_valids, model_values + model_stds, alpha = 0.3, color=(1.0, 0.0, 0.0))
        axes.plot_surface(h_linears, h_valids, model_values - model_stds, alpha = 0.3, color=(1.0, 0.0, 0.0))
        axes.scatter(  #
            accuracy_df["h_linear"].to_numpy()[indices_in_slice],  #
            accuracy_df["desired_h_valid"].to_numpy()[indices_in_slice],  #
            accuracy_df["distance"].to_numpy()[indices_in_slice],  #
        )
        axes.set_zlim((np.min(model_values), np.max(model_values)))
        axes.set_xlabel("$h_\\mathrm{linear}$")
        axes.set_ylabel("$h_\\mathrm{V}$")
        axes.set_zlabel("distance at final iteration")
        plt.show()


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
    parser.add_argument("-f", "--fit", action="store_true",
                        help="Fit a model to the data rather than assuming it is a full Cartesian grid.")
    args = parser.parse_args()

    main(  #
        load_dir=pathlib.Path(args.load_dir),  #
        which_datasets=args.which_datasets,  #
        display=args.display,  #
        save_to=None if args.save_to is None else pathlib.Path(args.save_to),  #
        analysis_format=args.analysis,  #
        fit=args.fit,  #
    )
