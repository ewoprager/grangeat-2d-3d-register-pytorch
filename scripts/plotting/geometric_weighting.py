import itertools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from reg23_experiments.analysis.helpers import dataframe_rectangular_columns_to_tensor

RESULTS_DIR = pathlib.Path("experimental_results/program_truncation")
CROPPING_RESULTS_DIR = RESULTS_DIR / "2026-08-13_11-40-04_gw-0-cropping"
REEVAL_RESULTS_DIR = RESULTS_DIR / "2026-08-13_23-32-31_gw-1-reeval"
OUTPUT_DIR = pathlib.Path("figures/geometric_weighting")


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


def cartesian_plots(  #
        *,  #
        independent_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
        ylim: tuple[float, float] | None = None,  #
) -> list:
    assert 2 <= len(independent_values)
    assert dependent_values.size() == torch.Size([len(v) for _, v in independent_values])
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()
    plots = []
    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in independent_values[:-2]]):
        axis_index = () if index_value_pairs == () else tuple(i for i, _ in index_value_pairs)
        series = []
        for j, v in enumerate(independent_values[-2][1]):
            dependent_index = axis_index + (j,)
            serie = {  #
                "label": f"{independent_values[-2][0]}={var_to_string(independent_values[-2][0], v)}",  #
                "xvalues": independent_values[-1][1].tolist(),  #
                "yvalues": dependent_values[*dependent_index, :].tolist(),  #
            }
            if dependent_errors is not None:
                serie["yerr"] =  dependent_errors[*dependent_index, :].tolist()
            series.append(serie)
        plot = {  #
            "title": ";".join([  #
                f"{independent_values[i][0]}={var_to_string(independent_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ]),  #
            "xlabel": independent_values[-1][0],  #
            "ylabel": dependent_variable,  #
            "series": series,  #
        }
        if ylim is not None:
            plot["ylim"] = list(ylim)
        plots.append(plot)
    return plots


def main():
    instance_dirs: list[pathlib.Path] = [CROPPING_RESULTS_DIR]
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
    # assert "variables" in variables_config
    # variables: list[str] = list(variables_config["variables"].keys())
    assert "cartesian" in variables_config
    variables: list[str] = list(variables_config["cartesian"].keys())

    variable_hierarchy: list[str] = ["weight_alpha", "cropping", "cropping_method", "truncation_percent",
                                     "apply_scaling", "iterations_per_weight_update", "iterations_per_crop_update",
                                     "crop_expand", "mask", "desired_h_valid", "xray_path"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierarchy)}
    variables = sorted(  #
        variables,  #
        key=lambda name: variable_importances[name] if name in variable_importances else len(variable_hierarchy),  #
        reverse=True  #
    )

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

    independent_variables = axis_values
    dependent_variable = "distance from gold-standard"
    dependent_values = distances
    dependent_errors = distance_stds if distance_std_available else None

    ylim: tuple[float, float] | None = (0.0, dependent_values.amax(dim=-1).quantile(q=0.75).item()) if len(
        independent_variables) > 2 else None

    plots = cartesian_plots(  #
        independent_values=independent_variables,  #
        dependent_variable=dependent_variable,  #
        dependent_values=dependent_values,  #
        dependent_errors=dependent_errors,  #
        ylim=ylim,  #
    )

    with open(OUTPUT_DIR / "plots.yaml", 'w') as file:
        yaml.safe_dump(plots, file)


if __name__ == "__main__":
    main()
