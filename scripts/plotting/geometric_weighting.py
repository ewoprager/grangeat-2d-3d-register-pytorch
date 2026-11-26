import itertools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from reg23_experiments.analysis.manipulation import CartesianZippedTensors, dataframe_to_cartesian_zipped_tensors

RESULTS_DIR = pathlib.Path("experimental_results/program_truncation")
OLORIN_RESULTS_DIR = pathlib.Path("experimental_results/from_olorin/program_truncation")
N1_RESULTS_DIR = RESULTS_DIR / "2026-09-24_11-58-12_n1_sims_scal"
N2_RESULTS_DIR = RESULTS_DIR / "2026-09-25_12-35-52_n2_cropping"
N3_RESULTS_DIR = OLORIN_RESULTS_DIR / "2026-09-25_12-43-02_n3_weighting"
N4_RESULTS_DIR = OLORIN_RESULTS_DIR / "2026-09-25_16-51-10_n4_weighting_grad"
N5_RESULTS_DIR = RESULTS_DIR / "2026-09-28_18-17-11_n5_filtering"
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
        cartesian_axes_values: list[tuple[str, np.ndarray]],  #
        zipped_axis_values: list[tuple[str, np.ndarray]],  #
        dependent_variable: str,  #
        dependent_values: torch.Tensor,  #
        dependent_errors: torch.Tensor | None = None,  #
) -> list:
    axes_threshold = -1 if zipped_axis_values else -2
    assert 1 <= len(cartesian_axes_values) <= 2 + abs(axes_threshold)
    axes_lengths = [len(v) for _, v in cartesian_axes_values]
    if zipped_axis_values:
        zipped_length = len(zipped_axis_values[0][1])
        assert all(len(t[1]) == zipped_length for t in zipped_axis_values)
        axes_lengths += [zipped_length]
    assert dependent_values.size() == torch.Size(axes_lengths)
    if dependent_errors is not None:
        assert dependent_errors.size() == dependent_values.size()

    # getting the median largest distance value
    ylim: tuple[float, float] | None = (0.0, dependent_values.amax(dim=-1).quantile(q=0.9).item()) if len(
        cartesian_axes_values) > 2 else None

    x_label = cartesian_axes_values[-1][0]

    plots = []
    for index_value_pairs in itertools.product(*[enumerate(v) for _, v in cartesian_axes_values[:axes_threshold]]):
        axis_index = tuple(i for i, _ in index_value_pairs)
        series = []

        if zipped_axis_values:
            zipped_variables = [t[0] for t in reversed(zipped_axis_values)]
            for i, zipped_values in enumerate(zip(*[t[1] for t in reversed(zipped_axis_values)])):
                dependent_index = axis_index + (slice(None), i)
                line_label = ";".join(  #
                    f"{var}={var_to_string(var, val)}"  #
                    for var, val in zip(zipped_variables, zipped_values)  #
                )
                serie = {  #
                    "label": line_label,  #
                    "xvalues": cartesian_axes_values[-1][1].tolist(),  #
                    "yvalues": dependent_values[dependent_index].tolist(),  #
                }
                if dependent_errors is not None:
                    serie["yerr"] = dependent_errors[dependent_index].tolist()
                series.append(serie)
        else:
            line_variable = cartesian_axes_values[-2][0]
            line_values = cartesian_axes_values[-2][1]
            for j, line_value in enumerate(line_values):
                dependent_index = axis_index + (j, slice(None))
                serie = {  #
                    "label": f"{line_variable}={var_to_string(line_variable, line_value)}",  #
                    "xvalues": cartesian_axes_values[-1][1].tolist(),  #
                    "yvalues": dependent_values[dependent_index].tolist(),  #
                }
                if dependent_errors is not None:
                    serie["yerr"] = dependent_errors[dependent_index].tolist()
                series.append(serie)

        plot = {  #
            "title": ";".join([  #
                f"{cartesian_axes_values[i][0]}={var_to_string(cartesian_axes_values[i][0], w)}"  #
                for i, w in enumerate([v for _, v in index_value_pairs])  #
            ]),  #
            "xlabel": x_label,  #
            "ylabel": dependent_variable,  #
            "series": series,  #
        }
        if ylim is not None:
            plot["ylim"] = list(ylim)
        plots.append(plot)
    return plots


def simple_shared_cartesian(directory, name):
    instance_dirs: list[pathlib.Path] = [directory]
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

    # -----
    # Including extra datapoints from '2026-09-24_11-58-12_n1_sims_scal'
    if directory != N1_RESULTS_DIR:
        assert distance_std_available
        extra_df = pd.concat([  #
            pd.read_parquet(element)  #
            for element in N1_RESULTS_DIR.iterdir()  #
            if element.stem.startswith("data") and element.suffix == ".parquet"  #
        ], ignore_index=True)
        extra_df = extra_df.drop(columns=["apply_scaling", "ct_path"])
        extra_df["xray_path"] = extra_df["xray_path"].apply(lambda p: pathlib.Path(p).name)
        specific_rows = extra_df[extra_df["sim_metric"] == "gradient_correlation"]
        #
        df = df.drop(columns=["ct_path"])
        df["xray_path"] = df["xray_path"].apply(lambda p: pathlib.Path(p).name)
        #
        df = pd.concat([df, specific_rows], ignore_index=True)
        #
        if "sim_metric" in cartesian_variables:
            cartesian_variables.remove("sim_metric")
        if "weighting_method" in cartesian_variables:
            cartesian_variables.remove("weighting_method")

    variable_hierarchy: list[str] = ["weighting", "weight_alpha", "iterations_per_crop_update", "cropping",
                                     "cropping_method", "truncation_percent", "apply_scaling",
                                     "iterations_per_weight_update", "crop_expand", "mask", "desired_h_valid",
                                     "xray_path"]  # most to least important
    variable_importances = {name: importance for importance, name in enumerate(variable_hierarchy)}
    cartesian_variables = sorted(  #
        cartesian_variables,  #
        key=lambda name: variable_importances[name] if name in variable_importances else len(variable_hierarchy),  #
        reverse=True  #
    )

    dependent_variables = ["distance"]
    if distance_std_available:
        dependent_variables.append("distance_std")

    czt: CartesianZippedTensors = dataframe_to_cartesian_zipped_tensors(  #
        df,  #
        cartesian_variables=cartesian_variables + ["iteration"],  #
        dependent_variables=dependent_variables,  #
    )

    dependent_variable = "distance from gold-standard"
    dependent_values = czt.dependent_variable_tensors["distance"]
    dependent_errors = czt.dependent_variable_tensors["distance_std"] if distance_std_available else None

    plots = cartesian_plots(  #
        cartesian_axes_values=czt.cartesian_axes_values,  #
        zipped_axis_values=czt.zipped_axis_values,  #
        dependent_variable=dependent_variable,  #
        dependent_values=dependent_values,  #
        dependent_errors=dependent_errors,  #
    )

    with open(OUTPUT_DIR / f"{name}.yaml", 'w') as file:
        yaml.safe_dump(plots, file)


def main():
    if True:
        # 1: sims_scal
        simple_shared_cartesian(N1_RESULTS_DIR, "n1_sims_scal")
    if True:
        # 2: cropping
        simple_shared_cartesian(N2_RESULTS_DIR, "n2_cropping")
    if True:
        # 3: weighting
        simple_shared_cartesian(N3_RESULTS_DIR, "n3_weighting")
    if True:
        # 4: weighting grad
        simple_shared_cartesian(N4_RESULTS_DIR, "n4_weighting_grad")
    if True:
        # 5: filtering
        simple_shared_cartesian(N5_RESULTS_DIR, "n5_filtering")


if __name__ == "__main__":
    main()
