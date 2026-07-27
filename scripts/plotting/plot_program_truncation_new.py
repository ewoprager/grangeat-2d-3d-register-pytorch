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
    if "range" in variables_config:
        assert isinstance(variables_config["range"], dict)
        range_variables: list[str] = list(variables_config["range"].keys())
    else:
        range_variables = []

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
    # Collapse to just last iteration for level_000
    fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
    for i, xray in enumerate(df["xray_path"].unique()):
        accuracy_df = df[df["xray_path"] == xray]
        accuracy_df = accuracy_df[accuracy_df["iteration"] == accuracy_df["iteration"].max()].drop(columns=["iteration"])
        # Remove unnecessary dependent variable columns
        # accuracy_df.drop(columns=["crop_width", "crop_height"], inplace=True)
        # CT and X-ray paths to h_linear
        if False:
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
        y_sigma = np.minimum(y_sigma, 3.0)
        # Get the independent value vectors as a matrix
        independent_variables: list[str] = ["desired_h_valid", "weight_alpha"]
        approx_length_scales: list[float] = [30.0, 0.4]
        X: np.ndarray = accuracy_df[independent_variables].to_numpy()
        kernel = sklearn.gaussian_process.kernels.ConstantKernel() * sklearn.gaussian_process.kernels.RBF(
            length_scale=approx_length_scales)
        # gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=np.square(y_sigma)).fit(X, y)
        # gpr = sklearn.gaussian_process.GaussianProcessRegressor(alpha=y_sigma).fit(X, y)
        # gpr = sklearn.gaussian_process.GaussianProcessRegressor(alpha=15.0).fit(X, y)

        n = 50
        h_valids = np.linspace(5.0, 80.0, n)
        alphas = np.linspace(0.0, 1.0, n)
        alphas, h_valids = np.meshgrid(alphas, h_valids)
        # xs = h_valids
        # ys = alphas
        x_name = "desired_h_valid"
        y_name = "weight_alpha"
        # values = {  #
        #     x_name: xs.flatten(),  #
        #     y_name: ys.flatten(),  #
        # }
        # model_values, model_stds = gpr.predict(np.stack([values[name] for name in independent_variables], axis=1),
        #                                        return_std=True)
        # model_values = model_values.reshape(h_valids.shape)
        # model_stds = model_stds.reshape(h_valids.shape)
        # axes[i].plot_surface(xs, ys, model_values)
        # axes[i].plot_surface(xs, ys, model_values + model_stds, alpha=0.3, color=(1.0, 0.0, 0.0))
        # axes[i].plot_surface(xs, ys, model_values - model_stds, alpha=0.3, color=(1.0, 0.0, 0.0))
        axes.scatter(  #
            accuracy_df[x_name].to_numpy(),  #
            accuracy_df[y_name].to_numpy(),  #
            accuracy_df["distance"].to_numpy(),  #
            label=pathlib.Path(xray).name,#
        )
        # axes.set_zlim((0.0, np.quantile(accuracy_df["distance"].to_numpy(), 0.75)))
        axes.set_xlabel(f"{x_name}")
        axes.set_ylabel(f"{y_name}")
        axes.set_zlabel("distance at final iteration")
    plt.legend()
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
    args = parser.parse_args()

    main(  #
        load_dir=pathlib.Path(args.load_dir),  #
        which_datasets=args.which_datasets,  #
        display=args.display,  #
        save_to=None if args.save_to is None else pathlib.Path(args.save_to),  #
        analysis_format=args.analysis,  #
    )
