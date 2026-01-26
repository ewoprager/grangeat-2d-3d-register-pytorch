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
    multiindex = df.set_index(ordered_axes)[value_column].sort_index()
    axis_values = {  #
        name: torch.from_numpy(multiindex.index.get_level_values(name).unique().to_numpy())  #
        for name in ordered_axes  #
    }
    axis_lengths = [len(axis) for _, axis in axis_values.items()]
    return torch.from_numpy(multiindex.to_numpy()).view(axis_lengths), axis_values


def main(*, load_dir: str | pathlib.Path, which_dataset: str, display: bool) -> None:
    assert load_dir.is_dir()
    if which_dataset == "latest":
        parquets = []
        for element in load_dir.iterdir():
            if not element.is_file() or element.suffix != ".parquet":
                continue
            parquets.append(str(element.stem))
        parquets.sort()
        which_dataset = parquets[-1]
    data_file: pathlib.Path = load_dir / (which_dataset + ".parquet")
    assert data_file.is_file()
    df = pd.read_parquet(data_file)

    # basic
    if False:
        truncation_fractions = []
        medians = None
        for element in data_dir.iterdir():
            if not element.is_dir():
                continue
            truncation_fractions.append(float(str(element.stem)[-5:].replace("p", ".")))
            iteration_counts = torch.load(element / "iteration_counts.pkl")
            m = iteration_counts.to(dtype=torch.float32).quantile(0.5, dim=1).unsqueeze(0)
            if medians is None:
                medians = m
            else:
                medians = torch.cat([medians, m], dim=0)

        nominal_distances = torch.linspace(0.1, 20.0, 8).numpy()
        for i in range(len(truncation_fractions)):
            f = float(i) / float(len(truncation_fractions) - 1)
            plt.plot(nominal_distances, medians[i].cpu().numpy(), label=f"tf={truncation_fractions[i]}",
                     color=(f, 1.0 - f, 0.0))
        plt.ylim((0.0, 20.0))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.show()

    # with and without masking
    if False:
        tf_to_ic_no_masking = {}  # truncation fraction to iteration counts
        tf_to_ic_masking = {}  # truncation fraction to iteration counts
        medians = None
        for element in data_dir.iterdir():
            if not element.is_dir():
                continue
            parameters = torch.load(element / "parameters.pkl")
            iteration_counts = torch.load(element / "iteration_counts.pkl")
            if parameters["mask"] == "Every evaluation":
                tf_to_ic_masking[parameters["truncation_fraction"]] = iteration_counts
            else:
                assert parameters["mask"] == "None"
                tf_to_ic_no_masking[parameters["truncation_fraction"]] = iteration_counts

        tf_to_ic_masking = dict(sorted(tf_to_ic_masking.items()))
        tf_to_ic_no_masking = dict(sorted(tf_to_ic_no_masking.items()))

        config = torch.load(data_dir / "config.pkl")
        nominal_distances = config["nominal_distances"]
        nd_n = nominal_distances.numel()

        tf_n = len(tf_to_ic_masking)
        assert len(tf_to_ic_no_masking) == tf_n
        tfs = torch.empty(tf_n)  # truncation fractions
        medians = torch.empty([2, tf_n, nd_n])
        for i, ((tf, ics_masking), (_, ics_no_masking)) in enumerate(
                zip(tf_to_ic_masking.items(), tf_to_ic_no_masking.items())):
            tfs[i] = tf
            medians[0, i, :] = ics_no_masking.to(dtype=torch.float32).quantile(0.5, dim=1).unsqueeze(0)
            medians[1, i, :] = ics_masking.to(dtype=torch.float32).quantile(0.5, dim=1).unsqueeze(0)

        plt.figure()
        plt.title("No masking")
        for i in range(tf_n):
            f = float(i) / float(tf_n - 1)
            plt.plot(nominal_distances, medians[0, i].cpu().numpy(), label=f"tf={tfs[i]}", color=(f, 1.0 - f, 0.0))
        plt.ylim((0, config["maximum_iterations"]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Nominal distance in SE(3)")
        plt.ylabel("Iterations until within {:.2f} of G.T. in SE(3)".format(config["distance_threshold"]))
        plt.legend()

        plt.figure()
        plt.title("With masking")
        for i in range(tf_n):
            f = float(i) / float(tf_n - 1)
            plt.plot(nominal_distances, medians[1, i].cpu().numpy(), label=f"tf={tfs[i]}", color=(f, 1.0 - f, 0.0))
        plt.ylim((0, config["maximum_iterations"]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Nominal distance in SE(3)")
        plt.ylabel("Iterations until within {:.2f} of G.T. in SE(3)".format(config["distance_threshold"]))
        plt.legend()

        plt.show()

    # convergence curves with and without masking
    if True:
        # distance_by_starting_distance_iteration = (  #
        #     df  #
        #     .loc[(df["mask"] == "None")]  #
        #     .set_index(["downsample_level", "truncation_fraction", "starting_distance", "iteration"])  #
        #     ["distance"]  #
        # )
        # downsample_level_vals = distance_by_starting_distance_iteration.index.get_level_values(
        #     "downsample_level").unique()
        # truncation_fraction_vals = distance_by_starting_distance_iteration.index.get_level_values(
        #     "truncation_fraction").unique()
        # starting_distance_vals = distance_by_starting_distance_iteration.index.get_level_values(
        #     "starting_distance").unique()
        # print(torch.from_numpy(downsample_level_vals.to_numpy()))
        # iteration_vals = distance_by_starting_distance_iteration.index.get_level_values("iteration").unique()
        # t = torch.from_numpy(distance_by_starting_distance_iteration.to_numpy()).view(
        #     [len(downsample_level_vals), len(truncation_fraction_vals), len(starting_distance_vals),
        #      len(iteration_vals)])
        # print(t.size())
        # print(t)

        distances, axis_values = dataframe_to_tensor(  #
            df.loc[(df["mask"] == "None")],  #
            ordered_axes=["downsample_level", "truncation_fraction", "starting_distance", "iteration"],  #
            value_column="distance")

        fig, axes = plt.subplots(distances.size(0), distances.size(1))
        fig.subplots_adjust(
            left=0.05,    # margin on left side of figure
            right=0.98,   # right margin
            bottom=0.08,  # bottom margin
            top=0.95,     # top margin
            wspace=0.2,   # width space between columns
            hspace=0.3    # height space between rows
        )
        for k, dl in enumerate(axis_values["downsample_level"]):
            for j, tf in enumerate(axis_values["truncation_fraction"]):
                for i, sd in enumerate(axis_values["starting_distance"]):
                    axes[k, j].plot(axis_values["iteration"], distances[k, j, i, :], label="s.d. {:.3f}".format(sd))
                    axes[k, j].set_title("d.l. {}; t.f. {:.3f}".format(dl, tf))
                    axes[k, j].set_xlabel("iteration")
                    axes[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[k, j].set_ylabel("distance from G.T.")
                    axes[k, j].set_ylim((0.0, None))
                    axes[k, j].legend()
        plt.show()

    return

    result = (df.loc[  # FILTER rows
        (df["downsample_ratio"] == 1) & (df["mask"] == "None") & (
            df["truncation_fraction"].isin(list_of_values))].sort_values("iteration")  # SORT rows
    ["distance"]  # SELECT column
                  )

    ds_to_tf_to_dist_no_masking = {}
    ds_to_tf_to_dist_masking = {}
    for element in data_dir.iterdir():
        if not element.is_dir():
            continue
        parameters = torch.load(element / "parameters.pkl")
        convergence_series = torch.load(element / "convergence_series.pkl")
        if parameters["mask"] == "Every evaluation":
            if parameters["downsample_level"] in ds_to_tf_to_dist_masking and parameters["truncation_fraction"] in \
                    ds_to_tf_to_dist_masking[parameters["downsample_level"]]:
                ds_to_tf_to_dist_masking[parameters["downsample_level"]][
                    parameters["truncation_fraction"]] = convergence_series
            else:
                ds_to_tf_to_dist_masking[parameters["downsample_level"]] = {"truncation_fraction": convergence_series}
        else:
            assert parameters["mask"] == "None"
            ds_to_tf_to_dist_no_masking[parameters["truncation_fraction"]] = convergence_series
    truncation_fractions, distances_no_masking = zip(*sorted(ds_to_tf_to_dist_no_masking.items()))
    _, distances_masking = zip(*sorted(ds_to_tf_to_dist_masking.items()))

    truncation_fractions = torch.tensor(truncation_fractions)  # size = (truncation fraction count,)
    distances_no_masking = torch.stack(
        distances_no_masking)  # size = (truncation fraction count, nominal distance count, iteration count)
    distances_masking = torch.stack(
        distances_masking)  # size = (truncation fraction count, nominal distance count, iteration count)

    assert len(truncation_fractions.size()) == 1
    assert distances_no_masking.size() == distances_masking.size()
    assert distances_no_masking.size(0) == truncation_fractions.numel()

    config = torch.load(data_dir / "config.pkl")
    nominal_distances = config["nominal_distances"]
    nd_n = nominal_distances.numel()
    tf_n = truncation_fractions.numel()
    it_n = distances_no_masking.size(2)

    plt.figure()
    for i in range(tf_n):
        f = float(i) / float(tf_n - 1)
        plt.plot(range(1, it_n + 1), distances_no_masking[i, 0].cpu().numpy(),
                 label=f"no masking; tf={truncation_fractions[i].item()}", color=(f, 1.0 - f, 0.0), linestyle="--")
        plt.plot(range(1, it_n + 1), distances_masking[i, 0].cpu().numpy(),
                 label=f"masking; tf={truncation_fractions[i].item()}", color=(f, 1.0 - f, 0.0))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Iteration")
    plt.ylabel("Avg. distance of best from G.T. in SE(3)")
    plt.legend()

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
