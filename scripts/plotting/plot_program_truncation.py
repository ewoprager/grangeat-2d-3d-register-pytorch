import argparse

import torch
import pathlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

from reg23_experiments.notification import logs_setup


def main(*, load_dir: str | pathlib.Path, which_dataset: str, display: bool) -> None:
    assert load_dir.is_dir()
    if which_dataset == "latest":
        subdirs = []
        for element in load_dir.iterdir():
            if not element.is_dir():
                continue
            subdirs.append(str(element.stem))
        subdirs.sort()
        which_dataset = subdirs[-1]
    data_dir: pathlib.Path = load_dir / which_dataset
    assert data_dir.is_dir()

    config = torch.load(data_dir / "config.pkl")

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
    if True:
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
        tfs = torch.empty(tf_n) # truncation fractions
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
            plt.plot(nominal_distances, medians[0, i].cpu().numpy(), label=f"tf={tfs[i]}",
                     color=(f, 1.0 - f, 0.0))
        plt.ylim((0, config["maximum_iterations"]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Nominal distance in SE(3)")
        plt.ylabel("Iterations until within {:.2f} of G.T. in SE(3)".format(config["distance_threshold"]))
        plt.legend()

        plt.figure()
        plt.title("With masking")
        for i in range(tf_n):
            f = float(i) / float(tf_n - 1)
            plt.plot(nominal_distances, medians[1, i].cpu().numpy(), label=f"tf={tfs[i]}",
                     color=(f, 1.0 - f, 0.0))
        plt.ylim((0, config["maximum_iterations"]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Nominal distance in SE(3)")
        plt.ylabel("Iterations until within {:.2f} of G.T. in SE(3)".format(config["distance_threshold"]))
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
