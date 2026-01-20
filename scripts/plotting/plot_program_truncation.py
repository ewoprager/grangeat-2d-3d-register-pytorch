import argparse

import torch
import pathlib
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

    # basic
    data_path = data_dir / "iteration_counts.pkl"
    if data_path.is_file():
        data = torch.load(data_path)
        means = data.to(dtype=torch.float32).quantile(0.5, dim=1)
        plt.plot(means.cpu().numpy())
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
