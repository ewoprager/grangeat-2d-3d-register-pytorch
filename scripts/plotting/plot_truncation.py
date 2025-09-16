import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib

from notification import logs_setup
from registration.lib.plot import fit_power_relationship, to_latex_scientific


def main(load_path: pathlib.Path, save_path: pathlib.Path):
    assert load_path.is_dir()

    for file in load_path.iterdir():
        if not file.is_file() or file.suffix != ".pkl":
            continue

        pdata = torch.load(file, weights_only=False)
        if not isinstance(pdata, dict):
            logger.error("File '{}' is of unrecognized type '{}'.".format(str(file), type(pdata).__name__))
            continue

        assert "truncation_fractions" in pdata
        truncation_fractions = pdata["truncation_fractions"]

        name = file.stem

        if "vals_at_gt" in pdata:
            vals_at_gt = pdata["vals_at_gt"]
            vals_at_gt_stds, vals_at_gt_means = torch.std_mean(vals_at_gt, dim=1)
            vals_at_gt_medians = vals_at_gt.quantile(q=0.5, dim=1)
            coefficients_stds = fit_power_relationship(truncation_fractions[1:], vals_at_gt_stds[1:])
            stds_fit = coefficients_stds[0] * truncation_fractions[1:].pow(coefficients_stds[1])
            coefficients_means = fit_power_relationship(truncation_fractions[1:], vals_at_gt_means[1:] + 1.0)
            means_fit = coefficients_means[0] * truncation_fractions[1:].pow(coefficients_means[1]) - 1.0
            coefficients_medians = fit_power_relationship(truncation_fractions[1:], vals_at_gt_medians[1:] + 1.0)
            medians_fit = coefficients_medians[0] * truncation_fractions[1:].pow(coefficients_medians[1]) - 1.0

            fig, axes = plt.subplots()
            axes.scatter(truncation_fractions[1:].log().cpu().numpy(), (vals_at_gt_means[1:] + 1.0).log().cpu().numpy(),
                         label="mean")
            axes.scatter(truncation_fractions[1:].log().cpu().numpy(),
                         (vals_at_gt_medians[1:] + 1.0).log().cpu().numpy(), label="median")
            axes.set_xlabel("ln(truncation fraction)")
            axes.set_ylabel("ln(obj. func. value at ground truth)")
            axes.set_title("{}: G.T. value average log logs".format(name))
            plt.legend()

            fig, axes = plt.subplots()
            spacing = (truncation_fractions[1] - truncation_fractions[0]).item()
            axes.boxplot(vals_at_gt.t().cpu().numpy(), positions=truncation_fractions.cpu().numpy(),
                         widths=0.5 * spacing)
            axes.scatter(truncation_fractions.cpu().numpy(), vals_at_gt_means.cpu().numpy(), label="means")
            axes.plot(truncation_fractions[1:].cpu().numpy(), means_fit.cpu().numpy(),
                      label="mean fit: $y = {}x^{{{}}}$".format(to_latex_scientific(coefficients_means[0].item()),
                                                                to_latex_scientific(coefficients_means[1].item())))
            axes.plot(truncation_fractions[1:].cpu().numpy(), medians_fit.cpu().numpy(),
                      label="median fit: $y = {}x^{{{}}}$".format(to_latex_scientific(coefficients_medians[0].item()),
                                                                  to_latex_scientific(coefficients_medians[1].item())))
            axes.set_xlabel("truncation fraction")
            axes.set_ylabel("obj. func. value at ground truth")
            axes.set_xlim(truncation_fractions[0] - spacing, truncation_fractions[-1] + spacing)
            axes.set_title("{}: G.T. value box whisker".format(name))
            plt.legend()

            fig, axes = plt.subplots()
            axes.scatter(truncation_fractions.cpu().numpy(), vals_at_gt_stds.cpu().numpy(), label="std. devs.")
            axes.plot(truncation_fractions[1:].cpu().numpy(), stds_fit.cpu().numpy(),
                      label="fit: $y = {}x^{{{}}}$".format(to_latex_scientific(coefficients_stds[0].item()),
                                                           to_latex_scientific(coefficients_stds[1].item())))
            axes.set_xlabel("truncation fraction")
            axes.set_ylabel("obj. func. value at ground truth")
            axes.set_title("{}: G.T. value standard deviations".format(name))
            plt.legend()

        if "opt_distances" in pdata:
            opt_distances = pdata["opt_distances"]

            fig, axes = plt.subplots()
            axes.boxplot(opt_distances.t().cpu().numpy(), positions=truncation_fractions.cpu().numpy(), widths=0.025)
            axes.set_xlabel("truncation fraction")
            axes.set_ylabel("distance in SE(3) between G.T. and optimum")
            axes.set_title("{}: opt. distance box whisker".format(name))

            # plt.legend()

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
    parser.add_argument("-l", "--load-path", type=str, default="data/temp/truncation/measurement",
                        help="Set a directory from which to load .pkl files to plot.")
    parser.add_argument("-s", "--save-path", type=str, default="figures/truncation/measurement",
                        help="Set a directory in which to save the resulting figures..")
    # parser.add_argument(
    #     "-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    main(pathlib.Path(args.load_path), pathlib.Path(args.save_path))
