import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import logs_setup
from registration import plot_data
from registration.lib import sinogram


def to_latex_scientific(x: float, precision: int = 2, include_plus: bool = False):
    if x == 0:
        return f"{0:.{precision}f}"
    exponent: int = int(f"{x:e}".split("e")[1])
    mantissa: float = x / (10.0 ** exponent)
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    if include_plus:
        return fr"{mantissa:+.{precision}f} \times 10^{{{exponent}}}"
    return fr"{mantissa:.{precision}f} \times 10^{{{exponent}}}"


SAVE_DIRECTORY = pathlib.Path("data/register_plot_data")


def main(file: str | None):
    if file is None:
        files = list(SAVE_DIRECTORY.glob("*.pkl"))
        file = max(files, key=lambda f: f.stem)
    logger.info("Loading save file '{}'".format(str(file)))
    pdata = torch.load(file, weights_only=False)
    assert isinstance(pdata, plot_data.RegisterPlotData)

    if len(pdata.datasets) == 0:
        logger.warning("No datasets in save file '{}'.".format(file))
        exit(0)

    fixed_numels_drr = np.array(
        [dataset.fixed_image_numel for dataset in pdata.datasets if dataset.obj_func_name == "drr"])
    fixed_numels_grangeat = np.array(
        [dataset.fixed_image_numel for dataset in pdata.datasets if dataset.obj_func_name == "grangeat_classic"])
    fixed_numels_healpix = np.array(
        [dataset.fixed_image_numel for dataset in pdata.datasets if dataset.obj_func_name == "grangeat_healpix"])
    times_per_iteration_drr = np.array(
        [dataset.time_per_iteration for dataset in pdata.datasets if dataset.obj_func_name == "drr"])
    times_per_iteration_grangeat = np.array(
        [dataset.time_per_iteration for dataset in pdata.datasets if dataset.obj_func_name == "grangeat_classic"])
    times_per_iteration_healpix = np.array(
        [dataset.time_per_iteration for dataset in pdata.datasets if dataset.obj_func_name == "grangeat_healpix"])
    truth_start_distances_drr = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "drr"])
    truth_start_distances_grangeat = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat_classic"])
    truth_start_distances_healpix = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat_healpix"])
    truth_converged_distances_drr = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "drr"])
    truth_converged_distances_grangeat = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat_classic"])
    truth_converged_distances_healpix = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat_healpix"])

    #
    # PSO iteration time DRR vs. Grangeat resampling
    #
    # lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    # lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)
    fig, axes = plt.subplots()
    axes.grid(True, which="both")
    axes.scatter(fixed_numels_drr, times_per_iteration_drr, label="DRR")
    axes.scatter(fixed_numels_grangeat, times_per_iteration_grangeat, label="Grangeat classic")
    axes.scatter(fixed_numels_healpix, times_per_iteration_healpix, label="Grangeat HEALPix")
    axes.set_xlabel("Fixed image element count")
    axes.set_ylabel("PSO iteration time [s]")
    axes.set_xlim(0.0, None)
    axes.set_ylim(0.0, None)
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/pso_against_volume_size.pgf")

    #
    # Box and whisker plot
    #
    all_xs: np.ndarray = np.concat(
        (truth_start_distances_drr, truth_start_distances_grangeat, truth_start_distances_healpix))
    bin_width: float = 0.1
    bin_centres: np.ndarray = bin_width * np.arange(int(np.floor(all_xs.min() / bin_width)),
                                                    int(np.ceil(all_xs.max() / bin_width))).astype(np.float32)
    bins: np.ndarray = bin_centres - 0.5 * bin_width
    display_width_fraction: float = 0.7

    fig, axes = plt.subplots()

    def add_box_plot(xs: np.ndarray, ys: np.ndarray, offset: float, colour: str, label: str) -> int:
        digitised: np.ndarray = np.digitize(xs, bins) - 1
        ys_binned_every: list[np.ndarray] = [ys[digitised == i] for i in range(len(bins))]
        ys_binned: list[np.ndarray] = [e for e in ys_binned_every if len(e) >= 3]
        positions: np.ndarray = np.array([e for i, e in enumerate(bin_centres) if len(ys_binned_every[i]) >= 3])
        axes.boxplot(ys_binned, positions=positions + offset, widths=display_width_fraction * bin_width / 3.0,
                     label=label,  #
                     patch_artist=True,  #
                     boxprops=dict(facecolor=colour, linewidth=0, alpha=0.8),  #
                     medianprops=dict(color="black", linewidth=1.),  #
                     whiskerprops=dict(color=colour),  #
                     capprops=dict(color=colour),  #
                     flierprops=dict(markeredgecolor=colour, marker='x', markersize=4, alpha=0.5))
        return max([i for i in range(len(bin_centres)) if len(ys_binned_every[i]) >= 3])

    highest_drr_bin: int = add_box_plot(truth_start_distances_drr, truth_converged_distances_drr,
                                        -display_width_fraction * bin_width / 3.0,
                                        plt.rcParams['axes.prop_cycle'].by_key()['color'][0], "DRR")
    highest_grangeat_bin: int = add_box_plot(truth_start_distances_grangeat, truth_converged_distances_grangeat, 0.0,
                                             plt.rcParams['axes.prop_cycle'].by_key()['color'][1], "Grangeat classic")
    highest_healpix_bin: int = add_box_plot(truth_start_distances_healpix, truth_converged_distances_healpix,
                                            display_width_fraction * bin_width / 3.0,
                                            plt.rcParams['axes.prop_cycle'].by_key()['color'][2], "Grangeat HEALPix")
    highest_bin = min(max(highest_drr_bin, highest_grangeat_bin, highest_healpix_bin), int(np.ceil(1.0 / bin_width)) - 2)
    axes.set_aspect("equal")
    axes.grid(True, which="minor", axis="x")
    axes.grid(True, which="both", axis="y")
    axes.set_xlim(0.0, bin_centres[highest_bin] + bin_width)
    axes.set_ylim(0.0, bin_centres[highest_bin] + bin_width)
    xticks = bin_centres[:(highest_bin + 1)]
    axes.set_xticks(xticks)
    axes.set_xticks(np.concat(
        (bin_centres[:(highest_bin + 1)] - 0.5 * bin_width, np.array([bin_centres[highest_bin] + 0.5 * bin_width]))),
        minor=True)
    axes.tick_params(axis="x", which="major", length=3)
    axes.tick_params(axis="x", which="minor", length=10)
    axes.set_xticklabels(["{:.1f}".format(x) for x in xticks])
    axes.set_xlabel("Starting to G.T. distance in SE(3), binned")
    axes.set_ylabel("Converged to G.T. distance in SE(3)")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/conv_dist_vs_start_dist.pgf")

    #
    # Stats
    #
    sum_: float = 0.0
    count: int = 0
    fixed_numels_resample = np.concat((fixed_numels_grangeat, fixed_numels_healpix))
    times_per_iteration_resample = np.concat((times_per_iteration_grangeat, times_per_iteration_healpix))
    for i, fixed_numel_drr in enumerate(fixed_numels_drr):
        try:
            j = list(fixed_numels_resample).index(fixed_numel_drr)
        except ValueError:
            continue
        sum_ += times_per_iteration_resample[j] / times_per_iteration_drr[i]
        count += 1

    average_iteration_time_ratio_resample_to_drr = sum_ / float(count)
    logger.info("Average PSO iteration time ratio resample to DRR = {:.4f}".format(
        average_iteration_time_ratio_resample_to_drr))

    sum_: float = 0.0
    count: int = 0
    for i, fixed_numel_grangeat in enumerate(fixed_numels_grangeat):
        try:
            j = list(fixed_numels_healpix).index(fixed_numel_grangeat)
        except ValueError:
            continue
        sum_ += times_per_iteration_healpix[j] / times_per_iteration_grangeat[i]
        count += 1
    average_iteration_time_ratio_healpix_to_classic = np.nan
    if count > 0:
        average_iteration_time_ratio_healpix_to_classic = sum_ / float(count)
        logger.info("Average PSO iteration time ratio HEALPix to classic = {:.4f}".format(
            average_iteration_time_ratio_healpix_to_classic))

    # gradient_ratio_classic_to_drr = gradient_classic / gradient_drr
    # print("Average converged to starting SE(3) dist classic to DRR =", gradient_ratio_classic_to_drr)
    # gradient_ratio_healpix_to_classic = gradient_healpix / gradient_classic
    # print("Average converged to starting SE(3) dist HEALPix to classic =", gradient_ratio_healpix_to_classic)

    with open("data/temp/pso_stats.txt", "w") as file:
        file.write("Particle count = {}\n"
                   "Iteration count = {}\n"
                   "Average PSO iteration time ratio Grangeat to DRR = {:.4f}\n"
                   "Average PSO iteration time ratio HEALPix to classic = {:.4f}\n".format(pdata.particle_count,
                                                                                           pdata.iteration_count,
                                                                                           average_iteration_time_ratio_resample_to_drr,
                                                                                           average_iteration_time_ratio_healpix_to_classic))

    plt.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # for outputting PGFs
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["scatter.marker"] = 'x'
    plt.rcParams["font.size"] = 15  # figures are includes in latex at half size, so 18 is desired size. matplotlib
    # scales up by 1.2 (God only knows why), so setting to 15

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-f", "--file", type=str, default=None, help="Provide a path to a specific saved "
                                                                     "file to plot. If not provided, the latest save "
                                                                     "file named according to the date/time convention "
                                                                     "will be used.")
    args = parser.parse_args()

    main(args.file)
