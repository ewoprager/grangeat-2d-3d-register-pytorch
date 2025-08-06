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


def main():
    files = list(SAVE_DIRECTORY.glob("*.pkl"))
    latest_file = max(files, key=lambda f: f.stem)
    pdata = torch.load(latest_file, weights_only=False)
    assert isinstance(pdata, plot_data.RegisterPlotData)

    if len(pdata.datasets) == 0:
        logger.warn("No datasets in save file '{}'.".format(latest_file))
        exit(0)

    iteration_count = pdata.iteration_count
    particle_count = pdata.particle_count

    # pdata = plot_data.RegisterPlotData(iteration_count=iteration_count, particle_count=particle_count,
    #                                    datasets=[dataset for dataset in pdata.datasets if
    #                                              dataset.ground_truth_transformation.distance(
    #                                                  dataset.starting_transformation.to(
    #                                                      device=dataset.ground_truth_transformation.device)) < 0.7])

    fixed_numels_drr = np.array(
        [dataset.fixed_image_numel for dataset in pdata.datasets if dataset.obj_func_name == "drr"])
    fixed_numels_grangeat = np.array([dataset.fixed_image_numel for dataset in pdata.datasets if
                                      dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramClassic])
    fixed_numels_healpix = np.array([dataset.fixed_image_numel for dataset in pdata.datasets if
                                     dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramHEALPix])
    times_per_iteration_drr = np.array(
        [dataset.time_per_iteration for dataset in pdata.datasets if dataset.obj_func_name == "drr"])
    times_per_iteration_grangeat = np.array([dataset.time_per_iteration for dataset in pdata.datasets if
                                             dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramClassic])
    times_per_iteration_healpix = np.array([dataset.time_per_iteration for dataset in pdata.datasets if
                                            dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramHEALPix])
    truth_start_distances_drr = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "drr"])
    truth_start_distances_grangeat = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramClassic])
    truth_start_distances_healpix = np.array([dataset.ground_truth_transformation.distance(
        dataset.starting_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramHEALPix])
    truth_converged_distances_drr = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "drr"])
    truth_converged_distances_grangeat = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramClassic])
    truth_converged_distances_healpix = np.array([dataset.ground_truth_transformation.distance(
        dataset.converged_transformation.to(device=dataset.ground_truth_transformation.device)) for dataset in
        pdata.datasets if dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramHEALPix])

    #
    # PSO iteration time DRR vs. Grangeat resampling
    #
    # lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    # lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)
    fig, axes = plt.subplots()
    axes.scatter(fixed_numels_drr, times_per_iteration_drr, label="DRR")
    axes.scatter(fixed_numels_grangeat, times_per_iteration_grangeat, label="Grangeat classic")
    axes.scatter(fixed_numels_healpix, times_per_iteration_healpix, label="Grangeat HEALPix")
    axes.set_xlabel("Fixed image element count")
    axes.set_ylabel("PSO iteration time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/pso_against_volume_size.pgf")

    #
    # Converged distance from truth against starting distance from truth DRR vs. Grangeat resampling
    #
    # gradient_drr = np.dot(truth_start_distances_drr, truth_converged_distances_drr) / np.dot(truth_start_distances_drr,
    #                                                                                          truth_start_distances_drr)
    # gradient_classic = np.dot(truth_start_distances_grangeat, truth_converged_distances_grangeat) / np.dot(
    #     truth_start_distances_grangeat, truth_start_distances_grangeat)
    # gradient_healpix = np.dot(truth_start_distances_healpix, truth_converged_distances_healpix) / np.dot(
    #     truth_start_distances_healpix, truth_start_distances_healpix)
    # lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    # lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)
    fig, axes = plt.subplots()
    axes.scatter(truth_start_distances_drr, truth_converged_distances_drr, label="DRR")
    axes.scatter(truth_start_distances_grangeat, truth_converged_distances_grangeat, label="Grangeat classic")
    axes.scatter(truth_start_distances_healpix, truth_converged_distances_healpix, label="Grangeat HEALPix")
    plt.axis("square")
    axes.set_aspect("equal")
    axes.set_xlim(0, None)
    axes.set_ylim(0, None)
    # xs = np.array(plt.xlim())
    # ys_drr = gradient_drr * xs
    # axes.plot(xs, ys_drr, label="Linear fit: $y = {}x$".format(to_latex_scientific(gradient_drr)))
    # ys_classic = gradient_classic * xs
    # axes.plot(xs, ys_classic, label="Linear fit: $y = {}x$".format(to_latex_scientific(gradient_classic)))
    # ys_healpix = gradient_healpix * xs
    # axes.plot(xs, ys_healpix, label="Linear fit: $y = {}x$".format(to_latex_scientific(gradient_healpix)))
    # xs = np.array(plt.xlim())
    # ys_xnumel_drr = lin_coeffs_xnumel_drr[0] * xs + lin_coeffs_xnumel_drr[1]
    # axes.plot(xs, ys_xnumel_drr, label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_drr[0]),
    #                                                                      to_latex_scientific(lin_coeffs_xnumel_drr[1],
    #                                                                                          include_plus=True)))
    # ys_xnumel_resample = lin_coeffs_xnumel_resample[0] * xs + lin_coeffs_xnumel_resample[1]
    # axes.plot(xs, ys_xnumel_resample,
    #           label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_resample[0]),
    #                                                   to_latex_scientific(lin_coeffs_xnumel_resample[1],
    #                                                                       include_plus=True)))
    axes.set_xlabel("Starting to G.T. distance in SE(3)")
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
    print("Average PSO iteration time ratio resample to DRR =", average_iteration_time_ratio_resample_to_drr)

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
        print("Average PSO iteration time ratio HEALPix to classic =", average_iteration_time_ratio_healpix_to_classic)

    # gradient_ratio_classic_to_drr = gradient_classic / gradient_drr
    # print("Average converged to starting SE(3) dist classic to DRR =", gradient_ratio_classic_to_drr)
    # gradient_ratio_healpix_to_classic = gradient_healpix / gradient_classic
    # print("Average converged to starting SE(3) dist HEALPix to classic =", gradient_ratio_healpix_to_classic)

    with open("data/temp/pso_stats.txt", "w") as file:
        file.write("Average PSO iteration time ratio Grangeat to DRR = {:.4f}\n"
                   "Average PSO iteration time ratio HEALPix to classic = {:.4f}\n".format(
            average_iteration_time_ratio_resample_to_drr, average_iteration_time_ratio_healpix_to_classic))

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
    # parser.add_argument(
    #     "-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    main()
