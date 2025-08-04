import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

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


def main():
    pdata = torch.load("data/register_plot_data.pkl", weights_only=False)
    assert isinstance(pdata, plot_data.RegisterPlotData)

    iteration_count = pdata.iteration_count
    particle_count = pdata.particle_count

    volume_numels_drr = np.array(
        [dataset.ct_volume_numel for dataset in pdata.datasets if dataset.obj_func_name == "drr"])
    volume_numels_grangeat = np.array([dataset.ct_volume_numel for dataset in pdata.datasets if
                                       dataset.obj_func_name == "grangeat" and dataset.sinogram_type == sinogram.SinogramClassic])
    volume_numels_healpix = np.array([dataset.ct_volume_numel for dataset in pdata.datasets if
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
    axes.scatter(volume_numels_drr, times_per_iteration_drr, label="DRR")
    axes.scatter(volume_numels_grangeat, times_per_iteration_grangeat, label="Grangeat classic")
    axes.scatter(volume_numels_healpix, times_per_iteration_healpix, label="Grangeat HEALPix")
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
    axes.set_xlabel("CT volume element count")
    axes.set_ylabel("PSO iteration time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/pso_against_volume_size.pgf")

    #
    # Converged distance from truth against starting distance from truth DRR vs. Grangeat resampling
    #
    # lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    # lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)
    fig, axes = plt.subplots()
    axes.scatter(truth_start_distances_drr, truth_converged_distances_drr, label="DRR")
    axes.scatter(truth_start_distances_grangeat, truth_converged_distances_grangeat, label="Grangeat classic")
    axes.scatter(truth_start_distances_healpix, truth_converged_distances_healpix, label="Grangeat HEALPix")
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
    axes.set_xlabel("Distance of starting alignment in SE(3) from g.t.")
    axes.set_ylabel("Distance of converged alignment in SE(3) from g.t.")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/conv_dist_vs_start_dist.pgf")

    plt.show()

    # #  # # Stats  # #  # average_time_ratio_drr_to_grangeat = (drr_times / resample_times).mean()  # print("Average time ratio DRR to Grangeat =", average_time_ratio_drr_to_grangeat)  #  # sum_: float = 0.0  # count: int = 0  # for i, healpix_size in enumerate(healpix_sizes):  #     try:  #         j = list(classic_sizes).index(healpix_size)  #     except ValueError:  #         continue  #     sum_ += classic_times[j] / healpix_times[i]  #     count += 1  # average_sinogram3d_time_ratio_classic_to_healpix = sum_ / float(count)  # print("Average sinogram3d eval. time ratio Classic to HEALPix =", average_sinogram3d_time_ratio_classic_to_healpix)  #  # sum_: float = 0.0  # count: int = 0  # for i, x_ray_numel_classic in enumerate(x_ray_numels_classic):  #     try:  #         j = list(x_ray_numels_healpix).index(x_ray_numel_classic)  #     except ValueError:  #         continue  #     sum_ += resample_times_classic[i] / resample_times_healpix[j]  #     count += 1  # average_resample_time_ratio_classic_to_healpix = sum_ / float(count)  # print("Average resampling eval. time ratio Classic to HEALPix =", average_resample_time_ratio_classic_to_healpix)  #  # with open("data/temp/stats.txt", "w") as file:  #     file.write("Average time ratio DRR to Grangeat = {:.4f}\n"  #                "Average 3D sinogram evaluation time ratio Classic to HEALPix = {:.4f}\n"  #                "Average resampling eval. time ratio Classic to HEALPix = {:.4f}\n".format(  #         average_time_ratio_drr_to_grangeat, average_sinogram3d_time_ratio_classic_to_healpix,  #         average_resample_time_ratio_classic_to_healpix))


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
