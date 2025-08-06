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
    pdata = torch.load("data/drr_vs_grangeat.pkl", weights_only=False)
    assert isinstance(pdata, plot_data.DrrVsGrangeatPlotData)

    volume_numels = np.array([dataset.ct_volume_numel for dataset in pdata.datasets])
    sinogram3d_sizes = np.array([dataset.sinogram3d_size for dataset in pdata.datasets])
    x_ray_numels = np.array([dataset.x_ray_numel for dataset in pdata.datasets])
    sinogram2d_sizes = np.array([dataset.sinogram2d_size for dataset in pdata.datasets])
    drr_times = np.array([dataset.drr_time for dataset in pdata.datasets])
    resample_times = np.array([dataset.resample_time for dataset in pdata.datasets])
    x_ray_numels_classic = np.array(
        [dataset.x_ray_numel for dataset in pdata.datasets if dataset.sinogram_type == sinogram.SinogramClassic])
    x_ray_numels_healpix = np.array(
        [dataset.x_ray_numel for dataset in pdata.datasets if dataset.sinogram_type == sinogram.SinogramHEALPix])
    resample_times_classic = np.array(
        [dataset.resample_time for dataset in pdata.datasets if dataset.sinogram_type == sinogram.SinogramClassic])
    resample_times_healpix = np.array(
        [dataset.resample_time for dataset in pdata.datasets if dataset.sinogram_type == sinogram.SinogramHEALPix])

    #
    # DRR vs. Grangeat resampling
    #
    lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    lin_coeffs_xnumel_classic = np.polyfit(x_ray_numels_classic, resample_times_classic, 1)
    lin_coeffs_xnumel_healpix = np.polyfit(x_ray_numels_healpix, resample_times_healpix, 1)

    fig, axes = plt.subplots()
    axes.scatter(x_ray_numels, drr_times, label="DRR")
    axes.scatter(x_ray_numels_classic, resample_times_classic, label="Grangeat classic")
    axes.scatter(x_ray_numels_healpix, resample_times_healpix, label="Grangeat HEALPix")
    xs = np.array(plt.xlim())
    ys_xnumel_drr = lin_coeffs_xnumel_drr[0] * xs + lin_coeffs_xnumel_drr[1]
    axes.plot(xs, ys_xnumel_drr, label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_drr[0]),
                                                                         to_latex_scientific(lin_coeffs_xnumel_drr[1],
                                                                                             include_plus=True)))
    ys_xnumel_classic = lin_coeffs_xnumel_classic[0] * xs + lin_coeffs_xnumel_classic[1]
    axes.plot(xs, ys_xnumel_classic,
              label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_classic[0]),
                                                      to_latex_scientific(lin_coeffs_xnumel_classic[1],
                                                                          include_plus=True)))
    ys_xnumel_healpix = lin_coeffs_xnumel_healpix[0] * xs + lin_coeffs_xnumel_healpix[1]
    axes.plot(xs, ys_xnumel_healpix,
              label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_healpix[0]),
                                                      to_latex_scientific(lin_coeffs_xnumel_healpix[1],
                                                                          include_plus=True)))
    axes.set_xlabel("DRR / resampling element count")
    axes.set_ylabel("Evaluation time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/against_drr_size.pgf")

    #
    # Sinogram 3D evaluation time
    #
    sinogram3d_evaluation_times = np.genfromtxt("data/sinogram3d_evaluation_times.txt", dtype=None)
    classic_sizes = np.array([float(tup[1]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramClassic"])
    classic_times = np.array([float(tup[2]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramClassic"])
    healpix_sizes = np.array([float(tup[1]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramHEALPix"])
    healpix_times = np.array([float(tup[2]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramHEALPix"])

    # normal plot
    fig, axes = plt.subplots()
    axes.scatter(classic_sizes, classic_times, label="Classic")
    axes.scatter(healpix_sizes, healpix_times, label="HEALPix")
    axes.set_xlabel("Sinogram size $N$")
    axes.set_ylabel("Evaluation time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/sinogram3d_against_size.pgf")

    # log-log plot
    lin_coeffs_classic = np.polyfit(np.log10(classic_sizes), np.log10(classic_times), 1)
    lin_coeffs_healpix = np.polyfit(np.log10(healpix_sizes), np.log10(healpix_times), 1)
    fig, axes = plt.subplots()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.scatter(classic_sizes, classic_times, label="Classic")
    axes.scatter(healpix_sizes, healpix_times, label="HEALPix")
    xs = np.array(plt.xlim())
    ys_classic = np.pow(10.0, lin_coeffs_classic[0] * np.log10(xs) + lin_coeffs_classic[1])
    axes.plot(xs, ys_classic,
              label="Power fit: $t = {}N^{{{}}}$".format(to_latex_scientific(np.pow(10.0, lin_coeffs_classic[1])),
                                                         to_latex_scientific(lin_coeffs_classic[0])))
    ys_healpix = np.pow(10.0, lin_coeffs_healpix[0] * np.log10(xs) + lin_coeffs_healpix[1])
    axes.plot(xs, ys_healpix,
              label="Power fit: $t = {}N^{{{}}}$".format(to_latex_scientific(np.pow(10.0, lin_coeffs_healpix[1])),
                                                         to_latex_scientific(lin_coeffs_healpix[0])))
    axes.set_xlabel("Sinogram size $N$")
    axes.set_ylabel("Evaluation time $t$ [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/ln_sinogram3d_against_ln_size.pgf")

    #
    # Sinogram 2D evaluation time
    #
    sinogram2d_evaluation_times = np.genfromtxt("data/sinogram2d_evaluation_times.txt", dtype=None)
    sinogram2d_sizes = np.array([float(tup[0]) for tup in sinogram2d_evaluation_times])
    sinogram2d_times = np.array([float(tup[1]) for tup in sinogram2d_evaluation_times])

    # normal plot
    fig, axes = plt.subplots()
    axes.scatter(sinogram2d_sizes, sinogram2d_times)
    axes.set_xlabel("Sinogram size $M$")
    axes.set_ylabel("Evaluation time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/sinogram2d_against_size.pgf")

    # log-log plot
    lin_coeffs = np.polyfit(np.log10(sinogram2d_sizes), np.log10(sinogram2d_times), 1)
    fig, axes = plt.subplots()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.scatter(sinogram2d_sizes, sinogram2d_times)
    xs = np.array(plt.xlim())
    ys = np.pow(10.0, lin_coeffs[0] * np.log10(xs) + lin_coeffs[1])
    axes.plot(xs, ys, label="Power fit: $t = {}M^{{{}}}$".format(to_latex_scientific(np.pow(10.0, lin_coeffs[1])),
                                                                 to_latex_scientific(lin_coeffs[0])))
    axes.set_xlabel("Sinogram size $M$")
    axes.set_ylabel("Evaluation time $t$ [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/ln_sinogram2d_against_ln_size.pgf")

    plt.show()

    #
    # Stats
    #
    average_time_ratio_grangeat_to_drr = (resample_times / drr_times).mean()
    print("Average time ratio Grangeat to DRR =", average_time_ratio_grangeat_to_drr)

    # sum_: float = 0.0
    # count: int = 0
    # for i, healpix_size in enumerate(healpix_sizes):
    #     try:
    #         j = list(classic_sizes).index(healpix_size)
    #     except ValueError:
    #         continue
    #     sum_ += healpix_times[i] / classic_times[j]
    #     count += 1
    # average_sinogram3d_time_ratio_healpix_to_classic = sum_ / float(count)
    average_sinogram3d_time_ratio_healpix_to_classic = np.pow(10.0, lin_coeffs_healpix[1] - lin_coeffs_classic[1])
    print("Average sinogram3d eval. time ratio HEALPix to classic =", average_sinogram3d_time_ratio_healpix_to_classic)

    sum_: float = 0.0
    count: int = 0
    for i, x_ray_numel_classic in enumerate(x_ray_numels_classic):
        try:
            j = list(x_ray_numels_healpix).index(x_ray_numel_classic)
        except ValueError:
            continue
        sum_ += resample_times_healpix[j] / resample_times_classic[i]
        count += 1
    average_resample_time_ratio_healpix_to_classic = sum_ / float(count)
    print("Average resampling eval. time ratio HEALPix to classic =", average_resample_time_ratio_healpix_to_classic)

    with open("data/temp/stats.txt", "w") as file:
        file.write("Average time ratio Grangeat to DRR = {:.4f}\n"
                   "Average sinogram3d eval. time ratio HEALPix to classic = {:.4f}\n"
                   "Average resampling eval. time ratio HEALPix to classic = {:.4f}\n".format(
            average_time_ratio_grangeat_to_drr, average_sinogram3d_time_ratio_healpix_to_classic,
            average_resample_time_ratio_healpix_to_classic))


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
