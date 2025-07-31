import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

import logs_setup
from registration import plot_data


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

    lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)

    fig, axes = plt.subplots()
    axes.scatter(x_ray_numels, drr_times, label="DRR")
    axes.scatter(x_ray_numels, resample_times, label="Grangeat resampling")
    xs = np.array(plt.xlim())
    ys_xnumel_drr = lin_coeffs_xnumel_drr[0] * xs + lin_coeffs_xnumel_drr[1]
    axes.plot(xs, ys_xnumel_drr,
              label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_drr[0]),
                                                        to_latex_scientific(lin_coeffs_xnumel_drr[1],
                                                                            include_plus=True)))
    ys_xnumel_resample = lin_coeffs_xnumel_resample[0] * xs + lin_coeffs_xnumel_resample[1]
    axes.plot(xs, ys_xnumel_resample,
              label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_xnumel_resample[0]),
                                                        to_latex_scientific(lin_coeffs_xnumel_resample[1],
                                                                            include_plus=True)))
    axes.set_xlabel("DRR / resampling element count")
    axes.set_ylabel("Evaluation time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/against_drr_size.pgf")

    sinogram3d_evaluation_times = np.genfromtxt("data/sinogram3d_evaluation_times.txt", dtype=None)
    classic_sizes = np.array([float(tup[1]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramClassic"])
    classic_times = np.array([float(tup[2]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramClassic"])
    healpix_sizes = np.array([float(tup[1]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramHEALPix"])
    healpix_times = np.array([float(tup[2]) for tup in sinogram3d_evaluation_times if tup[0] == "SinogramHEALPix"])

    lin_coeffs_classic = np.polyfit(np.log(classic_sizes), np.log(classic_times), 1)
    lin_coeffs_healpix = np.polyfit(np.log(healpix_sizes), np.log(healpix_times), 1)

    fig, axes = plt.subplots()
    axes.scatter(classic_sizes, classic_times, label="Classic")
    axes.scatter(healpix_sizes, healpix_times, label="HEALPix")
    axes.set_xlabel("Sinogram size")
    axes.set_ylabel("Evaluation time [s]")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/sinogram3d_against_size.pgf")

    fig, axes = plt.subplots()
    axes.scatter(np.log(classic_sizes), np.log(classic_times), label="Classic")
    axes.scatter(np.log(healpix_sizes), np.log(healpix_times), label="HEALPix")
    xs = np.array(plt.xlim())
    ys_classic = lin_coeffs_classic[0] * xs + lin_coeffs_classic[1]
    axes.plot(xs, ys_classic, label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_classic[0]),
                                                                        to_latex_scientific(lin_coeffs_classic[1],
                                                                                            include_plus=True)))
    ys_healpix = lin_coeffs_healpix[0] * xs + lin_coeffs_healpix[1]
    axes.plot(xs, ys_healpix, label="Linear fit: $y = {}x {}$".format(to_latex_scientific(lin_coeffs_healpix[0]),
                                                                        to_latex_scientific(lin_coeffs_healpix[1],
                                                                                            include_plus=True)))
    axes.set_xlabel("ln(Sinogram size)")
    axes.set_ylabel("ln(Evaluation time / s)")
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("data/temp/ln_sinogram3d_against_ln_size.pgf")

    plt.show()

    # Stats
    average_time_ratio_drr_to_grangeat = (drr_times / resample_times).mean()
    print("Average time ratio DRR to Grangeat =", average_time_ratio_drr_to_grangeat)

    sum_: float = 0.0
    count: int = 0
    for i, healpix_size in enumerate(healpix_sizes):
        try:
            j = list(classic_sizes).index(healpix_size)
        except ValueError:
            continue
        sum_ += classic_times[j] / healpix_times[i]
        count += 1
    average_sinogram3d_time_ratio_classic_to_healpix = sum_ / float(count)
    print("Average sinogram3d eval. time ratio Classic to HEALPix =", average_sinogram3d_time_ratio_classic_to_healpix)

    with open("data/temp/stats.txt", "w") as file:
        file.write("Average time ratio DRR to Grangeat = {:.4f}\n"
                   "Average 3D sinogram evaluation time ratio Classic to HEALPix = {:.4f}\n".format(
            average_time_ratio_drr_to_grangeat, average_sinogram3d_time_ratio_classic_to_healpix))


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
