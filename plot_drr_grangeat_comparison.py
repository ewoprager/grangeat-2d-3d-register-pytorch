import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

import logs_setup
from registration import plot_data


def main():
    pdata = torch.load("data/drr_vs_grangeat.pkl", weights_only=False)

    volume_numels = np.array([dataset.ct_volume_numel for dataset in pdata.datasets])
    sinogram3d_sizes = np.array([dataset.sinogram3d_size for dataset in pdata.datasets])
    x_ray_numels = np.array([dataset.x_ray_numel for dataset in pdata.datasets])
    sinogram2d_sizes = np.array([dataset.sinogram2d_size for dataset in pdata.datasets])
    drr_times = np.array([dataset.drr_time for dataset in pdata.datasets])
    resample_times = np.array([dataset.resample_time for dataset in pdata.datasets])

    lin_coeffs_vnumel_drr = np.polyfit(volume_numels, drr_times, 1)
    lin_coeffs_vnumel_resample = np.polyfit(volume_numels, resample_times, 1)
    lin_coeffs_xnumel_drr = np.polyfit(x_ray_numels, drr_times, 1)
    lin_coeffs_xnumel_resample = np.polyfit(x_ray_numels, resample_times, 1)

    fig, axes = plt.subplots()
    axes.scatter(volume_numels, drr_times, label="DRR")
    axes.scatter(volume_numels, resample_times, label="Grangeat resampling")
    xs = np.array(plt.xlim())
    ys_vnumel_drr = lin_coeffs_vnumel_drr[0] * xs + lin_coeffs_vnumel_drr
    axes.plot(xs, ys_vnumel_drr, label="Linear fit")
    ys_vnumel_resample = lin_coeffs_vnumel_resample[0] * xs + lin_coeffs_vnumel_resample
    axes.plot(xs, ys_vnumel_resample, label="Linear fit")
    axes.set_title("DRR vs. Resample time against CT volume size")
    axes.set_xlabel("CT volume element count")
    axes.set_ylabel("Evaluation time [s]")
    plt.legend()
    plt.savefig("data/temp/against_volume_size.svg")

    fig, axes = plt.subplots()
    axes.scatter(x_ray_numels, drr_times, label="DRR")
    axes.scatter(x_ray_numels, resample_times, label="Grangeat resampling")
    xs = np.array(plt.xlim())
    ys_xnumel_drr = lin_coeffs_xnumel_drr[0] * xs + lin_coeffs_xnumel_drr
    axes.plot(xs, ys_xnumel_drr, label="Linear fit")
    ys_xnumel_resample = lin_coeffs_xnumel_resample[0] * xs + lin_coeffs_xnumel_resample
    axes.plot(xs, ys_xnumel_resample, label="Linear fit")
    axes.set_title("DRR vs. Resample time against DRR size")
    axes.set_xlabel("DRR element count")
    axes.set_ylabel("Evaluation time [s]")
    plt.legend()
    plt.savefig("data/temp/against_drr_size.svg")

    plt.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    # parser.add_argument(
    #     "-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    main()
