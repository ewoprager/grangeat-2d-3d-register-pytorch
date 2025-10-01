from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from registration.lib.plot import Series, LinearFit, PowerFit, QuadraticFit


def initial():
    N = 100_000
    sigma = 0.001
    dimensionality = 100
    mean = np.zeros(dimensionality)
    mean[0] = 1.0
    points = np.random.multivariate_normal(mean=mean, cov=sigma * np.eye(dimensionality),
                                           size=N)  # size = (N, dimensionality)

    if False:
        # scatter plot
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()

    magnitudes = np.sqrt((points * points).sum(axis=1))  # size = (N,)

    if True:
        # histogram of magnitudes
        plt.hist(magnitudes, bins=100)
        plt.show()

    cosines = points[:, 0] / magnitudes

    if True:
        # histogram of cosines
        plt.hist(cosines, bins=100)
        plt.show()

    mean = cosines.mean()
    print("Mean cosine value = ", mean)


def avg_cosine_at_dist(*, distance: float, sigma: float, sample_count: int = 1000, dimensionality: int = 300) -> float:
    mean = np.zeros(dimensionality)
    mean[0] = distance
    points = np.random.multivariate_normal(mean=mean, cov=sigma * np.eye(dimensionality),
                                           size=sample_count)  # size = (sample_count, dimensionality)
    magnitudes = np.sqrt((points * points).sum(axis=1))  # size = (sample_count,)
    cosines = points[:, 0] / magnitudes  # size = (sample_count,)
    return cosines.mean()


def sigma_function_peak(sigma_max: float, n: int) -> np.ndarray:
    return np.concat((np.linspace(0.0, sigma_max, n // 2, endpoint=False), np.linspace(sigma_max, 0.0, n - (n // 2))))


def sigma_function_flat(sigma_max: float, central_fraction: float, n: int) -> np.ndarray:
    ramp_n = int(np.floor(0.5 * (1.0 - central_fraction) * float(n)))
    return np.concat((np.linspace(0.0, sigma_max, ramp_n, endpoint=False), np.full(n - 2 * ramp_n - 1, sigma_max),
                      np.linspace(sigma_max, 0.0, ramp_n + 1)))


def sigma_function_cosine(sigma_max: float, n: int) -> np.ndarray:
    return sigma_max * np.cos(np.linspace(-0.5 * np.pi, 0.5 * np.pi, n))


def main1():
    N = 60
    sigma_maxes = np.geomspace(0.00001, 0.01, 10)

    colors = np.stack(
        [np.linspace(0.0, 1.0, len(sigma_maxes)), np.linspace(1.0, 0.0, len(sigma_maxes)), np.zeros(len(sigma_maxes))])
    colors = [(r.item(), g.item(), b.item()) for r, g, b in zip(colors[0, :], colors[1, :], colors[2, :])]

    fig, axes = plt.subplots(1, 2)

    distances = np.linspace(1.0, 0.0, N)
    truncation_fractions = 1.0 - distances
    for j in tqdm(range(len(sigma_maxes))):
        avg_cosines = np.zeros(len(distances))
        # sigmas = sigma_function_peak(sigma_maxes[j].item(), N)
        # sigmas = sigma_function_flat(sigma_maxes[j].item(), 0.95, N)
        sigmas = sigma_function_cosine(sigma_maxes[j].item(), N)
        axes[0].plot(truncation_fractions, sigmas, color=colors[j])
        for i in range(len(distances)):
            avg_cosines[i] = avg_cosine_at_dist(distance=distances[i].item(), sigma=sigmas[i].item())
        axes[1].plot(truncation_fractions, -avg_cosines, color=colors[j])

        fit_ys = -torch.tensor(avg_cosines)[:(N // 2)]
        fit_xs = torch.tensor(truncation_fractions)[:(N // 2)]
        valid = (fit_ys + 1.0) > 0.0
        fit_ys = fit_ys[valid]
        fit_xs = fit_xs[valid]
        # power_fit = PowerFit.build(xs=fit_xs, series=Series(ys=fit_ys, intersects_origin=False, origin_y_offset=-1.0))
        # if power_fit is not None:
        #     power_fit.generate_and_plot_ys(axes=axes[1], xs=fit_xs, label_prefix="Power fit", color=colors[j],
        #                                    linestyle="--")
        quadratic_fit = QuadraticFit(xs=fit_xs, series=Series(ys=fit_ys, intersects_origin=False, origin_y_offset=-1.0))
        quadratic_fit.generate_and_plot_ys(axes=axes[1], xs=fit_xs, label_prefix="Power fit", color=colors[j],
                                           linestyle=":")

    axes[0].set_xlabel("truncation fraction")
    axes[0].set_ylabel("$\\sigma$")
    axes[1].set_xlabel("truncation fraction")
    axes[1].set_ylabel("-$\\cos \\theta$ average")
    plt.show()


def sample_cosine_in_cuboid(*, full_vector: np.ndarray, gen: np.random.Generator = np.random.default_rng()) -> Tuple[
    float, float]:
    point: np.ndarray = gen.uniform(low=0.0, high=full_vector)
    old_fraction: float = point.sum() / full_vector.sum()
    k = 90.0
    fraction = 1.0 / (1.0 + np.exp(-k * (old_fraction - 0.5)))
    point *= fraction / old_fraction
    cosine: float = np.corrcoef(point, full_vector)[0, 1].item()
    return cosine, fraction


def main2():
    dimensionality: int = 300
    gen = np.random.default_rng()
    full_vector: np.ndarray = gen.uniform(low=0.0, high=1.0, size=dimensionality)
    sample_count: int = 100_000
    cosines = np.zeros(sample_count)
    fractions = np.zeros(sample_count)
    for i in tqdm(range(sample_count)):
        cosines[i], fractions[i] = sample_cosine_in_cuboid(full_vector=full_vector, gen=gen)

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(1.0 - fractions)
    axes[0].set_xlabel("truncation fraction")
    axes[1].scatter(1.0 - fractions, -cosines)
    axes[1].set_xlabel("truncation fraction")
    axes[1].set_ylabel("-$\\cos \\theta$")
    plt.show()


if __name__ == "__main__":
    print("Hello, world!")
    main2()
