from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.odr import quadratic
from torchvision.transforms.v2.functional import horizontal_flip

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


def sample_cosine_in_cuboid_at_fraction(*, full_vector: np.ndarray, fraction: float,
                                        gen: np.random.Generator = np.random.default_rng()) -> Tuple[float, float]:
    point: np.ndarray = full_vector * gen.beta(a=2.0 * fraction, b=2.0 * (1.0 - fraction), size=full_vector.shape)

    # scale the point in the `1` direction to adjust the distribution of fractions
    # - This results in an approximately uniform distribution of fractions, independently of the dimensionality)
    # - This does not affect the relationship between fraction and cosine, only the distribution of samples taken over the fraction value
    # !!! this is currently wrong !!!
    # old_fraction: float = point.sum() / full_vector.sum()
    # k = 4.9 * np.sqrt(float(full_vector.size))
    # fraction = 1.0 / (1.0 + np.exp(-k * (old_fraction - 0.5)))
    # point *= fraction / old_fraction

    # ?
    # scale the point towards the `full_vector`, keeping it in the plane of constant `fraction` (perpendicular to the `1` direction)
    # - This **does** affect the relationship between fraction and cosine.
    #

    cosine: float = np.corrcoef(point, full_vector)[0, 1].item()
    return cosine, point.sum() / full_vector.sum()


def main2():
    dimensionality: int = 1000
    gen = np.random.default_rng()
    full_vector: np.ndarray = gen.uniform(low=0.0, high=1.0, size=dimensionality)
    sample_count: int = 10_000
    cosines = np.zeros(sample_count)
    fractions = np.zeros(sample_count)
    for i in tqdm(range(sample_count)):
        cosines[i], fractions[i] = sample_cosine_in_cuboid_at_fraction(full_vector=full_vector,
                                                                       fraction=float(i + 1) / float(sample_count + 1),
                                                                       gen=gen)

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(1.0 - fractions)
    axes[0].set_xlabel("truncation fraction")

    xs = 1.0 - fractions
    ys = -cosines
    bins = np.linspace(min(xs), max(xs), 14)
    bin_indices = np.digitize(xs, bins)
    width = 0.5 * (bins[1] - bins[0])
    axes[1].boxplot([ys[bin_indices == i] for i in range(1, len(bins) + 1)], positions=bins, widths=width)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axes[1].set_xlabel("truncation fraction")
    axes[1].set_ylabel("-$\\cos \\theta$")

    medians = np.array([np.median(ys[bin_indices == i]) for i in range(1, len(bins) + 1)])
    to_fit_xs = torch.tensor(bins[:-3])
    to_fit_ys = torch.tensor(medians[:-3])
    power_fit = PowerFit.build(xs=to_fit_xs, series=Series(ys=to_fit_ys, intersects_origin=True, origin_y_offset=-1.0))
    power_fit.generate_and_plot_ys(axes=axes[1], xs=to_fit_xs, label_prefix="Power fit")
    plt.show()


def sample_at_fraction(*, full_vector: np.ndarray, fraction: float, dimensionality: int,
                       gen: np.random.Generator = np.random.default_rng()) -> float:
    point: np.ndarray = full_vector * gen.beta(a=2.0 * fraction, b=2.0 * (1.0 - fraction), size=dimensionality)
    return point.sum() / full_vector.sum()


def beta_for_truncation_fraction():
    gen = np.random.default_rng()
    sample_count_per_point: int = 100_000
    fraction_count = 23
    fractions = np.linspace(1e-6, 1.0 - 1e-6, fraction_count)
    dims = np.array([1, 10, 100, 1000, 10_000], dtype=np.int64)
    medians = np.zeros((len(dims), len(fractions)))
    q1s = np.zeros((len(dims), len(fractions)))
    q3s = np.zeros((len(dims), len(fractions)))
    stds = np.zeros((len(dims), len(fractions)))
    for j in range(len(dims)):
        sample_count: int = int(round(float(sample_count_per_point / np.sqrt(float(dims[j])))))
        full_vector: np.ndarray = gen.uniform(low=0.0, high=1.0, size=dims[j].item())
        for i in tqdm(range(len(fractions))):
            v = np.zeros(sample_count)
            for k in range(sample_count):
                v[k] = sample_at_fraction(full_vector=full_vector, fraction=fractions[i].item(),
                                          dimensionality=dims[j].item(), gen=gen)
            medians[j, i] = np.quantile(v, 0.5)
            q1s[j, i] = np.quantile(v, 0.25)
            q3s[j, i] = np.quantile(v, 0.75)
            stds[j, i] = v.std()

    colors = np.stack([np.linspace(0.0, 1.0, len(dims)), np.linspace(1.0, 0.0, len(dims)), np.zeros(len(dims))])
    colors = [(r.item(), g.item(), b.item()) for r, g, b in zip(colors[0, :], colors[1, :], colors[2, :])]

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    for i in range(len(dims)):
        axes[0].plot(fractions, medians[i, :], color=colors[i], label="D = {}".format(dims[i]))
        axes[0].plot(fractions, q1s[i, :], color=colors[i], linestyle='--', label="Quartiles")
        axes[0].plot(fractions, q3s[i, :], color=colors[i], linestyle='--')
    axes[0].set_xlabel("$f^*$")
    axes[0].set_ylabel("$f$")
    axes[0].legend()

    # xs_fit = torch.tensor(fractions)
    # ys_fit = torch.tensor(stds[0, :])
    # quadratic_fit = QuadraticFit(xs=xs_fit, series=Series(ys=ys_fit, intersects_origin=True, origin_y_offset=0.0))
    # quadratic_fit.generate_and_plot_ys(axes=axes[1], xs=xs_fit, label_prefix="Power fit", color=colors[0])

    half_fraction_count = fraction_count // 2
    xs_fit = torch.tensor(fractions[half_fraction_count:])
    horizontal_offset = -xs_fit[0].item()
    xs_fit += horizontal_offset
    ys_fit = torch.tensor(stds[0, :][half_fraction_count:])
    maximum = ys_fit[0]
    ys_fit = maximum - ys_fit
    power_fit = PowerFit.build(xs=xs_fit, series=Series(ys=ys_fit, intersects_origin=True, origin_y_offset=0.0))
    if power_fit is not None:
        ys = power_fit.generate_ys(xs=xs_fit)
        axes[1].plot(xs_fit[1:] - horizontal_offset, maximum - ys,
                     label="Power fit: {}".format(power_fit.text_equation()))
    else:
        print("Power fit failed to build.")

    axes[1].scatter(fractions, stds[0, :], marker='x', color=colors[0])
    axes[1].set_xlabel("$f^*$")
    axes[1].set_xlabel("$f$")
    axes[1].set_title("$\\sigma$ for $D = {}$".format(dims[0]))
    axes[1].legend()

    axes[2].scatter(fractions, stds[-1, :], marker='x', color=colors[-1])
    axes[2].set_xlabel("$f^*$")
    axes[2].set_xlabel("$f$")
    axes[2].set_title("$\\sigma$ for $D = {}$".format(dims[-1]))
    axes[2].legend()

    xs_fit = torch.tensor(dims.astype(np.float64))
    ys_fit = torch.tensor(stds.max(axis=1))
    power_fit = PowerFit.build(xs=xs_fit, series=Series(ys=ys_fit, intersects_origin=False, origin_y_offset=0.0))
    power_fit.generate_and_plot_ys(axes=axes[3], xs=xs_fit, label_prefix="Power fit")
    axes[3].scatter(xs_fit.numpy(), ys_fit.numpy(), marker='x')
    axes[3].set_xlabel("dimensionality")
    axes[3].set_ylabel("$\\sigma_{max}$")
    axes[3].legend()

    plt.show()


if __name__ == "__main__":
    print("Hello, world!")
    beta_for_truncation_fraction()
