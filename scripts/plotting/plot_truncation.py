import argparse
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib

from reg23_experiments.utils import logs_setup
from reg23_experiments.analysis.helpers import to_latex_scientific
from reg23_experiments.analysis.structs import Series, LinearFit, PowerFit, QuadraticFit
from reg23_experiments.analysis.fit import fit_power_relationship, torch_polyfit


class Analysis:
    def __init__(self, *, xs: torch.Tensor, ys: torch.Tensor, dependent_common_dim: int, intersects_origin: bool,
                 origin_y_offset: float = 0.0):
        """
        Series names: 'std', 'q1', 'median', 'q3'.
        :param xs: !!! Must be evenly spaced for analysis to be valid
        :param ys:
        :param dependent_common_dim: the dimension along which the data corresponds to common values of the dependent
        variable (i.e. the dimension along which the means would be calculated for the mean at each value of the
        dependent variable)
        """
        self._xs = xs
        self._ys = ys
        self._dependent_common_dim = dependent_common_dim
        self._intersects_origin = intersects_origin
        self._transformed_series: dict[str, Series] = {  #
            "std": Series(self._ys.std(dim=dependent_common_dim), intersects_origin, 0.0),  #
            "q1": Series(self._ys.quantile(q=0.25, dim=dependent_common_dim), intersects_origin, origin_y_offset),  #
            "median": Series(self._ys.quantile(q=0.5, dim=dependent_common_dim), intersects_origin, origin_y_offset),  #
            "q3": Series(self._ys.quantile(q=0.75, dim=dependent_common_dim), intersects_origin, origin_y_offset)}
        self._linear_fits: dict[str, LinearFit] = {  #
            key: LinearFit(xs=self._xs, series=series) for key, series in self._transformed_series.items()}
        self._power_fits: dict[str, PowerFit | None] = {  #
            key: PowerFit.build(xs=self._xs, series=series) for key, series in self._transformed_series.items()}
        self._quadratic_fits: dict[str, QuadraticFit] = {  #
            key: QuadraticFit(xs=self._xs, series=series) for key, series in self._transformed_series.items()}

    def box_whisker(self, *, axes) -> None:
        spacing = (self._xs[1] - self._xs[0]).item()
        axes.boxplot((self._ys.t() if self._dependent_common_dim == 1 else self._ys).cpu().numpy(),
                     positions=self._xs.cpu().numpy(), widths=0.5 * spacing)
        axes.set_xlim(self._xs[0] - spacing, self._xs[-1] + spacing)

    def scatter_series(self, *, axes, series_name: str) -> None:
        assert series_name in self._transformed_series
        axes.scatter(self._xs.cpu().numpy(), self._transformed_series[series_name].ys.cpu().numpy(), label=series_name)

    def plot_log_log_fit(self, *, axes, series_name: str) -> None:
        assert series_name in self._power_fits
        if self._power_fits[series_name] is None:
            logger.warning("Power fit not available for series '{}'.".format(series_name))
            return
        self._power_fits[series_name].scatter_log_log(axes, label=series_name)
        self._power_fits[series_name].plot_fit_log_log(axes, label_prefix="{} fit".format(series_name))
        logger.info("Log-log plot of {} has correlation coefficient of {:.4f}.".format(series_name, self._power_fits[
            series_name].log_log_correlation_coefficient()))

    def plot_linear_fit(self, *, axes, series_name: str) -> None:
        assert series_name in self._linear_fits
        self._linear_fits[series_name].generate_and_plot_ys(axes=axes, xs=self._xs,
                                                            label_prefix="{} linear fit".format(series_name))
        logger.info("Linear plot of {} has correlation coefficient of {:.4f}.".format(series_name, self._linear_fits[
            series_name].correlation_coefficient()))

    def plot_power_fit(self, *, axes, series_name: str) -> None:
        assert series_name in self._power_fits
        if self._power_fits[series_name] is None:
            logger.warning("Power fit not available for series '{}'.".format(series_name))
            return
        self._power_fits[series_name].generate_and_plot_ys(axes=axes, xs=self._xs,
                                                           label_prefix="{} power fit".format(series_name))

    def plot_quadratic_fit(self, *, axes, series_name: str) -> None:
        assert series_name in self._quadratic_fits
        self._quadratic_fits[series_name].generate_and_plot_ys(axes=axes, xs=self._xs,
                                                               label_prefix="{} linear fit".format(series_name))


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

        if True and "vals_at_gt" in pdata:
            vals_at_gt = pdata["vals_at_gt"]
            analysis = Analysis(xs=truncation_fractions, ys=vals_at_gt, dependent_common_dim=1, intersects_origin=True,
                                origin_y_offset=-1.0)

            if False:
                # log log plots of quantiles
                fig, axes = plt.subplots()
                analysis.plot_log_log_fit(axes=axes, series_name="q1")
                analysis.plot_log_log_fit(axes=axes, series_name="median")
                analysis.plot_log_log_fit(axes=axes, series_name="q3")
                axes.set_xlabel("ln(truncation fraction)")
                axes.set_ylabel("ln(obj. func. value at ground truth)")
                axes.set_title("{}: G.T. value average log logs".format(name))
                plt.legend()

                # quadratic plots of quantiles
                fig, axes = plt.subplots()
                analysis.scatter_series(axes=axes, series_name="q1")
                analysis.plot_quadratic_fit(axes=axes, series_name="q1")
                analysis.scatter_series(axes=axes, series_name="median")
                analysis.plot_quadratic_fit(axes=axes, series_name="median")
                analysis.scatter_series(axes=axes, series_name="q3")
                analysis.plot_quadratic_fit(axes=axes, series_name="q3")
                axes.set_xlabel("truncation fraction")
                axes.set_ylabel("obj. func. value at ground truth")
                axes.set_title("{}: G.T. value average quadratic fits".format(name))
                plt.legend()

                # scatter of standard deviation
                fig, axes = plt.subplots()
                analysis.scatter_series(axes=axes, series_name="std")
                analysis.plot_linear_fit(axes=axes, series_name="std")
                analysis.plot_power_fit(axes=axes, series_name="std")
                axes.set_xlabel("truncation fraction")
                axes.set_ylabel("obj. func. value at ground truth")
                axes.set_title("{}: G.T. value standard deviations".format(name))
                plt.legend()

                # quadratic plots of standard deviation
                fig, axes = plt.subplots()
                analysis.scatter_series(axes=axes, series_name="std")
                analysis.plot_quadratic_fit(axes=axes, series_name="std")
                axes.set_xlabel("truncation fraction")
                axes.set_ylabel("obj. func. value at ground truth")
                axes.set_title("{}: G.T. value std. dev. quadratic fit".format(name))
                plt.legend()

            if True:
                # box whisker plot of data
                fig, axes = plt.subplots()
                analysis.box_whisker(axes=axes)
                # analysis.plot_power_fit(axes=axes, series_name="q1")
                # analysis.plot_power_fit(axes=axes, series_name="median")
                # analysis.plot_power_fit(axes=axes, series_name="q3")

                xs = truncation_fractions.cpu().numpy()
                ys = vals_at_gt.mean(dim=1).cpu().numpy()
                xs_fit = xs * ys * ys
                ys_fit = 1.0 - xs - ys * ys
                sum_x2 = (xs_fit * xs_fit).sum()
                sum_xy = (xs_fit * ys_fit).sum()
                m = sum_xy / sum_x2

                ys_fitted = m * xs_fit
                ys_fitted_plot = -np.sqrt(np.abs(1.0 - xs - ys_fitted))
                axes.plot(xs, ys_fitted_plot,
                          label="theoretical fit: $y = -\\sqrt{{ \\frac{{ 1 - x }}{{ 1 {} x }} }}$".format(
                              to_latex_scientific(m, include_plus=True)))

                axes.set_xlabel("truncation fraction")
                axes.set_ylabel("obj. func. value at ground truth")
                axes.set_title("{}: G.T. value box whisker".format(name))
                axes.legend()

        if False and "opt_distances" in pdata:
            opt_distances = pdata["opt_distances"]
            analysis = Analysis(xs=truncation_fractions, ys=opt_distances, dependent_common_dim=1,
                                intersects_origin=True)

            # log log plots of quantiles
            fig, axes = plt.subplots()
            analysis.plot_log_log_fit(axes=axes, series_name="q1")
            analysis.plot_log_log_fit(axes=axes, series_name="median")
            analysis.plot_log_log_fit(axes=axes, series_name="q3")
            axes.set_xlabel("ln(truncation fraction)")
            axes.set_ylabel("ln(obj. func. value at ground truth)")
            axes.set_title("{}: distance in SE(3) between G.T. and optimum avg. log logs".format(name))
            plt.legend()

            # box whisker plot of data
            fig, axes = plt.subplots()
            analysis.box_whisker(axes=axes)
            analysis.plot_power_fit(axes=axes, series_name="q1")
            analysis.plot_power_fit(axes=axes, series_name="median")
            analysis.plot_power_fit(axes=axes, series_name="q3")
            axes.set_xlabel("truncation fraction")
            axes.set_ylabel("obj. func. value at ground truth")
            axes.set_title("{}: distance in SE(3) between G.T. and optimum box whisker".format(name))
            plt.legend()

            # scatter of standard deviation
            fig, axes = plt.subplots()
            analysis.scatter_series(axes=axes, series_name="std")
            analysis.plot_power_fit(axes=axes, series_name="std")
            axes.set_xlabel("truncation fraction")
            axes.set_ylabel("obj. func. value at ground truth")
            axes.set_title("{}: distance in SE(3) between G.T. and optimum std. dev.".format(name))
            plt.legend()

        if False and "pdf_means" in pdata:
            fig, axes = plt.subplots()
            pdf_means = pdata["pdf_means"].mean(axis=1)

            xs = (1.0 - truncation_fractions).cpu().numpy()
            ys = pdf_means.cpu().numpy()
            axes.scatter(xs, ys)

            linear_fit_ys = np.exp(ys) - 1.0
            sum_xy = np.sum(xs * linear_fit_ys)
            sum_x2 = np.sum(xs * xs)
            m = sum_xy / sum_x2
            fitted_ys = np.log(1.0 + m * xs)
            axes.plot(xs, fitted_ys, label="fit: $y = \\log (1 + {}x)$".format(to_latex_scientific(m)))

            axes.set_xlabel("1 - truncation fraction")
            axes.set_ylabel("intensity ratio")
            axes.set_title("{}: mean intensity ratio".format(name))
            axes.legend()

        if False and "pdf_stds" in pdata:
            fig, axes = plt.subplots()
            pdf_stds = pdata["pdf_stds"].mean(axis=1)

            xs = (1.0 - truncation_fractions).cpu().numpy()
            ys = pdf_stds.cpu().numpy()
            axes.scatter(xs, ys)

            linear_fit_ys = np.exp(ys) - 1.0
            sum_xy = np.sum(xs * linear_fit_ys)
            sum_x2 = np.sum(xs * xs)
            m = sum_xy / sum_x2
            fitted_ys = np.log(1.0 + m * xs)
            axes.plot(xs, fitted_ys, label="fit: $y = \\log (1 + {}x)$".format(to_latex_scientific(m)))

            axes.set_xlabel("1 - truncation fraction")
            axes.set_ylabel("intensity ratio")
            axes.set_title("{}: std. dev. intensity ratio".format(name))
            axes.legend()

        if False:
            plt.show()
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
