import argparse
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib

from notification import logs_setup
from registration.lib.plot import fit_power_relationship, to_latex_scientific, torch_polyfit


class Series(NamedTuple):
    ys: torch.Tensor
    intersects_origin: bool
    origin_y_offset: float = 0.0


class DataFit(ABC):
    @abstractmethod
    def text_equation(self) -> str:
        pass

    @abstractmethod
    def generate_ys(self, xs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str) -> None:
        pass


class LinearFit(DataFit):
    def __init__(self, *, xs: torch.Tensor, series: Series):
        self._xs = xs
        self._series = series
        ys_fit = series.ys - self._series.origin_y_offset
        self._coefficients = torch_polyfit(xs, ys_fit, degree=1)

    @property
    def xs(self) -> torch.Tensor:
        return self._xs

    @property
    def series(self) -> Series:
        return self._series

    @property
    def coefficients(self) -> torch.Tensor:
        return self._coefficients

    def text_equation(self) -> str:
        return "$y = {}x {}$".format(to_latex_scientific(self.coefficients[1].item()),
                                     to_latex_scientific(self.coefficients[0].item(), include_plus=True))

    def generate_ys(self, xs: torch.Tensor) -> torch.Tensor:
        return self.coefficients[0] + self.coefficients[1] * xs + self._series.origin_y_offset

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str) -> None:
        ys_fit = self.coefficients[0] + self.coefficients[1] * xs + self._series.origin_y_offset
        axes.plot(xs.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()))

    def scatter(self, axes, **kwargs) -> None:
        axes.scatter(self.xs, self.series.ys, **kwargs)

    def correlation_coefficient(self) -> torch.Tensor:
        return torch.corrcoef(torch.stack((self.xs.flatten(), self.series.ys.flatten()), dim=0))[0, 1]


class QuadraticFit(DataFit):
    def __init__(self, *, xs: torch.Tensor, series: Series):
        self._xs = xs
        self._series = series
        ys_fit = series.ys - self._series.origin_y_offset
        self._coefficients = torch_polyfit(xs, ys_fit, degree=2)

    @property
    def xs(self) -> torch.Tensor:
        return self._xs

    @property
    def series(self) -> Series:
        return self._series

    @property
    def coefficients(self) -> torch.Tensor:
        return self._coefficients

    def text_equation(self) -> str:
        return "$y = {}x^2 {}x {}$".format(to_latex_scientific(self.coefficients[2].item()),
                                           to_latex_scientific(self.coefficients[1].item(), include_plus=True),
                                           to_latex_scientific(self.coefficients[0].item(), include_plus=True))

    def generate_ys(self, xs: torch.Tensor) -> torch.Tensor:
        return self.coefficients[0] + self.coefficients[1] * xs + self.coefficients[
            2] * xs * xs + self._series.origin_y_offset

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str) -> None:
        ys_fit = self.coefficients[0] + self.coefficients[1] * xs + self.coefficients[
            2] * xs * xs + self._series.origin_y_offset
        axes.plot(xs.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()))

    def scatter(self, axes, **kwargs) -> None:
        axes.scatter(self.xs, self.series.ys, **kwargs)


class PowerFit(DataFit):
    def __init__(self, *, xs: torch.Tensor, series: Series, linear_fit_log_log: LinearFit):
        self._xs = xs
        self._series = series
        self._linear_fit_log_log = linear_fit_log_log

    @staticmethod
    def build(*, xs: torch.Tensor, series: Series) -> Optional['PowerFit']:
        xs_fit = xs[1:] if series.intersects_origin else xs
        ys_fit = (series.ys[1:] if series.intersects_origin else series.ys) - series.origin_y_offset
        if not ((xs_fit > 0.0).all() and (ys_fit > 0.0).all()):
            return None
        return PowerFit(xs=xs, series=series, linear_fit_log_log=LinearFit(  #
            xs=xs_fit.log(), series=Series(ys_fit.log(), intersects_origin=False, origin_y_offset=0.0)))

    @property
    def xs(self) -> torch.Tensor:
        return self._xs

    @property
    def series(self) -> Series:
        return self._series

    @property
    def coefficients(self) -> torch.Tensor:
        return torch.tensor([self._linear_fit_log_log.coefficients[0].exp(), self._linear_fit_log_log.coefficients[1]])

    def text_equation(self) -> str:
        return "$y = {}x^{{{}}}$".format(to_latex_scientific(self.coefficients[0].item()),
                                         to_latex_scientific(self.coefficients[1].item()))

    def generate_ys(self, xs: torch.Tensor) -> torch.Tensor:
        xs_fit = xs[1:] if self._series.intersects_origin else xs
        return self.coefficients[0] * xs_fit.pow(self.coefficients[1]) + self._series.origin_y_offset

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str) -> None:
        xs_fit = xs[1:] if self._series.intersects_origin else xs
        ys_fit = self.coefficients[0] * xs_fit.pow(self.coefficients[1]) + self._series.origin_y_offset
        axes.plot(xs_fit.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()))

    def scatter_log_log(self, axes, **kwargs) -> None:
        self._linear_fit_log_log.scatter(axes, **kwargs)

    def plot_fit_log_log(self, axes, label_prefix: str) -> None:
        self._linear_fit_log_log.generate_and_plot_ys(axes=axes, xs=self._linear_fit_log_log.xs,
                                                      label_prefix=label_prefix)

    def log_log_correlation_coefficient(self) -> torch.Tensor:
        return self._linear_fit_log_log.correlation_coefficient()


class Analysis:
    def __init__(self, *, xs: torch.Tensor, ys: torch.Tensor, dependent_common_dim: int, intersects_origin: bool,
                 origin_y_offset: float = 0.0):
        """
        Series names: 'std', 'q1', 'median', 'q3'.
        :param xs: !!! Must be evenly spaced for analysis to be valid
        :param ys:
        :param dependent_common_dim: the dimension along which the data corresponds to common values of the dependent variable (i.e. the dimension along which the means would be calculated for the mean at each value of the dependent variable)
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

        if "vals_at_gt" in pdata:
            vals_at_gt = pdata["vals_at_gt"]
            analysis = Analysis(xs=truncation_fractions, ys=vals_at_gt, dependent_common_dim=1, intersects_origin=True,
                                origin_y_offset=-1.0)

            if True:
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

            if False:
                # box whisker plot of data
                fig, axes = plt.subplots()
                analysis.box_whisker(axes=axes)
                analysis.plot_power_fit(axes=axes, series_name="q1")
                analysis.plot_power_fit(axes=axes, series_name="median")
                analysis.plot_power_fit(axes=axes, series_name="q3")
                axes.set_xlabel("truncation fraction")
                axes.set_ylabel("obj. func. value at ground truth")
                axes.set_title("{}: G.T. value box whisker".format(name))
                plt.legend()

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
