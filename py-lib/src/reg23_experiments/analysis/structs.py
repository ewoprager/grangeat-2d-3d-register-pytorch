from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np
import torch
import scipy

from reg23_experiments.analysis.helpers import to_latex_scientific
from reg23_experiments.analysis.fit import torch_polyfit

__all__ = ["Series", "DataFit", "LinearFit", "QuadraticFit", "PowerFit", "CustomFitPowerRatio"]


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
    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str, **kwargs) -> None:
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

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str, **kwargs) -> None:
        ys_fit = self.coefficients[0] + self.coefficients[1] * xs + self._series.origin_y_offset
        axes.plot(xs.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()),
                  **kwargs)

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

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str, **kwargs) -> None:
        ys_fit = self.coefficients[0] + self.coefficients[1] * xs + self.coefficients[
            2] * xs * xs + self._series.origin_y_offset
        axes.plot(xs.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()),
                  **kwargs)

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

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str, **kwargs) -> None:
        xs_fit = xs[1:] if self._series.intersects_origin else xs
        ys_fit = self.coefficients[0] * xs_fit.pow(self.coefficients[1]) + self._series.origin_y_offset
        axes.plot(xs_fit.cpu().numpy(), ys_fit.cpu().numpy(), label="{}: {}".format(label_prefix, self.text_equation()),
                  **kwargs)

    def scatter_log_log(self, axes, **kwargs) -> None:
        self._linear_fit_log_log.scatter(axes, **kwargs)

    def plot_fit_log_log(self, axes, label_prefix: str) -> None:
        self._linear_fit_log_log.generate_and_plot_ys(axes=axes, xs=self._linear_fit_log_log.xs,
                                                      label_prefix=label_prefix)

    def log_log_correlation_coefficient(self) -> torch.Tensor:
        return self._linear_fit_log_log.correlation_coefficient()


class CustomFitPowerRatio(DataFit):
    def __init__(self, *, xs: torch.Tensor, series: Series):
        self._xs = xs
        self._series = series

        independent_variable = (xs[1:-1] if series.intersects_origin else xs[:-1]).cpu().numpy()
        dependent_variable = ((series.ys[1:-1] if series.intersects_origin else series.ys[
            :-1]) - series.origin_y_offset).cpu().numpy()
        initial_parameters = np.array([3.0, 0.2])
        result = scipy.optimize.least_squares(  #
            fun=CustomFitPowerRatio._function,  #
            x0=initial_parameters,  #
            jac=CustomFitPowerRatio._jacobian,  #
            bounds=scipy.optimize.Bounds(lb=np.zeros(2), keep_feasible=True),  #
            args=(independent_variable, dependent_variable),  #
            verbose=1)
        print(result)
        self._parameters = result.x

    def text_equation(self) -> str:
        return "({}, {})".format(self._parameters[0], self._parameters[1])

    def generate_ys(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.tensor(CustomFitPowerRatio._model(self._parameters, (
            xs[1:] if self._series.intersects_origin else xs).cpu().numpy()))

    def generate_and_plot_ys(self, *, axes, xs: torch.Tensor, label_prefix: str, **kwargs) -> None:
        xs_fit = (xs[1:] if self._series.intersects_origin else xs).cpu().numpy()
        ys_fit = CustomFitPowerRatio._model(self._parameters, xs_fit) + self._series.origin_y_offset
        axes.plot(xs_fit, ys_fit, label="{}: {}".format(label_prefix, self.text_equation()), **kwargs)

    @staticmethod
    def _model(parameters: np.ndarray, independent_variable: np.ndarray) -> np.ndarray:
        x_a = np.pow(independent_variable, parameters[0])
        bx = parameters[1] * independent_variable
        return (x_a + bx) / (2.0 - x_a + bx)

    @staticmethod
    def _function(parameters: np.ndarray, independent_variable: np.ndarray,
                  dependent_variable: np.ndarray) -> np.ndarray:
        return CustomFitPowerRatio._model(parameters, independent_variable) - dependent_variable

    @staticmethod
    def _jacobian(parameters: np.ndarray, independent_variable: np.ndarray,
                  dependent_variable: np.ndarray) -> np.ndarray:
        ret = np.empty((independent_variable.size, parameters.size))
        x_a = np.pow(independent_variable, parameters[0])  # size matches that of independent_variable
        bx = parameters[1] * independent_variable  # size matches that of independent_variable
        den = 2.0 - x_a + bx  # size matches that of independent_variable
        den = den * den  # size matches that of independent_variable
        ret[:, 0] = 2.0 * (1.0 + bx) * x_a * np.log(independent_variable) / den
        ret[:, 1] = 2.0 * (1.0 - x_a) * independent_variable / den
        return ret
