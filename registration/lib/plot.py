from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np
import torch
import pyvista as pv

from registration.lib.structs import Sinogram3dGrid


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


def torch_polyfit(xs: torch.Tensor, ys: torch.Tensor, *, degree: int = 1) -> torch.Tensor:
    assert xs.size() == ys.size()

    A = torch.stack([xs.pow(i) for i in range(degree + 1)], dim=1)  # size = (n, degree + 1)
    B = ys.unsqueeze(1)

    ret = torch.linalg.lstsq(A, B)

    return ret.solution


def fit_power_relationship(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    y = a * x^b
    :param xs:
    :param ys:
    :return: (2,) tensor containing [a, b]
    """
    assert xs.size() == ys.size()
    assert (xs > 0.0).all()
    assert (ys > 0.0).all()

    ln_xs = xs.log()
    ln_ys = ys.log()
    coefficients = torch_polyfit(ln_xs, ln_ys, degree=1)
    return torch.tensor([coefficients[0].exp(), coefficients[1]])


def save_colourmap_for_latex(filename: str, colourmap: torch.Tensor):
    colourmap = colourmap.clone().cpu()
    indices = [torch.arange(0, n, 1) for n in colourmap.size()]
    indices = torch.meshgrid(indices)
    rows = indices[::-1] + (colourmap,)
    rows = torch.stack([row.flatten() for row in rows], dim=-1)
    np.savetxt("data/temp/{}.dat".format(filename), rows.numpy())


def visualise_planes_as_points(grid: Sinogram3dGrid, scalars: torch.Tensor | None):
    ct = grid.theta.cos().flatten()
    st = grid.theta.sin().flatten()
    cp = grid.phi.cos().flatten()
    sp = grid.phi.sin().flatten()
    points = grid.r.flatten().unsqueeze(-1) * torch.stack((ct * cp, ct * sp, st), dim=-1)
    pl = pv.Plotter()
    if scalars is None:
        pl.add_points(points.cpu().numpy())
    else:
        pl.add_points(points.cpu().numpy(), scalars=scalars.flatten().cpu().numpy())
    pl.show()


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