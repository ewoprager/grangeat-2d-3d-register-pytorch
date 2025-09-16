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
