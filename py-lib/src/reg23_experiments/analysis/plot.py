import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt

from reg23_experiments.data.structs import Sinogram3dGrid

__all__ = ["separate_subplots", "visualise_planes_as_points"]


def separate_subplots(n_rows: int, n_cols: int, **fig_kwargs) -> tuple[np.ndarray, np.ndarray]:
    figs = np.empty((n_rows, n_cols), dtype=object)
    axs = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            figs[i, j] = plt.figure(**fig_kwargs)
            axs[i, j] = figs[i, j].add_subplot(111)
    return figs, axs


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
