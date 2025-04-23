import numpy as np
import torch
import pyvista as pv

from registration.lib.structs import *


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