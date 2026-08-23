import logging

import numpy as np
import torch

__all__ = ["save_colourmap_for_latex"]

logger = logging.getLogger(__name__)


def save_colourmap_for_latex(filename: str, colourmap: torch.Tensor):
    colourmap = colourmap.clone().cpu()
    indices = [torch.arange(0, n, 1) for n in colourmap.size()]
    indices = torch.meshgrid(indices)
    rows = indices[::-1] + (colourmap,)
    rows = torch.stack([row.flatten() for row in rows], dim=-1)
    np.savetxt("data/temp/{}.dat".format(filename), rows.numpy())
