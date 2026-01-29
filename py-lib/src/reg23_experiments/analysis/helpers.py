import logging

import numpy as np
import torch
import pandas as pd

__all__ = ["to_latex_scientific", "save_colourmap_for_latex", "dataframe_to_tensor"]

logger = logging.getLogger(__name__)


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


def save_colourmap_for_latex(filename: str, colourmap: torch.Tensor):
    colourmap = colourmap.clone().cpu()
    indices = [torch.arange(0, n, 1) for n in colourmap.size()]
    indices = torch.meshgrid(indices)
    rows = indices[::-1] + (colourmap,)
    rows = torch.stack([row.flatten() for row in rows], dim=-1)
    np.savetxt("data/temp/{}.dat".format(filename), rows.numpy())


def dataframe_to_tensor(df: pd.DataFrame, *, ordered_axes: list[str], value_column: str) -> tuple[
    torch.Tensor, dict[str, np.ndarray]]:
    s = df.set_index(ordered_axes)[value_column].sort_index()
    axis_levels = [  #
        s.index.levels[s.index.names.index(name)]  #
        for name in ordered_axes  #
    ]
    full_index = pd.MultiIndex.from_product(axis_levels, names=ordered_axes)
    s = s.reindex(full_index)
    if s.isna().any():
        logger.warning("Grid is incomplete — missing coordinate combinations.")
    axis_values = {  #
        name: level.to_numpy()  #
        for name, level in zip(ordered_axes, axis_levels)  #
    }
    axis_lengths = [len(level) for level in axis_levels]
    tensor = torch.from_numpy(s.to_numpy()).view(*axis_lengths)
    return tensor, axis_values
