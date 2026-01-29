import logging

import numpy as np
import torch
import pandas as pd

__all__ = ["to_latex_scientific", "save_colourmap_for_latex", "get_axis_values_if_dataframe_rectangular_over_columns",
           "dataframe_rectangular_columns_to_tensor"]

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


def get_axis_values_if_dataframe_rectangular_over_columns(  #
        df: pd.DataFrame, *, columns: list[str]) -> (list[np.ndarray] | None):
    # full MultiIndex for unique values
    full_index = pd.MultiIndex.from_product([df[col].unique().tolist() for col in columns], names=columns)
    # multiIndex from the data
    df_index = pd.MultiIndex.from_frame(df[columns])
    # check if the index from the data matches the full index
    if not df_index.isin(full_index).all() or len(df_index) != len(full_index):
        return None
    # if it does, extract the values along each axis
    df_index = df_index.sort_values()
    return [df_index.levels[df_index.names.index(name)].to_numpy() for name in columns]


def dataframe_rectangular_columns_to_tensor(df: pd.DataFrame, *, ordered_axes: list[str], value_column: str) -> tuple[
    torch.Tensor, dict[str, np.ndarray]]:
    # set the index to be a MultiIndex derived from the existing columns named in `ordered_axes`, then take just the
    # series for the `value_column`, and sort the rows by the index.
    s: pd.Series = df.set_index(ordered_axes)[value_column].sort_index()
    # use the `levels` property of the MultiIndex to get an Index object containing the unique values for each level in
    # a list. Then extract these Index objects from the list in the order of the names given in `ordered_axes`.
    axis_index_objects: list[pd.Index] = [  #
        s.index.levels[s.index.names.index(name)]  #
        for name in ordered_axes  #
    ]
    # create a MultiIndex object for the full grid of values, with every combination of the values from each axis.
    full_index = pd.MultiIndex.from_product([e.tolist() for e in axis_index_objects], names=ordered_axes)
    # re-index the DataFrame with the full index
    s = s.reindex(full_index)
    # check for missing values - these will be populated with nans.
    if s.isna().any():
        logger.warning("Grid is incomplete — missing coordinate combinations.")
    # get the unique values of each axis from the index objects and store in a dict to return
    axis_values: dict[str, np.ndarray] = {  #
        name: index.to_numpy()  #
        for name, index in zip(ordered_axes, axis_index_objects)  #
    }
    # get the length of each axis from the index objects
    axis_lengths = [len(index) for index in axis_index_objects]
    # convert the DataFrame to a flat tensor and view with the lengths of each axis
    tensor = torch.from_numpy(s.to_numpy()).view(*axis_lengths)
    return tensor, axis_values
