import logging
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch

__all__ = ["get_axis_values_if_dataframe_rectangular_over_columns", "dataframe_rectangular_columns_to_tensor",
           "dataframe_to_cartesian_zipped_tensors", "CartesianZippedTensors"]

logger = logging.getLogger(__name__)


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
    torch.Tensor, list[tuple[str, np.ndarray]]]:
    # set the index to be a MultiIndex derived from the existing columns named in `ordered_axes`, then take just the
    # series for the `value_column`, and sort the rows by the index.
    s: pd.Series = df.set_index(ordered_axes)[value_column].sort_index()
    # use the `levels` property of the MultiIndex to get an Index object containing the unique values for each level in
    # a list. Then extract these Index objects from the list in the order of the names given in `ordered_axes`.
    if s.index.nlevels == 1:
        axis_index_objects: list[pd.Index] = [s.index]
    else:
        axis_index_objects: list[pd.Index] = [  #
            s.index.get_level_values(name).unique()  #
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
    axis_values: list[tuple[str, np.ndarray]] = [  #
        (name, index.to_numpy())  #
        for name, index in zip(ordered_axes, axis_index_objects)  #
    ]
    # get the length of each axis from the index objects
    axis_lengths = [len(index) for index in axis_index_objects]
    # convert the DataFrame to a flat tensor and view with the lengths of each axis
    tensor = torch.from_numpy(s.to_numpy()).view(*axis_lengths)
    return tensor, axis_values


class CartesianZippedTensors(NamedTuple):
    dependent_variable_tensors: dict[str, torch.Tensor]
    cartesian_axes_values: list[tuple[str, np.ndarray]]
    zipped_axis_values: list[tuple[str, np.ndarray]]
    constant_values: dict[str, Any]


def dataframe_to_cartesian_zipped_tensors(  #
        df: pd.DataFrame,  #
        *,  #
        cartesian_variables: list[str],  #
        dependent_variables: list[str],  #
) -> CartesianZippedTensors:
    # Remove constant variables
    constant_values = df.loc[:, df.nunique() == 1].iloc[0].to_dict()
    constant_variables = list(constant_values.keys())
    df = df.drop(columns=constant_variables)

    # Construct a MultiIndex from the values in the Cartesian columns
    cartesian_index = pd.MultiIndex.from_product([df[v].unique() for v in cartesian_variables],
                                                 names=cartesian_variables)

    # Find potentially zipped variables
    zipped_variables: list[str] = [  #
        v for v in df.columns  #
        if v not in cartesian_variables and v not in dependent_variables  #
    ]

    if zipped_variables:
        # Use the cartesian variables, and then the zipped variables as a MultiIndex
        df = df.set_index(cartesian_variables + zipped_variables).sort_index()

        # Extract the rows for the first Cartesian variable combination, and get their index
        first_cartesian_rows = df.loc[tuple(df.index.get_level_values(v)[0] for v in cartesian_variables)].sort_index()
        zipped_axis_values = [  #
            (v, first_cartesian_rows.index.get_level_values(v).to_numpy())  #
            for v in zipped_variables  #
        ]
        zipped_index = first_cartesian_rows.index

        # The zipped indices should match for every Cartesian variable combination; take the Cartesian product of the
        # two
        # indices
        full_index = pd.MultiIndex.from_tuples(  #
            [(*x, *y) for x in cartesian_index for y in zipped_index],  #
            names=[*cartesian_index.names, *zipped_index.names]  #
        )

        df = df.reindex(full_index).sort_index()
    else:
        # There are no potentially zipped variables, so reindex with the cartesian variables
        df = df.set_index(cartesian_variables)
        df = df.reindex(cartesian_index).sort_index()
        zipped_axis_values = []

    # check for missing values - these will have been populated with nans.
    if df.isna().any().any():
        logger.warning("Grid is incomplete — missing coordinate combinations.")

    # get the unique values of each Cartesian axis and store as an ordered list of named value arrays
    cartesian_axws_values: list[tuple[str, np.ndarray]] = [  #
        (v, df.index.get_level_values(v).unique().to_numpy())  #
        for v in cartesian_variables  #
    ]
    # get the length of each axis
    cartesian_axes_lengths = [len(values) for _, values in cartesian_axws_values]

    axes_lengths = cartesian_axes_lengths + [-1] if zipped_variables else cartesian_axes_lengths

    # convert the DataFrame to a flat tensor for each dependent variables and view with the lengths of each axis
    tensors = {v: torch.from_numpy(df[v].to_numpy()).view(*axes_lengths)  #
               for v in dependent_variables  #
               }

    return CartesianZippedTensors(  #
        dependent_variable_tensors=tensors,  #
        cartesian_axes_values=cartesian_axws_values,  #
        zipped_axis_values=zipped_axis_values,  #
        constant_values=constant_values,  #
    )
