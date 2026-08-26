import logging
from typing import Any, Literal

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


class CartesianZippedTensors:
    def __init__(  #
            self,  #
            *,  #
            dependent_variable_tensors: dict[str, torch.Tensor],  #
            cartesian_axes_values: list[tuple[str, np.ndarray]],  #
            zipped_axis_values: list[tuple[str, np.ndarray]],  #
            constant_values: dict[str, Any],  #
    ):
        # there must be at least one dependent variable
        assert len(dependent_variable_tensors) > 0

        # all tensors of dependent variables must be the same size
        tensor_size = next(iter(dependent_variable_tensors.values())).size()
        assert all(t.size() == tensor_size for t in dependent_variable_tensors.values())

        # all cartesian and zipped axes values should be 1D arrays
        assert all(len(t[1].shape) == 1 for t in cartesian_axes_values)
        assert all(len(t[1].shape) == 1 for t in zipped_axis_values)

        # all zipped axis value arrays should have the same length
        if zipped_axis_values:
            zipped_n = len(zipped_axis_values[0][1])
            assert all(len(t[1]) == zipped_n for t in zipped_axis_values)

        # the lengths of the cartesian and zipped axes values should match the dependent tensor axes sizes
        expected_size = [len(t[1]) for t in cartesian_axes_values]
        if zipped_axis_values:
            expected_size += [zipped_n]
        assert tensor_size == torch.Size(expected_size)

        self._dependent_variable_tensors = dependent_variable_tensors
        self._cartesian_axes_values = cartesian_axes_values
        self._zipped_axis_values = zipped_axis_values
        self._constant_values = constant_values

    @property
    def dependent_variable_tensors(self) -> dict[str, torch.Tensor]:
        return self._dependent_variable_tensors

    @property
    def cartesian_axes_values(self) -> list[tuple[str, np.ndarray]]:
        return self._cartesian_axes_values

    @property
    def zipped_axis_values(self) -> list[tuple[str, np.ndarray]]:
        return self._zipped_axis_values

    @property
    def constant_values(self) -> dict[str, Any]:
        return self._constant_values

    def reduce(  #
            self,  #
            variable: str,  #
            *,  #
            method: Literal["take_last"],  #
    ) -> 'CartesianZippedTensors':
        try:
            cart_index = [t[0] for t in self.cartesian_axes_values].index(variable)
        except ValueError:
            cart_index = None

        if cart_index is not None:
            index = (slice(None),) * cart_index + (-1,)
            dvt = {  #
                f"{k}_at_last_{variable}": v[index]  #
                for k, v in self.dependent_variable_tensors.items()  #
            }
            cav = [t for t in self.cartesian_axes_values if t[0] != variable]
            return CartesianZippedTensors(  #
                dependent_variable_tensors=dvt,  #
                cartesian_axes_values=cav,  #
                zipped_axis_values=self.zipped_axis_values,  #
                constant_values=self.constant_values,  #
            )

        if variable in (t[0] for t in self.zipped_axis_values):
            logger.info(
                f"Requests reduction over zipped variable '{variable}'; reducing over full zip, including variables "
                f"{", ".join(t[0] for t in self.zipped_axis_values)}")
            index = (slice(None),) * len(self.cartesian_axes_values) + (-1,)
            dvt = {  #
                f"{k}_at_last_zipped": v[index]  #
                for k, v in self.dependent_variable_tensors.items()  #
            }
            return CartesianZippedTensors(  #
                dependent_variable_tensors=dvt,  #
                cartesian_axes_values=self.cartesian_axes_values,  #
                zipped_axis_values=[],  #
                constant_values=self.constant_values,  #
            )

        raise ValueError(f"Variable '{variable}' not present in CartesianZippedTensors")


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
    assert len(zipped_variables) != 1, (
        f"There must be at least two Zipped variables, if any; found 1: {zipped_variables[0]}")

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
