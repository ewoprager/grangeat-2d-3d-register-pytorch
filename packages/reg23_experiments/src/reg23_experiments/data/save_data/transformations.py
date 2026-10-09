"""
Stores a list of 6 d.o.f. transformations as rows of a pd.DataFrame, with the following columns:
Index column name: 'src_uid', 'dest_uid', 'name'
Index column type: str, str
Column name: 'x0', 'x1', 'x2', 'x3', 'x4', 'x5'
Type: float, float, float, float, float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'set': Add or change a named transformation to/in the list; additional keys required:
            - 'src_uid': The str UID of the X-ray or CT within whose space the transformation lies
            - 'dest_uid': The str UID of the X-ray or CT to which the transformation aligns the source image
            - 'name': The string name for the transformation
            - 'x0' ... 'x5': The float param values
        - 'remove': Remove a named transformation from the list; additional keys required:
            - 'src_uid': The str UID of the X-ray or CT within whose space the transformation lies
            - 'dest_uid': The str UID of the X-ray or CT to which the transformation aligns the source image
            - 'name': The name of the transformation to remove
"""

import logging
import pathlib
from typing import Literal

import pandas as pd
import pydantic
import torch

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["TransformationSaveManager"]

logger = logging.getLogger(__name__)


class _SetTransformation(pydantic.BaseModel):
    action: Literal["set"]
    src_uid: str
    dest_uid: str
    name: str
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float


class _RemoveTransformation(pydantic.BaseModel):
    action: Literal["remove"]
    src_uid: str
    dest_uid: str
    name: str


class Change(pydantic.BaseModel):
    value: _SetTransformation | _RemoveTransformation = pydantic.Field(discriminator="action")


def _incorporate_change(data: pd.DataFrame, change: Change) -> pd.DataFrame | Error:
    c = change.value
    if isinstance(c, _SetTransformation):
        # update / insert into the dataframe
        idx = (c.src_uid, c.dest_uid, c.name)
        for col in (f"x{i}" for i in range(6)):
            data.loc[idx, col] = getattr(c, col)
        return data
    elif isinstance(c, _RemoveTransformation):
        # check if the idx exists in the dataframe
        idx = (c.src_uid, c.dest_uid, c.name)
        if idx in data.index:
            data = data.drop(idx)
        else:
            logger.warning(f"Tried to remove non-existent transformation '{idx}' from save data.")
        return data
    else:
        return Error(f"Unrecognized change type '{type(c).__name__}'")


class TransformationSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_schema=Change,  #
            incorporate_change=_incorporate_change,  #
            default_value=pd.DataFrame(index=pd.MultiIndex.from_arrays(  #
                [[], [], []],  #
                names=["src_uid", "dest_uid", "name"]  #
            ), columns=[f"x{i}" for i in range(6)]),  #
        )
        self._state = SaveState(self._config)

    def get_all(self) -> pd.DataFrame:
        return self._state.get()

    def get_list_of_names(self, *, source_uid: str, destination_uid: str) -> list[str]:
        """
        Get the dataframe idx values for all the transformations with the given source image

        :param source_uid: str UID of the source image
        :param destination_uid: str UID of the destination image
        :return: A list of transformation names
        """
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return []
        return df[(  #
                (df.index.get_level_values("src_uid") == source_uid)  #
                & (df.index.get_level_values("dest_uid") == destination_uid)  #
        )].index.get_level_values("name").tolist()

    def get_name_dict(  #
            self,  #
            source_uid: str,  #
            destination_uid: str,  #
            *,  #
            device: torch.device = torch.device("cpu"),  #
    ) -> dict[str, Transformation]:
        """
        Get a dictionary mapping names to transformations for the given source and destination UIDs

        :param source_uid: str UID of the source image
        :param destination_uid: str UID of the destination image
        :param device: [Default: CPU] The torch device on which to put the returned transformations
        :return: A dict mapping names to transformations
        """
        df: pd.DataFrame = self._state.get()
        try:
            filtered = df.xs((source_uid, destination_uid), level=("src_uid", "dest_uid"))
        except KeyError:
            return {}
        return {  #
            str(name): Transformation.from_vector(  #
                torch.tensor([row[f"x{i}"] for i in range(6)], dtype=torch.float64, device=device),  #
            )  #
            for name, row in filtered.iterrows()  #
        }

    def get_transformation(  #
            self,  #
            *,  #
            source_uid: str,  #
            destination_uid: str,  #
            name: str,  #
            device: torch.device = torch.device("cpu"),  #
    ) -> Transformation | Error:
        df: pd.DataFrame = self._state.get()
        idx = (source_uid, destination_uid, name)
        if idx not in df.index:
            return Error(f"No transformation saved at idx '{idx}'.")
        columns = [f"x{i}" for i in range(6)]
        values = df.loc[idx, columns].tolist()
        return Transformation.from_vector(torch.tensor(values, dtype=torch.float64, device=device))

    def set(  #
            self,  #
            *,  #
            source_uid: str,  #
            destination_uid: str,  #
            name: str,  #
            transformation: Transformation,  #
    ) -> None | Error:
        t: torch.Tensor = transformation.vectorised()
        change = Change(value=_SetTransformation(  #
            action="set",  #
            src_uid=source_uid,  #
            dest_uid=destination_uid,  #
            name=name,  #
            **{f"x{i}": float(t[i].item()) for i in range(6)},  #
        ))
        return self._state.apply_change(change)

    def remove(  #
            self,  #
            *,  #
            source_uid: str,  #
            destination_uid: str,  #
            name: str,  #
    ) -> None | Error:
        change = Change(value=_RemoveTransformation(  #
            action="remove",  #
            src_uid=source_uid,  #
            dest_uid=destination_uid,  #
            name=name,  #
        ))
        return self._state.apply_change(change)
