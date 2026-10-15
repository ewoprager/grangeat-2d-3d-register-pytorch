"""
Stores a list 3D electrode positions as rows of a pd.DataFrame with the following index columns:
Column name: 'uid', 'index'
Type: str, int
and the following columns:
Column name: 'x', 'y', 'z'
Type: float, float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'add': Append a new point; additional keys required:
            - 'uid': The str UID of the associated CT image
            - 'x': The x position
            - 'y': The y position
            - 'z': The z position
        - 'move': Move an existing point; additional keys required:
            - 'uid': The str UID of the associated CT image
            - 'index': The index of the electrode to move
            - 'x': The new x position
            - 'y': The new y position
            - 'z': The new z position
        - 'remove': Remove the last point
            - 'uid': The str UID of the associated CT
"""

import pathlib
from typing import Literal

import pandas as pd
import pydantic
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["CTElectrodeSaveManager"]


class _AddElectrode(pydantic.BaseModel):
    action: Literal["add"]
    uid: str
    x: float
    y: float
    z: float


class _MoveElectrode(pydantic.BaseModel):
    action: Literal["move"]
    uid: str
    index: int
    x: float
    y: float
    z: float


class _RemoveElectrode(pydantic.BaseModel):
    action: Literal["remove"]
    uid: str


class Change(pydantic.BaseModel):
    value: _AddElectrode | _MoveElectrode | _RemoveElectrode = pydantic.Field(discriminator="action")


def _incorporate_change(data: pd.DataFrame, change: Change) -> pd.DataFrame | Error:
    c = change.value
    if isinstance(c, _AddElectrode):
        # count how many electrodes already exist
        previous_count = (data.index.get_level_values("uid") == c.uid).sum()
        index = pd.MultiIndex.from_tuples(
            [(  #
                c.uid,  #
                previous_count,  #
            )], names=["uid", "index"], )
        data = pd.concat([data, pd.DataFrame([{"x": c.x, "y": c.y, "z": c.z}], index=index)])
        return data
    elif isinstance(c, _MoveElectrode):
        # check if the electrode exists
        idx = (c.uid, c.index)
        if idx not in data.index:
            return Error(f"Tried to move non-existent electrode with index '{idx}'.")
        # make the changes
        data.loc[idx, "x"] = c.x
        data.loc[idx, "y"] = c.y
        data.loc[idx, "z"] = c.z
        return data
    elif isinstance(c, _RemoveElectrode):
        # count how many electrodes already exist
        previous_count = (data.index.get_level_values("uid") == c.uid).sum()
        # the electrode at the top index should exist
        idx = (c.uid, previous_count - 1)
        if idx not in data.index:
            return Error(f"Tried to remove last electrode, but it doesn't exist at expected index '{idx}'.")
        data = data.drop(idx)
        return data
    else:
        return Error(f"Unrecognized change type '{type(c).__name__}'")


def _compute_changes(  #
    *,  #
    uid: str,  #
    old_data: torch.Tensor,  #
    new_data: torch.Tensor,  #
    tol: float = 1e-8,  #
) -> list[Change]:
    uid = str(uid)
    ret: list[Change] = []
    if old_data.size()[0] > new_data.size()[0]:
        # have lost some points
        for i in range(old_data.size()[0] - new_data.size()[0]):
            ret.append(Change(value=_RemoveElectrode(action="remove", uid=uid)))
        old_data = old_data[:new_data.size()[0]]
    elif new_data.size()[0] > old_data.size()[0]:
        # have gained some points
        for i in range(old_data.size()[0], new_data.size()[0]):
            ret.append(
                Change(
                    value=_AddElectrode(  #
                        action="add",  #
                        uid=uid,  #
                        x=new_data[i, 0].item(),  #
                        y=new_data[i, 1].item(),  #
                        z=new_data[i, 2].item(),  #
                    ), ), )
        new_data = new_data[:old_data.size()[0]]
    if new_data.size()[0]:
        diff_mask = (new_data - old_data).abs().max(dim=1).values > tol
        idx = torch.nonzero(diff_mask, as_tuple=True)[0]
        for i in idx.tolist():
            ret.append(
                Change(
                    value=_MoveElectrode(  #
                        action="move",  #
                        uid=uid,  #
                        index=i,  #
                        x=new_data[i, 0].item(),  #
                        y=new_data[i, 1].item(),  #
                        z=new_data[i, 2].item(),  #
                    ), ), )
    return ret


class CTElectrodeSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_schema=Change,  #
            incorporate_change=_incorporate_change,  #
            default_value=pd.DataFrame(  #
                index=pd.MultiIndex.from_arrays([[], []], names=["uid", "index"]),  #
                columns=["x", "y", "z"],  #
            ),  #
        )
        self._state = SaveState(self._config)

    def get(self, uid: str) -> torch.Tensor | None:
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return None
        if not (df.index.get_level_values("uid") == uid).any():
            return None
        rows_for_this_xray = df.xs(uid, level="uid")
        if not len(rows_for_this_xray):
            return None
        return torch.tensor(rows_for_this_xray.sort_index().values)

    def set(self, uid: str, tensor: torch.Tensor) -> None | Error:
        old: torch.Tensor | None = self.get(uid)
        changes: list[Change] = _compute_changes(  #
            uid=uid,  #
            old_data=torch.empty((0, 3)) if old is None else old,  #
            new_data=tensor,  #
        )
        for change in changes:
            err = self._state.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
