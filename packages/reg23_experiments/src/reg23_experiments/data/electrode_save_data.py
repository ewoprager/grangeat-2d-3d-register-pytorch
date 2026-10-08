"""
Stores a list 2D electrode positions as rows of a pd.DataFrame with the following index columns:
Column name: 'xray_sop_instance_uid', 'index'
Type: str, int
and the following columns:
Column name: 'x', 'y'
Type: float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'add': Append a new point; additional keys required:
            - 'xray_sop_instance_uid': The str SOPInstanceUID of the associated X-ray image
            - 'x': The x position
            - 'y': The y position
        - 'move': Move an existing point; additional keys required:
            - 'xray_sop_instance_uid': The str SOPInstanceUID of the associated X-ray image
            - 'index': The index of the electrode to move
            - 'x': The new x position
            - 'y': The new y position
        - 'remove': Remove the last point
            - 'xray_sop_instance_uid': The str SOPInstanceUID of the associated X-ray
"""

import pathlib

import pandas as pd
import pydantic
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["ElectrodeSaveManager"]


class _AddElectrode(pydantic.BaseModel):
    xray_sop_instance_uid: str
    x: float
    y: float


class _MoveElectrode(pydantic.BaseModel):
    xray_sop_instance_uid: str
    index: int
    x: float
    y: float


class _RemoveElectrode(pydantic.BaseModel):
    xray_sop_instance_uid: str


def _incorporate_electrode_change(data: pd.DataFrame, change: pydantic.BaseModel) -> pd.DataFrame | Error:
    if isinstance(change, _AddElectrode):
        # count how many electrodes already exist
        previous_count = (data.index.get_level_values("xray_sop_instance_uid") == change.xray_sop_instance_uid).sum()
        index = pd.MultiIndex.from_tuples([(  #
            change.xray_sop_instance_uid,  #
            previous_count  #
        )], names=["xray_sop_instance_uid", "index"])
        data = pd.concat([data, pd.DataFrame([{"x": change.x, "y": change.y}], index=index)])
        return data
    elif isinstance(change, _MoveElectrode):
        # check if the electrode exists
        idx = (change.xray_sop_instance_uid, change.index)
        if idx not in data.index:
            return Error(f"Tried to move non-existent electrode with index '{idx}'.")
        # make the changes
        data.loc[idx, "x"] = change.x
        data.loc[idx, "y"] = change.y
        return data
    elif isinstance(change, _RemoveElectrode):
        # count how many electrodes already exist
        previous_count = (data.index.get_level_values("xray_sop_instance_uid") == change.xray_sop_instance_uid).sum()
        # the electrode at the top index should exist
        idx = (change.xray_sop_instance_uid, previous_count - 1)
        if idx not in data.index:
            return Error(f"Tried to remove last electrode, but it doesn't exist at expected index '{idx}'.")
        data = data.drop(idx)
        return data
    else:
        return Error(f"Unrecognized change type '{type(change).__name__}'")


def _compute_changes(  #
        *,  #
        uid: str,  #
        old_data: torch.Tensor,  #
        new_data: torch.Tensor,  #
        tol: float = 1e-8  #
) -> list[pydantic.BaseModel]:
    uid = str(uid)
    ret: list[pydantic.BaseModel] = []
    if old_data.size()[0] > new_data.size()[0]:
        # have lost some points
        for i in range(old_data.size()[0] - new_data.size()[0]):
            ret.append(_RemoveElectrode(xray_sop_instance_uid=uid))
        old_data = old_data[:new_data.size()[0]]
    elif new_data.size()[0] > old_data.size()[0]:
        # have gained some points
        for i in range(old_data.size()[0], new_data.size()[0]):
            ret.append(_AddElectrode(  #
                xray_sop_instance_uid=uid,  #
                x=new_data[i, 0].item(),  #
                y=new_data[i, 1].item(),  #
            ))
        new_data = new_data[:old_data.size()[0]]
    if new_data.size()[0]:
        diff_mask = (new_data - old_data).abs().max(dim=1).values > tol
        idx = torch.nonzero(diff_mask, as_tuple=True)[0]
        for i in idx.tolist():
            ret.append(_MoveElectrode(  #
                xray_sop_instance_uid=uid,  #
                index=i,  #
                x=new_data[i, 0].item(),  #
                y=new_data[i, 1].item(),  #
            ))
    return ret


class ElectrodeSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_spec={  #
                "add": _AddElectrode,  #
                "move": _MoveElectrode,  #
                "remove": _RemoveElectrode,  #
            },  #
            incorporate_change=_incorporate_electrode_change,  #
            default_value=pd.DataFrame(  #
                index=pd.MultiIndex.from_arrays([[], []], names=["xray_sop_instance_uid", "index"]),  #
                columns=["x", "y"],  #
            ),  #
        )
        self._state = SaveState(self._config)

    def get(self, uid: str) -> torch.Tensor | None:
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return None
        if not (df.index.get_level_values("xray_sop_instance_uid") == uid).any():
            return None
        rows_for_this_xray = df.xs(uid, level="xray_sop_instance_uid")
        if not len(rows_for_this_xray):
            return None
        return torch.tensor(rows_for_this_xray.sort_index().values)

    def set(self, uid: str, tensor: torch.Tensor) -> None | Error:
        old: torch.Tensor | None = self.get(uid)
        changes: list[pydantic.BaseModel] = _compute_changes(  #
            uid=uid,  #
            old_data=torch.empty((0, 2)) if old is None else old,  #
            new_data=tensor,  #
        )
        for change in changes:
            err = self._state.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
