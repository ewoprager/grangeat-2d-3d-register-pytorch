"""
Stores a list of 2D fiducial positions as rows of a pd.DataFrame with the following index columns:
Column name: 'xray_sop_instance_uid', 'name'
Type: str, str
and the following columns:
Column name: 'x', 'y'
Type: float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'set': Set (create or move) the position of a named marker; additional keys required:
            - 'xray_sop_instance_uid': The str UID of the associated X-ray image
            - 'name': The str name of the fiducial marker
            - 'x': The x position
            - 'y': The y position
        - 'remove': Remove a marker
            - 'xray_sop_instance_uid': The str UID of the associated X-ray image
            - 'name': The str name of the fiducial marker to remove
"""

import pathlib

import pandas as pd
import pydantic
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["XRayFiducialSaveManager"]


class _SetXRayFiducial(pydantic.BaseModel):
    xray_sop_instance_uid: str
    name: str
    x: float
    y: float


class _RemoveXRayFiducial(pydantic.BaseModel):
    xray_sop_instance_uid: str
    name: str


def _incorporate_change(data: pd.DataFrame, change: pydantic.BaseModel) -> pd.DataFrame | Error:
    if isinstance(change, _SetXRayFiducial):
        # Update / insert into the dataframe
        idx = (change.xray_sop_instance_uid, change.name)
        data.loc[idx, ["x", "y"]] = [change.x, change.y]
        return data
    elif isinstance(change, _RemoveXRayFiducial):
        # check if the idx exists in the dataframe
        idx = (change.xray_sop_instance_uid, change.name)
        if idx in data.index:
            data = data.drop(idx)
        else:
            return Error(f"Tried to remove non-existent fiducial '{idx}' from save data.")
        return data
    else:
        return Error(f"Unrecognized change type '{type(change).__name__}'")


def _compute_changes(  #
        *,  #
        uid: str,  #
        old_data: tuple[list[str], torch.Tensor],  #
        new_data: tuple[list[str], torch.Tensor],  #
        tol: float = 1e-8,  #
) -> list[pydantic.BaseModel]:
    assert len(old_data[0]) == old_data[1].size()[0]
    assert len(new_data[0]) == new_data[1].size()[0]
    assert len(old_data[1].size()) == 2
    assert old_data[1].size()[1] == 2
    assert len(new_data[1].size()) == 2
    assert new_data[1].size()[1] == 2
    uid = str(uid)
    ret: list[pydantic.BaseModel] = []
    old_set = set(old_data[0])
    new_set = set(new_data[0])

    # Points that have been removed
    for old_name in old_set - new_set:
        ret.append(_RemoveXRayFiducial(xray_sop_instance_uid=uid, name=old_name))

    # New points
    for new_name in new_set - old_set:
        index = new_data[0].index(new_name)
        ret.append(_SetXRayFiducial(  #
            xray_sop_instance_uid=uid,  #
            name=new_name,  #
            x=new_data[1][index, 0].item(),  #
            y=new_data[1][index, 1].item(),  #
        ))

    # Existing points that have moved
    for name in old_set & new_set:
        old_index = old_data[0].index(name)
        new_index = new_data[0].index(name)
        if (new_data[1][new_index] - old_data[1][old_index]).abs().max() > tol:
            ret.append(_SetXRayFiducial(  #
                xray_sop_instance_uid=uid,  #
                name=name,  #
                x=new_data[1][new_index, 0].item(),  #
                y=new_data[1][new_index, 1].item(),  #
            ))
    return ret


class XRayFiducialSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_spec={  #
                "set": _SetXRayFiducial,  #
                "remove": _RemoveXRayFiducial,  #
            },  #
            incorporate_change=_incorporate_change,  #
            default_value=pd.DataFrame(index=pd.MultiIndex.from_arrays(  #
                [[], []],  #
                names=["xray_sop_instance_uid", "name"]  #
            ), columns=["x", "y"]),  #
        )
        self._state = SaveState(self._config)

    def get(self, uid: str) -> tuple[list[str], torch.Tensor] | None:
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return None
        if not (df.index.get_level_values("xray_sop_instance_uid") == uid).any():
            return None
        rows_for_this_xray = df.xs(uid, level="xray_sop_instance_uid")
        if not len(rows_for_this_xray):
            return None
        return list(rows_for_this_xray.index.get_level_values("name")), torch.tensor(
            rows_for_this_xray.values.astype(float))

    def set(self, *, uid: str, names: list[str], points: torch.Tensor) -> None | Error:
        if len(points.size()) != 2 or points.size()[1] != 2:
            return Error(f"Value should be tensor of size (N,2); got '{points.size()}'.")
        if len(names) != points.size()[0]:
            return Error(f"Names and points should have the same length; got {len(names)} and {points.size()[0]}.")
        old: tuple[list[str], torch.Tensor] | None = self.get(uid)
        changes = _compute_changes(  #
            uid=uid,  #
            old_data=([], torch.empty((0, 2))) if old is None else old,  #
            new_data=(names, points),  #
        )
        for change in changes:
            err = self._state.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
