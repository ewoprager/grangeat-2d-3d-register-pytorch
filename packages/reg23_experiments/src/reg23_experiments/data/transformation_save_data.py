"""
Stores a list of 6 d.o.f. transformations as rows of a pd.DataFrame, with the following columns:
Index column name: 'xray_sop_instance_uid', 'name'
Index column type: str, str
Column name: 'x0', 'x1', 'x2', 'x3', 'x4', 'x5'
Type: float, float, float, float, float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'set': Add or change a named transformation to/in the list; additional keys required:
            - 'xray_sop_instance_uid': The str SOPInstanceUID of the X-ray associated with the transformation
            - 'name': The string name for the transformation
            - 'x0' ... 'x5': The float param values
        - 'remove': Remove a named transformation from the list; additional keys required:
            - 'xray_sop_instance_uid': The str SOPInstanceUID of the X-ray associated with the transformation to
            remove
            - 'name': The name of the transformation to remove
"""

import logging
import pathlib

import pandas as pd
import pydantic
import torch

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["TransformationSaveManager"]

logger = logging.getLogger(__name__)


class _SetTransformation(pydantic.BaseModel):
    xray_sop_instance_uid: str
    name: str
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float


class _RemoveTransformation(pydantic.BaseModel):
    xray_sop_instance_uid: str
    name: str


def _incorporate_change(data: pd.DataFrame, change: pydantic.BaseModel) -> pd.DataFrame | Error:
    if isinstance(change, _SetTransformation):
        # update / insert into the dataframe
        idx = (change.xray_sop_instance_uid, change.name)
        for col in (f"x{i}" for i in range(6)):
            data.loc[idx, col] = getattr(change, col)
        return data
    elif isinstance(change, _RemoveTransformation):
        # check if the idx exists in the dataframe
        idx = (change.xray_sop_instance_uid, change.name)
        if idx in data.index:
            data = data.drop(idx)
        else:
            logger.warning(f"Tried to remove non-existent transformation '{idx}' from save data.")
        return data
    else:
        return Error(f"Unrecognized change type '{type(change).__name__}'")


class TransformationSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_spec={  #
                "set": _SetTransformation,  #
                "remove": _RemoveTransformation,  #
            },  #
            incorporate_change=_incorporate_change,  #
            default_value=pd.DataFrame(index=pd.MultiIndex.from_arrays(  #
                [[], []],  #
                names=["xray_sop_instance_uid", "name"]  #
            ), columns=[f"x{i}" for i in range(6)]),  #
        )
        self._state = SaveState(self._config)

    def get_all(self) -> pd.DataFrame:
        return self._state.get()

    def get_names(self, uid: str) -> list[str]:
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return []
        if uid in df.index.get_level_values("xray_sop_instance_uid"):
            return df.xs(uid, level="xray_sop_instance_uid").index.tolist()
        else:
            return []

    def get_as_dict(self, uid: str, *, device: torch.device = torch.device("cpu")) -> dict[str, Transformation]:
        df: pd.DataFrame = self._state.get()
        df_for_xray = df.xs(uid, level="xray_sop_instance_uid")
        return {  #
            str(name): Transformation.from_vector(
                torch.tensor([row[f"x{i}"] for i in range(6)], dtype=torch.float64, device=device))  #
            for name, row in df_for_xray.iterrows()  #
        }

    def get_transformation(self, *, uid: str, name: str,
                           device: torch.device = torch.device("cpu")) -> Transformation | Error:
        df: pd.DataFrame = self._state.get()
        idx = (uid, name)
        if idx not in df.index:
            return Error(f"No transformation saved at idx '{idx}'.")
        columns = [f"x{i}" for i in range(6)]
        values = df.loc[idx, columns].tolist()
        return Transformation.from_vector(torch.tensor(values, dtype=torch.float64, device=device))

    def set(self, *, uid: str, name: str, transformation: Transformation) -> None | Error:
        t: torch.Tensor = transformation.vectorised()
        change = _SetTransformation(  #
            xray_sop_instance_uid=uid,  #
            name=name,  #
            **{f"x{i}": float(t[i].item()) for i in range(6)},  #
        )
        return self._state.apply_change(change)

    def remove(self, *, uid: str, name: str) -> None | Error:
        change = _RemoveTransformation(  #
            xray_sop_instance_uid=uid,  #
            name=name,  #
        )
        return self._state.apply_change(change)
