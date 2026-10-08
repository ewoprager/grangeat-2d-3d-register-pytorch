"""
Stores a list of X-ray registration configs as rows of a pd.DataFrame with the following index columns:
Column name: 'xray_sop_instance_uid'
Type: str
and the following columns:
Column name: 'horizontal_flip', 'crop_left', 'crop_right', 'crop_top', 'crop_bottom'
Type: bool, float, float, float, float

Changes are expressed as dicts with the following keys:
    'action': The string determining the action type. Possible values:
        - 'set': Set (add or change) the config of an X-ray; additional keys required:
            - 'xray_sop_instance_uid': The str UID of the associated X-ray image
            - 'horizontal_flip': Whether the X-ray is flipped horizontally relative to its corresponding CT scan.
            - 'crop_left': The left crop value, in the range (0, 1), with 0 being no cropping.
            - 'crop_right': The right crop value, in the range (0, 1), with 1 being no cropping.
            - 'crop_top': The top crop value, in the range (0, 1), with 0 being no cropping.
            - 'crop_bottom': The bottom crop value, in the range (0, 1), with 1 being no cropping.
        - 'remove': Remove a saved X-ray config
            - 'xray_sop_instance_uid': The str UID of the associated X-ray image
"""

import pathlib
from typing import Literal

import pandas as pd
import pydantic

from reg23_experiments.data.structs import Cropping, Error
from reg23_experiments.io.save_data import SaveConfig, SaveState

__all__ = ["XRayRegSaveManager"]


class _SetXRayRegData(pydantic.BaseModel):
    action: Literal["set"]
    xray_sop_instance_uid: str
    horizontal_flip: bool
    crop_left: float
    crop_right: float
    crop_top: float
    crop_bottom: float


class _RemoveXRayRegData(pydantic.BaseModel):
    action: Literal["remove"]
    xray_sop_instance_uid: str


class Change(pydantic.BaseModel):
    value: _SetXRayRegData | _RemoveXRayRegData = pydantic.Field(discriminator="action")


def _incorporate_change(data: pd.DataFrame, change: Change) -> pd.DataFrame | Error:
    c = change.value
    if isinstance(c, _SetXRayRegData):
        # Update / insert the row into the dataframe
        for col in ['horizontal_flip', 'crop_left', 'crop_right', 'crop_top', 'crop_bottom']:
            data.loc[c.xray_sop_instance_uid, col] = getattr(c, col)
        return data
    elif isinstance(c, _RemoveXRayRegData):
        # check if the idx exists in the dataframe
        if c.xray_sop_instance_uid in data.index:
            data = data.drop(c.xray_sop_instance_uid)
        else:
            return Error(f"Tried to remove config for non-existent X-ray '{c.xray_sop_instance_uid}' from save data.")
        return data
    else:
        return Error(f"Unrecognized change type '{type(c).__name__}'")


class XRayRegSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._config = SaveConfig(  #
            save_path=directory,  #
            change_schema=Change,  #
            incorporate_change=_incorporate_change,  #
            default_value=pd.DataFrame(  #
                index=pd.Index([], name="xray_sop_instance_uid"),  #
                columns=["horizontal_flip", "crop_left", "crop_right", "crop_top", "crop_bottom"]  #
            ),  #
        )
        self._state = SaveState(self._config)

    def get_all(self) -> pd.DataFrame:
        return self._state.get()

    def get_flipped(self, uid: str) -> bool | None:
        if (row := self._get_row(uid)) is None:
            return None
        return row["horizontal_flip"]

    def get_cropping(self, uid: str) -> Cropping | None:
        if (row := self._get_row(uid)) is None:
            return None
        return Cropping(right=row["crop_right"], top=row["crop_top"], left=row["crop_left"], bottom=row["crop_bottom"])

    def set(self, *, uid: str, flipped: bool, cropping: Cropping) -> None | Error:
        change = Change(value=_SetXRayRegData(  #
            action="set",  #
            xray_sop_instance_uid=uid,  #
            horizontal_flip=flipped,  #
            crop_left=cropping.left,  #
            crop_right=cropping.right,  #
            crop_top=cropping.top,  #
            crop_bottom=cropping.bottom,  #
        ))
        if isinstance(err := self._state.apply_change(change), Error):
            return err
        return None

    def _get_row(self, uid: str) -> pd.Series | None:
        df: pd.DataFrame = self._state.get()
        if df.empty:
            return None
        if uid not in df.index:
            return None
        return df.xs(uid)
