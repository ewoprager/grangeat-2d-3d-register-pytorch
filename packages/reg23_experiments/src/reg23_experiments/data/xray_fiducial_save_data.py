import pathlib

import pandas as pd
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import Change, SaveData, SaveDataManager

__all__ = ["XRayFiducialSaveData", "XRayFiducialSaveManager"]


class XRayFiducialSaveData(SaveData):
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

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'XRayFiducialSaveData':
        index = pd.MultiIndex.from_arrays([[], []], names=["xray_sop_instance_uid", "name"])
        columns = ["x", "y"]
        df = pd.DataFrame(index=index, columns=columns)
        return XRayFiducialSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'XRayFiducialSaveData':
        return XRayFiducialSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "set":
            # get the uid
            if "xray_sop_instance_uid" not in change:
                return Error("Key 'xray_sop_instance_uid' not found in 'add' action change.")
            uid = change["xray_sop_instance_uid"]
            if not isinstance(uid, str):
                return Error("'xray_sop_instance_uid' value in 'add' action change should be a `str`.")
            # get the name
            if "name" not in change:
                return Error("Key 'name' not found in 'add' action change.")
            name = change["name"]
            if not isinstance(name, str):
                return Error("'name' value in 'add' action change should be a `str`.")
            # get the x value
            if "x" not in change:
                return Error("Key 'x' not found in 'add' action change.")
            x = change["x"]
            if not isinstance(x, float):
                return Error("'x' value in 'add' action change should be a `float`.")
            # get the y value
            if "y" not in change:
                return Error("Key 'y' not found in 'add' action change.")
            y = change["y"]
            if not isinstance(y, float):
                return Error("'y' value in 'add' action change should be a `float`.")
            # Update / insert into the dataframe
            idx = (uid, name)
            self._contents.loc[idx, ["x", "y"]] = [x, y]
            return None
        elif change["action"] == "remove":
            # get the uid
            if "xray_sop_instance_uid" not in change:
                return Error("Key 'xray_sop_instance_uid' not found in 'remove' action change.")
            uid = change["xray_sop_instance_uid"]
            if not isinstance(uid, str):
                return Error("'xray_sop_instance_uid' value in 'remove' action change should be a `str`.")
            # get the name
            if "name" not in change:
                return Error("Key 'name' not found in 'remove' action change.")
            name = change["name"]
            if not isinstance(name, str):
                return Error("'name' value in 'remove' action change should be a `str`.")
            # check if the idx exists in the dataframe
            idx = (uid, name)
            if idx in self._contents.index:
                self._contents = self._contents.drop(idx)
            else:
                return Error(f"Tried to remove non-existent fiducial '{idx}' from save data.")
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


class XRayFiducialSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[XRayFiducialSaveData](cls=XRayFiducialSaveData,
                                                                        save_directory=directory)

    def get(self, uid: str) -> tuple[list[str], torch.Tensor] | None:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return None
        if not (df.index.get_level_values("xray_sop_instance_uid") == uid).any():
            return None
        rows_for_this_xray = df.xs(uid, level="xray_sop_instance_uid")
        if not len(rows_for_this_xray):
            return None
        return list(rows_for_this_xray.index.get_level_values("name")), torch.tensor(
            rows_for_this_xray.values.astype(float))

    def set(self, *, uid: str, name: str, value: torch.Tensor) -> None | Error:
        if value.size() != torch.Size([2]):
            return Error(f"Value should be tensor of size (2,); got '{value}'.")
        change = {  #
            "action": "set",  #
            "xray_sop_instance_uid": uid,  #
            "name": name,  #
            "x": value[0].item(),  #
            "y": value[1].item(),  #
        }
        err = self._save_data_manager.apply_change(change)
        if isinstance(err, Error):
            return err
        return None

    def remove(self, *, uid: str, name: str) -> None | Error:
        change = {  #
            "action": "remove",  #
            "xray_sop_instance_uid": uid,  #
            "name": name  #
        }
        err = self._save_data_manager.apply_change(change)
        if isinstance(err, Error):
            return err
        return None
