import pathlib
import logging

import pandas as pd
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import Change, SaveData, SaveDataManager

__all__ = ["CTFiducialSaveData", "CTFiducialSaveManager"]

logger = logging.getLogger(__name__)


class CTFiducialSaveData(SaveData):
    """
    Stores a list of 3D fiducial positions as rows of a pd.DataFrame with the following index columns:
    Column name: 'ct_series_uid', 'name'
    Type: str, str
    and the following columns:
    Column name: 'x', 'y', 'z'
    Type: float, float, float

    Changes are expressed as dicts with the following keys:
        'action': The string determining the action type. Possible values:
            - 'set': Set (create or move) the position of a named marker; additional keys required:
                - 'ct_series_uid': The str UID of the associated CT volume
                - 'name': The str name of the fiducial marker
                - 'x': The x position
                - 'y': The y position
                - 'z': The z position
            - 'remove': Remove a marker
                - 'ct_series_uid': The str UID of the associated CT volume
                - 'name': The str name of the fiducial marker to remove
    """

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'CTFiducialSaveData':
        index = pd.MultiIndex.from_arrays([[], []], names=["ct_series_uid", "name"])
        columns = ["x", "y", "z"]
        df = pd.DataFrame(index=index, columns=columns)
        return CTFiducialSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'CTFiducialSaveData':
        return CTFiducialSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "set":
            # get the uid
            if "ct_series_uid" not in change:
                return Error("Key 'ct_series_uid' not found in 'add' action change.")
            uid = change["ct_series_uid"]
            if not isinstance(uid, str):
                return Error("'ct_series_uid' value in 'add' action change should be a `str`.")
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
                return Error("'y' value in 'add' action change should be a `float`.")  # get the y value
            if "z" not in change:
                return Error("Key 'z' not found in 'add' action change.")
            z = change["z"]
            if not isinstance(z, float):
                return Error("'z' value in 'add' action change should be a `float`.")
            # check if the name is already present
            idx = (uid, name)
            if idx in self._contents.index:
                # set the column values for the existing row
                self._contents.loc[idx, "x"] = x
                self._contents.loc[idx, "y"] = y
                self._contents.loc[idx, "z"] = z
            else:
                # construct and append a new row
                new_row = pd.DataFrame([x, y, z],
                                       index=pd.MultiIndex.from_tuples([idx], names=self._contents.index.names))
                self._contents = pd.concat([self._contents, new_row])
            return None
        elif change["action"] == "remove":
            # get the uid
            if "ct_series_uid" not in change:
                return Error("Key 'ct_series_uid' not found in 'remove' action change.")
            uid = change["ct_series_uid"]
            if not isinstance(uid, str):
                return Error("'ct_series_uid' value in 'remove' action change should be a `str`.")
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
                logger.warning(f"Tried to remove non-existent fiducial '{idx}' from save data.")
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


def compute_changes(uid: str, old_data: torch.Tensor, new_data: torch.Tensor, tol: float = 1e-8) -> list[Change]:
    uid = str(uid)
    ret: list[Change] = []
    if old_data.size()[0] > new_data.size()[0]:
        # have lost some points
        for i in range(old_data.size()[0] - new_data.size()[0]):
            ret.append({  #
                "action": "remove",  #
                "xray_sop_instance_uid": uid,  #
            })
        old_data = old_data[:new_data.size()[0]]
    elif new_data.size()[0] > old_data.size()[0]:
        # have gained some points
        for i in range(old_data.size()[0], new_data.size()[0]):
            ret.append({  #
                "action": "add",  #
                "xray_sop_instance_uid": uid,  #
                "x": new_data[i, 0].item(),  #
                "y": new_data[i, 1].item(),  #
            })
        new_data = new_data[:old_data.size()[0]]
    if new_data.size()[0]:
        diff_mask = (new_data - old_data).abs().max(dim=1).values > tol
        idx = torch.nonzero(diff_mask, as_tuple=True)[0]
        for i in idx.tolist():
            ret.append({  #
                "action": "move",  #
                "xray_sop_instance_uid": uid,  #
                "index": i,  #
                "x": new_data[i, 0].item(),  #
                "y": new_data[i, 1].item(),  #
            })
    return ret


class CTFiducialSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[CTFiducialSaveData](cls=CTFiducialSaveData, save_directory=directory)

    def get(self, uid: str) -> torch.Tensor | None:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return None
        if not (df.index.get_level_values("ct_series_uid") == uid).any():
            return None
        rows_for_this_xray = df.xs(uid, level="ct_series_uid")
        if not len(rows_for_this_xray):
            return None
        return torch.tensor(rows_for_this_xray.sort_index().values)

    def set(self, uid: str, tensor: torch.Tensor) -> None | Error:
        old: torch.Tensor | None = self.get(uid)
        changes = compute_changes(uid, torch.empty((0, 2)) if old is None else old, tensor)
        for change in changes:
            err = self._save_data_manager.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
