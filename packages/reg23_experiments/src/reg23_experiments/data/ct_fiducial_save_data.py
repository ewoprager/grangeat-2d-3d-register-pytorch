import pathlib

import pandas as pd
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.save_data import Change, SaveData, SaveDataManager

__all__ = ["CTFiducialSaveData", "CTFiducialSaveManager"]


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
            # Update / insert into the dataframe
            idx = (uid, name)
            self._contents.loc[idx, ["x", "y", "z"]] = [x, y, z]
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
                return Error(f"Tried to remove non-existent fiducial '{idx}' from save data.")
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


def compute_changes(uid: str, old_data: tuple[list[str], torch.Tensor], new_data: tuple[list[str], torch.Tensor],
                    tol: float = 1e-8) -> list[Change]:
    assert len(old_data[0]) == old_data[1].size()[0]
    assert len(new_data[0]) == new_data[1].size()[0]
    assert len(old_data[1].size()) == 2
    assert old_data[1].size()[1] == 3
    assert len(new_data[1].size()) == 2
    assert new_data[1].size()[1] == 3
    uid = str(uid)
    ret: list[Change] = []
    old_set = set(old_data[0])
    new_set = set(new_data[0])

    # Points that have been removed
    for old_name in old_set - new_set:
        ret.append({  #
            "action": "remove",  #
            "ct_series_uid": uid,  #
            "name": old_name,  #
        })

    # New points
    for new_name in new_set - old_set:
        index = new_data[0].index(new_name)
        ret.append({  #
            "action": "set",  #
            "ct_series_uid": uid,  #
            "name": new_name,  #
            "x": new_data[1][index, 0].item(),  #
            "y": new_data[1][index, 1].item(),  #
            "z": new_data[1][index, 2].item(),  #
        })

    # Existing points that have moved
    for name in old_set & new_set:
        old_index = old_data[0].index(name)
        new_index = new_data[0].index(name)
        if (new_data[1][new_index] - old_data[1][old_index]).abs().max() > tol:
            ret.append({  #
                "action": "set",  #
                "ct_series_uid": uid,  #
                "name": name,  #
                "x": new_data[1][new_index, 0].item(),  #
                "y": new_data[1][new_index, 1].item(),  #
                "z": new_data[1][new_index, 2].item(),  #
            })
    return ret


class CTFiducialSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
        self._save_data_manager = SaveDataManager[CTFiducialSaveData](cls=CTFiducialSaveData, save_directory=directory)

    def get(self, uid: str) -> tuple[list[str], torch.Tensor] | None:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return None
        if not (df.index.get_level_values("ct_series_uid") == uid).any():
            return None
        rows_for_this_ct = df.xs(uid, level="ct_series_uid")
        if not len(rows_for_this_ct):
            return None
        return list(rows_for_this_ct.index.get_level_values("name")), torch.tensor(
            rows_for_this_ct.values.astype(float))

    def set(self, *, uid: str, names: list[str], points: torch.Tensor) -> None | Error:
        if len(points.size()) != 2 or points.size()[1] != 3:
            return Error(f"Value should be tensor of size (N,3); got '{points.size()}'.")
        if len(names) != points.size()[0]:
            return Error(f"Names and points should have the same length; got {len(names)} and {points.size()[0]}.")
        old: tuple[list[str], torch.Tensor] | None = self.get(uid)
        changes = compute_changes(uid, ([], torch.empty((0, 3))) if old is None else old, (names, points))
        for change in changes:
            err = self._save_data_manager.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
