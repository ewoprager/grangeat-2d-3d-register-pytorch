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


def compute_changes(uid: str, old_data: tuple[list[str], torch.Tensor], new_data: tuple[list[str], torch.Tensor],
                    tol: float = 1e-8) -> list[Change]:
    assert len(old_data[0]) == old_data[1].size()[0]
    assert len(new_data[0]) == new_data[1].size()[0]
    assert len(old_data[1].size()) == 2
    assert old_data[1].size()[1] == 2
    assert len(new_data[1].size()) == 2
    assert new_data[1].size()[1] == 2
    uid = str(uid)
    ret: list[Change] = []
    old_set = set(old_data[0])
    new_set = set(new_data[0])

    # Points that have been removed
    for old_name in old_set - new_set:
        ret.append({  #
            "action": "remove",  #
            "xray_sop_instance_uid": uid,  #
            "name": old_name,  #
        })

    # New points
    for new_name in new_set - old_set:
        index = new_data[0].index(new_name)
        ret.append({  #
            "action": "set",  #
            "xray_sop_instance_uid": uid,  #
            "name": new_name,  #
            "x": new_data[1][index, 0].item(),  #
            "y": new_data[1][index, 1].item(),  #
        })

    # Existing points that have moved
    for name in old_set & new_set:
        old_index = old_data[0].index(name)
        new_index = new_data[0].index(name)
        if (new_data[1][new_index] - old_data[1][old_index]).abs().max() > tol:
            ret.append({  #
                "action": "set",  #
                "xray_sop_instance_uid": uid,  #
                "name": name,  #
                "x": new_data[1][new_index, 0].item(),  #
                "y": new_data[1][new_index, 1].item(),  #
            })
    return ret


class XRayFiducialSaveManager:
    def __init__(self, directory: pathlib.Path):
        directory.mkdir(exist_ok=True, parents=True)
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

    def set(self, *, uid: str, names: list[str], points: torch.Tensor) -> None | Error:
        if len(points.size()) != 2 or points.size()[1] != 2:
            return Error(f"Value should be tensor of size (N,2); got '{points.size()}'.")
        if len(names) != points.size()[0]:
            return Error(f"Names and points should have the same length; got {len(names)} and {points.size()[0]}.")
        old: tuple[list[str], torch.Tensor] | None = self.get(uid)
        changes = compute_changes(uid, ([], torch.empty((0, 2))) if old is None else old, (names, points))
        for change in changes:
            err = self._save_data_manager.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
