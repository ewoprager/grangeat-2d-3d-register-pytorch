import pathlib
from typing import Any

import pandas as pd
import torch

from reg23_experiments.io.save_data import SaveData, SaveDataManager, Change, JsonSerializable
from reg23_experiments.data.structs import Error, Transformation


class ElectrodeSaveData(SaveData):
    """
    Stores a list 2D electrode positions as rows of a pd.DataFrame with the following index columns:
    Column name: 'xray_path', 'index'
    Type: str, int
    and the following columns:
    Column name: 'x', 'y'
    Type: float, float

    Changes are expressed as dicts with the following keys:
        'action': The string determining the action type. Possible values:
            - 'add': Append a new point; additional keys required:
                - 'xray_path': The string path of the associated X-ray image
                - 'x': The x position
                - 'y': The y position
            - 'move': Move an existing point; additional keys required:
                - 'xray_path': The string path of the associated X-ray image
                - 'index': The index of the electrode to move
                - 'x': The new x position
                - 'y': The new y position
            - 'remove': Remove the last point
                - 'xray_path': The string path of the associated X-ray
    """

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'ElectrodeSaveData':
        index = pd.MultiIndex.from_arrays([[], []], names=["xray_path", "index"])
        columns = ["x", "y"]
        df = pd.DataFrame(index=index, columns=columns)
        return ElectrodeSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'ElectrodeSaveData':
        return ElectrodeSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "add":
            if "xray_path" not in change:
                return Error("Key 'xray_path' not found in 'add' action change.")
            xray_path = change["xray_path"]
            if not isinstance(xray_path, str):
                return Error("'xray_path' value in 'add' action change should be a `str`.")
            if "x" not in change:
                return Error("Key 'x' not found in 'add' action change.")
            x = change["x"]
            if not isinstance(x, float):
                return Error("'x' value in 'add' action change should be a `float`.")
            if "y" not in change:
                return Error("Key y' not found in 'add' action change.")
            y = change["y"]
            if not isinstance(y, float):
                return Error("'y' value in 'add' action change should be a `float`.")
            previous_count = 0 if self._contents.index.get_level_values("xray_path").empty else \
                self._contents.xs(xray_path, level="xray_path").shape[0]
            index = pd.MultiIndex.from_tuples([(xray_path, previous_count)], names=["xray_path", "indices"])
            self._contents = pd.concat([self._contents, pd.DataFrame([{"x": x, "y": y}], index=index)])
            return None
        elif change["action"] == "move":
            if "xray_path" not in change:
                return Error("Key 'xray_path' not found in 'move' action change.")
            xray_path = change["xray_path"]
            if not isinstance(xray_path, str):
                return Error("'xray_path' value in 'move' action change should be a `str`.")
            if "x" not in change:
                return Error("Key 'x' not found in 'move' action change.")
            if "index" not in change:
                return Error("Key 'index' not found in 'move' action change.")
            index = change["index"]
            if not isinstance(index, int):
                return Error("'index' value in 'move' action change should be an `int`.")
            x = change["x"]
            if not isinstance(x, float):
                return Error("'x' value in 'move' action change should be a `float`.")
            if "y" not in change:
                return Error("Key y' not found in 'move' action change.")
            y = change["y"]
            if not isinstance(y, float):
                return Error("'y' value in 'move' action change should be a `float`.")
            self._contents.loc[(xray_path, index), "x"] = x
            self._contents.loc[(xray_path, index), "y"] = y
            return None
        elif change["action"] == "remove":
            if "xray_path" not in change:
                return Error("Key 'xray_path' not found in 'remove' action change.")
            xray_path = change["xray_path"]
            if not isinstance(xray_path, str):
                return Error("'xray_path' value in 'remove' action change should be a `str`.")
            previous_count = self._contents.xs(xray_path, level="xray_path").shape[0]
            self._contents = self._contents.drop(index=(xray_path, previous_count - 1))
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


def compute_changes(xray_path: str, old_data: torch.Tensor, new_data: torch.Tensor, tol: float = 1e-8) -> list[Change]:
    ret: list[Change] = []
    if old_data.size()[0] > new_data.size()[0]:
        # have lost some points
        for i in range(old_data.size()[0] - new_data.size()[0]):
            ret.append({  #
                "action": "remove",  #
                "xray_path": xray_path,  #
            })
        old_data = old_data[:new_data.size()[0]]
    elif new_data.size()[0] > old_data.size()[0]:
        # have gained some points
        for i in range(old_data.size()[0], new_data.size()[0]):
            ret.append({  #
                "action": "add",  #
                "xray_path": xray_path,  #
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
                "xray_path": xray_path,  #
                "index": i,  #
                "x": new_data[i, 0].item(),  #
                "y": new_data[i, 1].item(),  #
            })
    return ret


class ElectrodeSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[ElectrodeSaveData](cls=ElectrodeSaveData, directory=directory)

    def get(self, xray_path: str) -> torch.Tensor | None:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return None
        rows_for_this_xray = df.loc[xray_path]
        if not len(rows_for_this_xray):
            return None
        return torch.tensor(rows_for_this_xray.sort_index().values)

    def set(self, xray_path: str, tensor: torch.Tensor) -> None | Error:
        old: torch.Tensor | None = self.get(xray_path)
        changes = compute_changes(xray_path, torch.empty((0, 2)) if old is None else old, tensor)
        for change in changes:
            err = self._save_data_manager.apply_change(change)
            if isinstance(err, Error):
                return err
        return None
