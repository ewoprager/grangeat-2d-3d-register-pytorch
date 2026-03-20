import logging
import pathlib

import pandas as pd
import torch

from reg23_experiments.io.save_data import SaveData, SaveDataManager, Change, JsonSerializable
from reg23_experiments.data.structs import Error, Transformation

__all__ = ["TransformationSaveData", "TransformationSaveManager"]

logger = logging.getLogger(__name__)


class TransformationSaveData(SaveData):
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

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'TransformationSaveData':
        index = pd.MultiIndex.from_arrays([[], []], names=["xray_sop_instance_uid", "name"])
        columns = [f"x{i}" for i in range(6)]
        df = pd.DataFrame(index=index, columns=columns)
        return TransformationSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'TransformationSaveData':
        return TransformationSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "set":
            # get the uid
            if "xray_sop_instance_uid" not in change:
                return Error("Key 'xray_sop_instance_uid' not found in 'set' action change.")
            uid = change["xray_sop_instance_uid"]
            if not isinstance(uid, str):
                return Error("'xray_sop_instance_uid' value in 'set' action change should be a `str`.")
            # get the name
            if "name" not in change:
                return Error("Key 'name' not found in 'set' action change.")
            name = change["name"]
            if not isinstance(name, str):
                return Error("'name' value in 'set' action change should be a `str`.")
            # get the column values
            new_values = {}
            for i in range(6):
                key = f"x{i}"
                if key not in change:
                    return Error(f"Key '{key}' not found in 'set' action change.")
                new_values[key] = change[key]
            # check if the idx exists in the dataframe
            idx = (uid, name)
            if idx in self._contents.index:
                # set the column values for the existing row
                self._contents.loc[idx, new_values.keys()] = new_values.values()
            else:
                # construct and append a new row
                new_row = pd.DataFrame([new_values],
                                       index=pd.MultiIndex.from_tuples([idx], names=self._contents.index.names))
                self._contents = pd.concat([self._contents, new_row])
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
                logger.warning(f"Tried to remove non-existent transformation '{idx}' from save data.")
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


class TransformationSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[TransformationSaveData](cls=TransformationSaveData,
                                                                          save_directory=directory)

    def get_names(self, uid: str) -> list[str]:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return []
        if uid in df.index.get_level_values("xray_sop_instance_uid"):
            return df.xs(uid, level="xray_sop_instance_uid").index.tolist()
        else:
            return []

    def get_as_dict(self, uid: str, **tensor_kwargs) -> dict[str, Transformation]:
        df: pd.DataFrame = self._save_data_manager.get_data()
        df_for_xray = df.xs(uid, level="xray_sop_instance_uid")
        return {  #
            name: Transformation.from_vector(torch.tensor([row[f"x{i}"] for i in range(6)], **tensor_kwargs))  #
            for name, row in df_for_xray.iterrows()  #
        }

    def get_transformation(self, *, uid: str, name: str, **tensor_kwargs) -> Transformation | Error:
        df: pd.DataFrame = self._save_data_manager.get_data()
        idx = (uid, name)
        if idx not in df.index:
            return Error(f"No transformation saved at idx '{idx}'.")
        columns = [f"x{i}" for i in range(6)]
        values = df.loc[idx, columns].tolist()
        return Transformation.from_vector(torch.tensor(values, **tensor_kwargs))

    def set(self, *, uid: str, name: str, transformation: Transformation) -> None | Error:
        change: dict[str, JsonSerializable] = {  #
            "action": "set",  #
            "xray_sop_instance_uid": uid,  #
            "name": name,  #
        }
        t: torch.Tensor = transformation.vectorised()
        for i in range(6):
            change[f"x{i}"] = float(t[i].item())
        return self._save_data_manager.apply_change(change)

    def remove(self, *, uid: str, name: str) -> None | Error:
        change: dict[str, JsonSerializable] = {  #
            "action": "remove",  #
            "xray_sop_instance_uid": uid,  #
            "name": name,  #
        }
        return self._save_data_manager.apply_change(change)
