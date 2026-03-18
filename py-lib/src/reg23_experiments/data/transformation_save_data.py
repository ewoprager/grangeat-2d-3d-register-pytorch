import pathlib

import pandas as pd
import torch

from reg23_experiments.io.save_data import SaveData, SaveDataManager, Change, JsonSerializable
from reg23_experiments.data.structs import Error, Transformation

__all__ = ["TransformationSaveData", "TransformationSaveManager"]


class TransformationSaveData(SaveData):
    """
    ToDo: The 'name' should be the index column.
    Stores a list of 6 d.o.f. transformations as rows of a pd.DataFrame, with the following columns:
    Column name: 'name', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5'
    Type: str, float, float, float, float, float, float

    Changes are expressed as dicts with the following keys:
        'action': The string determining the action type. Possible values:
            - 'set': Add or change a named transformation to/in the list; additional keys required:
                - 'name': The string name for the transformation
                - 'x0' ... 'x5': The float param values
            - 'remove': Remove a named transformation from the list; additional keys required:
                - 'name': The name of the transformation to remove
    """

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'TransformationSaveData':
        columns = ["name"] + [f"x{i}" for i in range(6)]
        df = pd.DataFrame(columns=columns)
        return TransformationSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'TransformationSaveData':
        return TransformationSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "set":
            if "name" not in change:
                return Error("Key 'name' not found in 'set' action change.")
            indices_matching = self._contents.index[self._contents['name'] == change["name"]].tolist()
            if len(indices_matching) > 1:
                return Error(f"Found multiple transformations in the save data with name '{change["name"]}'.")
            new_values = {}
            for i in range(6):
                key = f"x{i}"
                if key not in change:
                    return Error(f"Key '{key}' not found in 'set' action change.")
                new_values[key] = change[key]
            if len(indices_matching) == 1:
                self._contents.loc[indices_matching[0], new_values.keys()] = new_values.values()
            else:
                new_values["name"] = change["name"]
                self._contents = pd.concat([self._contents, pd.DataFrame([new_values])], ignore_index=True)
            return None
        elif change["action"] == "remove":
            if "name" not in change:
                return Error("Key 'name' not found in 'remove' action change.")
            indices_matching = self._contents.index[self._contents['name'] == change["name"]].tolist()
            if len(indices_matching) > 1:
                return Error(f"Found multiple transformations in the save data with name '{change["name"]}'.")
            self._contents = self._contents.drop(indices_matching[0])
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


class TransformationSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[TransformationSaveData](cls=TransformationSaveData,
                                                                          save_directory=directory)

    def get_names(self) -> list[str]:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return []
        return df["name"].tolist()

    def get_as_dict(self, **tensor_kwargs) -> dict[str, Transformation]:
        df: pd.DataFrame = self._save_data_manager.get_data()
        return {  #
            row["name"]: Transformation.from_vector(torch.tensor([row[f"x{i}"] for i in range(6)], **tensor_kwargs))  #
            for _, row in df.iterrows()  #
        }

    def get_transformation(self, name: str, **tensor_kwargs) -> Transformation | Error:
        df: pd.DataFrame = self._save_data_manager.get_data()
        indices_matching = df.index[df['name'] == name].tolist()
        if len(indices_matching) > 1:
            return Error(f"Found multiple transformations in the save data with name '{name}'.")
        columns = [f"x{i}" for i in range(6)]
        values = df.loc[indices_matching[0], columns].tolist()
        return Transformation.from_vector(torch.tensor(values, **tensor_kwargs))

    def set(self, name: str, transformation: Transformation) -> None | Error:
        change: dict[str, JsonSerializable] = {  #
            "action": "set",  #
            "name": name,  #
        }
        t: torch.Tensor = transformation.vectorised()
        for i in range(6):
            change[f"x{i}"] = float(t[i].item())
        return self._save_data_manager.apply_change(change)

    def remove(self, name: str) -> None | Error:
        change: dict[str, JsonSerializable] = {  #
            "action": "remove",  #
            "name": name,  #
        }
        return self._save_data_manager.apply_change(change)
