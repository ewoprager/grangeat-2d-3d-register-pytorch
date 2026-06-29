import pathlib

import pandas as pd

from reg23_experiments.data.structs import Cropping, Error
from reg23_experiments.io.save_data import Change, SaveData, SaveDataManager

__all__ = ["XRayRegSaveData", "XRayRegSaveManager"]


class XRayRegSaveData(SaveData):
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

    file_suffix = ".parquet"

    def __init__(self, contents: pd.DataFrame | None = None):
        self._contents = pd.DataFrame() if contents is None else contents

    def get_data(self) -> pd.DataFrame:
        return self._contents

    @staticmethod
    def new_value() -> 'XRayRegSaveData':
        index = pd.Index([], name="xray_sop_instance_uid")
        columns = ["horizontal_flip", "crop_left", "crop_right", "crop_top", "crop_bottom"]
        df = pd.DataFrame(index=index, columns=columns)
        return XRayRegSaveData(df)

    @staticmethod
    def load_from_file(file: pathlib.Path) -> 'XRayRegSaveData':
        return XRayRegSaveData(pd.read_parquet(file))

    def apply_change(self, change: Change) -> None | Error:
        if "action" not in change:
            return Error("Key 'action' not found in change.")
        if change["action"] == "set":
            def _get_value_by_name[T](_name: str, _type_check: type[T]) -> T | Error:
                if _name not in change:
                    return Error(f"Key '{_name}' not found in 'set' action change.")
                _ret = change[_name]
                if not isinstance(_ret, bool):
                    return Error(f"'{_name}' value in 'set' action change should be a `{_type_check.__name__}`.")
                return _ret

            # Get the X-ray UID
            if isinstance(uid := _get_value_by_name("xray_sop_instance_uid", str), Error):
                return uid

            # The names and types of all expected columns
            column_names_types = [  #
                ("horizontal_flip", bool),  #
                ("crop_left", float),  #
                ("crop_right", float),  #
                ("crop_top", float),  #
                ("crop_bottom", float),  #
            ]
            # Get the column values
            column_values = []
            for name, type_check in column_names_types:
                if isinstance(res := _get_value_by_name(name, type_check), Error):
                    return res
                column_values.append(res)

            # Update / insert the row into the dataframe
            self._contents.loc[uid, [name for name, _ in column_names_types]] = column_values
            return None
        elif change["action"] == "remove":
            # get the uid
            if "xray_sop_instance_uid" not in change:
                return Error("Key 'xray_sop_instance_uid' not found in 'remove' action change.")
            uid = change["xray_sop_instance_uid"]
            if not isinstance(uid, str):
                return Error("'xray_sop_instance_uid' value in 'remove' action change should be a `str`.")
            # check if the idx exists in the dataframe
            if uid in self._contents.index:
                self._contents = self._contents.drop(uid)
            else:
                return Error(f"Tried to remove config for non-existent X-ray '{uid}' from save data.")
            return None
        else:
            return Error(f"Unrecognised action '{change["action"]}'.")

    def save_to_file(self, file: pathlib.Path) -> None:
        self._contents.to_parquet(file)


class XRayRegSaveManager:
    def __init__(self, directory: pathlib.Path):
        self._save_data_manager = SaveDataManager[XRayRegSaveData](cls=XRayRegSaveData, save_directory=directory)

    def get_flipped(self, uid: str) -> bool | None:
        if (row := self._get_row(uid)) is None:
            return None
        return row["horizontal_flip"]

    def get_cropping(self, uid: str) -> Cropping | None:
        if (row := self._get_row(uid)) is None:
            return None
        return Cropping(right=row["crop_right"], top=row["crop_top"], left=row["crop_left"], bottom=row["crop_bottom"])

    def set(self, *, uid: str, flipped: bool, cropping: Cropping) -> None | Error:
        change = {  #
            "action": "set",  #
            "xray_sop_instance_uid": uid,  #
            "horizontal_flip": flipped,  #
            "crop_left": cropping.left,  #
            "crop_right": cropping.right,  #
            "crop_top": cropping.top,  #
            "crop_bottom": cropping.bottom,  #
        }
        if isinstance(err := self._save_data_manager.apply_change(change), Error):
            return err
        return None

    def _get_row(self, uid: str) -> pd.Series | None:
        df: pd.DataFrame = self._save_data_manager.get_data()
        if df.empty:
            return None
        if uid not in df.index:
            return None
        return df.xs(uid)
