"""
A setup for saving data with versioning in a memory-efficient way.

Can be applied to any desired data structure, but the serialising/deserialising of changes to the data must
be implemented.

The whole data structure is saved as 'snapshots' every N (default=32) changes. Between snapshots, only the changes that
are made to the data are saved, specifically as lines in a jsonl log file.

Any change to the data must be represented by a `Change` object, which is just a Python object that can be trivially
serialised into JSON.

The user must provide a class that implements `SaveData`, which has the custom code for using a `Change` object to apply
the desired change to the data.

The user can then instantiate a SaveDataManager, passing their `SaveData` class as the Generic parameter, and `cls`
parameter. This object will automatically load existing save data if present, and manage all subsequent changes and
saving.

On loading of saved data (e.g. on program startup), the most recent snapshot is loaded, and all subsequent changes are
applied in order such that the data is restored to the same state it was in when the program was last run.
"""

import json
import pathlib
from datetime import datetime
from typing import Callable

import pandas as pd
import pydantic

from reg23_experiments.data.structs import Error
from reg23_experiments.io.serialize import serialize_recursive

__all__ = ["SaveType", "SavedObject"]


def _snapshot_file_path(*, snapshot_dir: pathlib.Path) -> pathlib.Path:
    return snapshot_dir / "snapshot.parquet"


def _log_file_path(*, snapshot_dir: pathlib.Path) -> pathlib.Path:
    return snapshot_dir / "log.jsonl"


def _is_valid_snapshot_dir(snapshot_dir: pathlib.Path) -> bool:
    return snapshot_dir.is_dir() and _snapshot_file_path(snapshot_dir=snapshot_dir).is_file()


def _find_latest_snapshot_dir(save_dir: pathlib.Path) -> pathlib.Path | Error:
    latest = ""
    for element in save_dir.iterdir():
        if _is_valid_snapshot_dir(element) and element.stem > latest:
            latest = element.stem
    if not latest:
        return Error(f"No valid snapshot directories found in save directory '{str(save_dir)}'.")
    return save_dir / latest


ChangeSpec = dict[str, type[pydantic.BaseModel]]


class ChangeBase(pydantic.BaseModel):
    action: str
    parameters: dict


class SaveType:
    def __init__(  #
            self,  #
            *,  #
            save_path: pathlib.Path,  #
            change_spec: ChangeSpec,  #
            incorporate_change: Callable[[pd.DataFrame, pydantic.BaseModel], pd.DataFrame | Error],  #
            default_value: pd.DataFrame,  #
            changes_per_snapshot: int = 32,  #
    ):
        self._save_path = save_path
        self._change_spec = change_spec
        self._incorporate_change = incorporate_change
        self._default_value = default_value
        self._changes_per_snapshot = changes_per_snapshot

    @property
    def incorporate_change(self) -> Callable[[pd.DataFrame, pydantic.BaseModel], pd.DataFrame]:
        return self._incorporate_change

    @property
    def default_value(self) -> pd.DataFrame:
        return self._default_value

    @property
    def changes_per_snapshot(self) -> int:
        return self._changes_per_snapshot

    def load_specific_save(self, *, snapshot: str, change_count: int = -1) -> tuple[pd.DataFrame, int] | Error:
        """
        Load a specific save, optionally specifying a change count to load.
        :param snapshot:
        :param change_count: The number of changes to load. Pass -1 (the default) to load all changes for the
        snapshot. If more changes are requested that exist, only existing changes will be loaded, and no error will be
        thrown. The number of changes actually loaded is returned, so this can easily be detected.
        :return: (the new snapshot of `cls` with the loaded data, the number of changes actually loaded),
        or the error if one occurred.
        """
        snapshot_dir = self._save_path / snapshot
        if not _is_valid_snapshot_dir(snapshot_dir):
            return Error(f"Error loading specific save: '{str(snapshot_dir)}' is not a valid snapshot directory.")
        if change_count < -1:
            return Error(
                f"Error loading specific save: invalid change count: '{change_count}'; use the value -1 to indicate "
                f"all changes, or any non-negative integer to indicate the number of changes to load.")
        ret = pd.read_parquet(_snapshot_file_path(snapshot_dir=snapshot_dir))
        log_file = _log_file_path(snapshot_dir=snapshot_dir)
        change_i = 0
        if log_file.is_file():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if -1 < change_count <= change_i:
                        break
                    try:
                        change: ChangeBase = ChangeBase.model_validate_json(line)
                    except pydantic.ValidationError as e:
                        return Error(
                            f"Change at line '{line}' in log file '{str(log_file)}' did not conform to schema: {e}")
                    except ValueError as e:
                        return Error(f"Error parsing line: '{line}' as JSON from log file '{str(log_file)}': {e}")
                    if change.action not in self._change_spec:
                        return Error(
                            f"Unrecognised action '{change.action}' in lin e'{line}', log file '{str(log_file)}'")
                    model = self._change_spec[change.action]
                    try:
                        change: model = model.model_validate(change.parameters)
                    except pydantic.ValidationError as e:
                        return Error(
                            f"Change at line '{line}' in log file '{str(log_file)}' did not conform to schema: {e}")
                    ret: pd.DataFrame | Error = self._incorporate_change(ret, change)
                    if isinstance(ret, Error):
                        return ret
                    change_i += 1
        return ret, change_i

    def load_latest_save(self) -> tuple[pathlib.Path, pd.DataFrame, int] | Error:
        """
        Load the latest data save in the given directory.
        :return: (the snapshot directory loaded from, the new instance of `cls` with the loaded data, the number of
        changes loaded since the last snapshot), or the error if one occurred.
        """
        snapshot_dir: pathlib.Path | Error = _find_latest_snapshot_dir(save_dir=self._save_path)
        if isinstance(snapshot_dir, Error):
            return Error(f"Error loading latest save: {snapshot_dir.description}.")
        res = self.load_specific_save(snapshot=snapshot_dir.name)
        if isinstance(res, Error):
            return Error(f"Error loading latest save: {res.description}.")
        save_data, change_count = res
        return snapshot_dir, save_data, change_count

    def create_new_snapshot(self) -> pathlib.Path:
        timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_dir: pathlib.Path = self._save_path / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _log_file_path(snapshot_dir=snapshot_dir).touch()
        return snapshot_dir


class SavedObject:
    def __init__(self, object_type: SaveType):
        """
        Constructor. Loads any existing data from the save directory `directory`.

        Each snapshot is stored in its own directory with the given `directory`, named with a timestamp:
        YYYY-MM-DD_hh-mm-ss. Subsequent changes are stored as lines of a file `log.jsonl` saved in the same directory.
        :param object_type: The type of the data stored
        """
        self._object_type = object_type
        # load from the latest save directory, if there is one
        res: tuple[pathlib.Path, pd.DataFrame, int] | Error = self._object_type.load_latest_save()
        if isinstance(res, Error):
            self._current_state: pd.DataFrame = self._object_type.default_value
            self._start_from_new_snapshot()
        else:
            self._current_snapshot_dir, self._current_state, self._change_count = res
            _log_file_path(snapshot_dir=self._current_snapshot_dir).touch()

    def get(self) -> pd.DataFrame:
        return self._current_state

    def _start_from_new_snapshot(self) -> None:
        self._current_snapshot_dir = self._object_type.create_new_snapshot()
        self._current_state.to_parquet(_snapshot_file_path(snapshot_dir=self._current_snapshot_dir))
        self._change_count: int = 0

    def _apply_change(self, change_object: SchemaType) -> None | Error:
        """
        Apply the given change to the data and save the change to log.jsonl. If at least `changes_per_snapshot` changes
        have been logged, save a new snapshot.
        :param change_object: The change to apply and save.
        :return: The error if one occurs.
        """
        res: pd.DataFrame | Error = self._object_type.incorporate_change(self._current_state, change_object)
        if isinstance(res, Error):
            return res
        self._current_state = res
        schema_value_name, trait = next(iter(change_object.traits().items()))
        change_value = getattr(change_object, schema_value_name)
        with open(_log_file_path(snapshot_dir=self._current_snapshot_dir), 'a', encoding='utf-8') as f:
            f.write(json.dumps(serialize_recursive(change_value, trait=trait)) + "\n")
        self._change_count += 1
        if self._change_count >= self._object_type.changes_per_snapshot:
            self._start_from_new_snapshot()
        return None
