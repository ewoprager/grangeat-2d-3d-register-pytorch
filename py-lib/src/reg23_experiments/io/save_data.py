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
from typing import Generic, TypeVar, ClassVar, Any
from abc import ABC, abstractmethod
from datetime import datetime

import pathlib

from reg23_experiments.data.structs import Error

__all__ = ["Change", "JsonSerializable", "SaveData", "SaveDataManager"]

type JsonSerializable = None | bool | int | float | str | list[JsonSerializable] | dict[str, JsonSerializable]
type Change = dict[str, JsonSerializable]


class SaveData(ABC):
    """
    An interface to be implemented by a class that wishes to store and manage the serialisation of some data.
    """
    file_suffix: ClassVar[str]

    @staticmethod
    @abstractmethod
    def new_value() -> 'SaveData':
        """
        This is used when there is no existing snapshot file to load from.
        :return: A default-initialised instance of this class, with default-initialised data (e.g. an empty data
        structure).
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_file(file: pathlib.Path) -> 'SaveData':
        """
        This is used to load snapshot files.
        :param file: Path to the file to load. Can be expected to have the suffix `file_suffix`
        :return: An instance of this class, initialised with the data loaded from the given file.
        """
        pass

    @abstractmethod
    def apply_change(self, change: Change) -> None | Error:
        """
        Apply the given change to the data. This is used when changes are applied.
        :param change: A trivially JSON-serialisable object that described a change to make to the data.
        :return: The error, if one has occurred.
        """
        pass

    @abstractmethod
    def get_data(self) -> Any:
        """
        :return: The data currently stored by this object.
        """
        pass

    @abstractmethod
    def save_to_file(self, file: pathlib.Path) -> None:
        """
        Save the data stored in this object to the given file. This is used for saving snapshots.
        :param file: Path to the file to save to. Can be expected to the have the suffix `file_suffix`.
        """
        pass


T_SaveData = TypeVar("T_SaveData", bound=SaveData)


class SaveDataManager(Generic[T_SaveData]):
    """
    Object parametrised by a user-implemented class type derived from `SaveData`. Manages saving and loading of data
    change-by-change between snapshots according to the methods implemented by the user's class.
    """

    def __init__(self, *, cls: type[T_SaveData], directory: pathlib.Path, changes_per_snapshot: int = 32):
        """
        Constructor. Loads any existing data from the save directory `directory`.

        Each snapshot is stored in its own directory with the given `directory`, named with a timestamp:
        YYYY-MM-DD_hh-mm-ss. Subsequent changes are stored as lines of a file `log.jsonl` saved in the same directory.

        :param cls: User-implemented class derived from `SaveData` used for interpretation of `Change` objects.
        :param directory: The save directory from which to load and to which to save data.
        :param changes_per_snapshot: The number of changes to save between snapshots. Does not affect previous saves.
        """
        # store the class type; must be done first
        self._cls = cls
        self._outer_directory = directory
        self._changes_per_snapshot = changes_per_snapshot
        # find the latest save directory, if there is one
        assert directory.is_dir()
        latest = ""
        for element in directory.iterdir():
            if self._is_valid_save_directory(element) and element.stem > latest:
                latest = element.stem
        latest_dir = directory / latest
        # load from the latest save directory if it exists, otherwise initialise a new one
        if self._is_valid_save_directory(latest_dir):
            self._current_dir: pathlib.Path = latest_dir
            self._current_state, self._change_count = self._state_from_save_data(latest_dir)
            pathlib.Path(self._current_dir / "log.jsonl").touch()
        else:
            self._current_state: T_SaveData = self._cls.new_value()
            self._start_from_new_snapshot()

    def get_data(self) -> Any:
        """
        :return: The data currently stored.
        """
        return self._current_state.get_data()

    def apply_change(self, change: Change) -> None | Error:
        """
        Apply the given change to the data and save the change to log.jsonl. If at least `changes_per_snapshot` changes
        have been logged, save a new snapshot.
        :param change: The change to apply and save.
        :return: The error if one occurs.
        """
        res: None | Error = self._current_state.apply_change(change)
        if isinstance(res, Error):
            return res
        log_file = (self._current_dir / "log.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(change) + "\n")
        self._change_count += 1
        if self._change_count >= self._changes_per_snapshot:
            self._start_from_new_snapshot()
        return None

    def _start_from_new_snapshot(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._current_dir: pathlib.Path = self._outer_directory / timestamp
        self._current_dir.mkdir(parents=True, exist_ok=True)
        self._current_state.save_to_file(self._snapshot_file(self._current_dir))
        self._change_count: int = 0
        pathlib.Path(self._current_dir / "log.jsonl").touch()

    def _snapshot_file(self, directory: pathlib.Path) -> pathlib.Path:
        return directory / ("snapshot" + self._cls.file_suffix)

    def _is_valid_save_directory(self, directory: pathlib.Path) -> bool:
        return directory.is_dir() and self._snapshot_file(directory).is_file()

    def _state_from_save_data(self, directory: pathlib.Path) -> tuple[T_SaveData, int]:
        assert self._is_valid_save_directory(directory)
        ret = self._cls.load_from_file(self._snapshot_file(directory))
        log_file = directory / "log.jsonl"
        change_count = 0
        if log_file.is_file():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        change = json.loads(line)
                        assert isinstance(change, dict)
                    except Exception as e:
                        raise RuntimeError(f"Error parsing line: '{line}' as JSON from log file '{str(log_file)}': {e}")
                    ret.apply_change(change)
                    change_count += 1
        return ret, change_count
