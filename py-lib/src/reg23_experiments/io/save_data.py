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
    file_suffix: ClassVar[str]

    @staticmethod
    @abstractmethod
    def new_value() -> 'SaveData':
        pass

    @staticmethod
    @abstractmethod
    def load_from_file(file: pathlib.Path) -> 'SaveData':
        pass

    @abstractmethod
    def apply_change(self, change: Change) -> None | Error:
        pass

    @abstractmethod
    def get_data(self) -> Any:
        pass

    @abstractmethod
    def save_to_file(self, file: pathlib.Path) -> None:
        pass


T_SaveData = TypeVar("T_SaveData", bound=SaveData)


class SaveDataManager(Generic[T_SaveData]):
    def __init__(self, *, cls: type[T_SaveData], directory: pathlib.Path, changes_per_snapshot: int = 32):
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
        return self._current_state.get_data()

    def apply_change(self, change: Change) -> None | Error:
        res: None | Error = self._current_state.apply_change(change)
        if isinstance(res, Error):
            return res
        log_file = (self._current_dir / "log.jsonl")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(change) + "\n")
        self._change_count += 1
        if self._change_count >= self._changes_per_snapshot:
            self._start_from_new_snapshot()

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
