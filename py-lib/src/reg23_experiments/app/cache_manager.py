import logging
from typing import Any
from platformdirs import user_cache_dir
import yaml

import pathlib

__all__ = ["CacheManager"]

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, app_name: str = "reg23 Experiments"):
        self._cache = pathlib.Path(user_cache_dir(app_name))
        self._cache.mkdir(parents=True, exist_ok=True)

    @property
    def last_ct_path(self) -> pathlib.Path | None:
        if not self._last_ct_path_path.is_file():
            return None
        ret = pathlib.Path(self._last_ct_path_path.read_text())
        if ret.exists():
            return ret
        else:
            logger.warning(f"Contents of cache file '{str(self._last_ct_path_path)}' invalid; deleting cache file.")
            self._last_ct_path_path.unlink()
            return None

    @last_ct_path.setter
    def last_ct_path(self, value: str | pathlib.Path) -> None:
        self._last_ct_path_path.write_text(str(value))

    @property
    def last_xray_paths(self) -> dict[str, pathlib.Path] | None:
        if not self._last_xray_paths_path.is_file():
            return None
        lines = self._last_xray_paths_path.read_text().split("\n")
        items = [line.split(":") for line in lines]
        for item in items:
            if len(item) != 2:
                logger.warning(
                    f"Contents of cache file '{str(self._last_xray_paths_path)}' invalid; deleting cache file.")
                self._last_xray_paths_path.unlink()
                return None

        ret = {  #
            item[0]: pathlib.Path(item[1])  #
            for item in items  #
        }
        for _, path in ret.items():
            if not path.is_file():
                logger.warning(
                    f"Contents of cache file '{str(self._last_xray_paths_path)}' invalid; deleting cache file.")
                self._last_xray_paths_path.unlink()
                return None
        return ret

    @last_xray_paths.setter
    def last_xray_paths(self, value: dict[str, str | pathlib.Path]) -> None:
        lines = [f"{k}:{str(v)}" for k, v in value.items()]
        self._last_xray_paths_path.write_text("\n".join(lines))

    @property
    def last_params(self) -> dict[str, Any] | None:
        if not self._last_params_path.is_file():
            return None
        try:
            with open(self._last_params_path, 'r') as file:
                ret = yaml.safe_load(file)
        except Exception:
            logger.warning(f"Contents of cache file '{str(self._last_params_path)}' invalid; deleting cache file.")
            self._last_params_path.unlink()
            return None
        return ret

    @last_params.setter
    def last_params(self, value: dict[str, Any]) -> None:
        with open(self._last_params_path, 'w') as file:
            yaml.safe_dump(value, file)

    @property
    def _last_ct_path_path(self) -> pathlib.Path:
        return self._cache / "last_ct_path.txt"

    @property
    def _last_xray_paths_path(self) -> pathlib.Path:
        return self._cache / "last_xray_paths.txt"

    @property
    def _last_params_path(self) -> pathlib.Path:
        return self._cache / "last_params.yaml"
