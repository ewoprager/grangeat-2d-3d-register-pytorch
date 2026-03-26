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
        logger.info(f"Loading/saving cached data in '{str(self._cache)}'")

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
    def _last_params_path(self) -> pathlib.Path:
        return self._cache / "last_params.yaml"
