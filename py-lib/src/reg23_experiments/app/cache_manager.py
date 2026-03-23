from platformdirs import user_cache_dir

import pathlib

__all__ = ["CacheManager"]


class CacheManager:
    def __init__(self, app_name: str = "reg23 Experiments"):
        self._cache = pathlib.Path(user_cache_dir(app_name))
        self._cache.mkdir(parents=True, exist_ok=True)

    @property
    def last_ct_path(self) -> pathlib.Path | None:
        if not self._last_ct_path_path.is_file():
            return None
        ret = pathlib.Path(self._last_ct_path_path.read_text())
        if ret.is_dir():
            return ret
        else:
            self._last_ct_path_path.unlink()
            return None

    @last_ct_path.setter
    def last_ct_path(self, value: str | pathlib.Path) -> None:
        self._last_ct_path_path.write_text(str(value))

    @property
    def _last_ct_path_path(self) -> pathlib.Path:
        return self._cache / "last_ct_path.txt"
