import logging
import pathlib
from typing import Any

import yaml
import zarr
from platformdirs import user_cache_dir

from reg23_app.state import AppState
from reg23_experiments.data.structs import Error
from reg23_experiments.io.helpers import contents_based_hash_slices
from reg23_experiments.io.large import large_volume_to_zarr

__all__ = ["CacheManager", "ZarrManager"]

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, app_name: str = "reg23_experiments"):
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


class ZarrManager:
    def __init__(self, *, state: AppState, micro_ct_zarr_path: pathlib.Path):
        self._state = state
        self._micro_ct_zarr_group = zarr.open_group(micro_ct_zarr_path)

    def get_micro_ct(self) -> zarr.Array | None | Error:
        if self._state.micro_ct_slice_path is None:
            return None
        file = pathlib.Path(self._state.micro_ct_slice_path)
        if not file.is_file():
            return Error(f"Chosen microCT file '{str(file)}' doesn't exist.")
        logger.info(f"Retrieving hash of microCT volume contents...")
        uid: str | Error = contents_based_hash_slices(file)
        if isinstance(uid, Error):
            logger.error(f"\tFailed: {uid.description}")
            return uid
        logger.info(f"\tDone; UID = '{uid}'")

        if uid in self._micro_ct_zarr_group.array_keys():
            logger.info(f"Get microCT by UID: Array already present in zarr cache")
            return self._micro_ct_zarr_group[uid]
        else:
            logger.info(f"Get microCT by UID: Array not present in zarr cache; streaming in...")
            res: zarr.Array | Error = large_volume_to_zarr(  #
                group=self._micro_ct_zarr_group,  #
                volume_id=uid,  #
                a_slice_file_path=file,  #
            )
            if isinstance(res, Error):
                logger.error(f"\tFailed: {res.description}")
                return res
            logger.info(f"\tDone; zarr cache entry created for volume '{uid}'")
            return res
