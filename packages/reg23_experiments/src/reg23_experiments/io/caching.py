"""
Tools for caching data that takes a while to compute, so that it can be re-used.

Cached data is stored in the following structure in the filesystem:

```text
cache/
├-- <type name>/  # Directory for a particular type of cached data, with a human-readable name
│   ├-- spec.md  # Human-readable specification for contained parameter files `params.yaml`, and details of what data files are saved of what types.
│   └-- <instance name>/  # Directory for a particular instance of cached data, with a human-readable name
│       ├-- params.yaml  # Parametrisation that uniquely defines this cache instance, consistent with `spec.md`; if matched, the instance is loaded rather than re-calculated
│       ├-- <data name>.pkl  # Files containing the pickles data of the instance, with human-readable names, consistent with `spec.md`.
│       └-- ...
└-- ... # other types of cached data
```
"""
import logging
import pathlib
import pickle
import pprint
from typing import Any

import yaml

from reg23_experiments.io.serialize import JsonSerializable

__all__ = ["save_to_cache", "load_from_cache"]

logger = logging.getLogger(__name__)


def _find_cache_instance(  #
        *,  #
        type_dir: pathlib.Path,  #
        params: JsonSerializable,  #
) -> pathlib.Path | None:
    for element in type_dir.iterdir():
        if not element.is_dir():
            continue
        params_file = element / "params.yaml"
        if not params_file.is_file():
            logger.warning(f"No 'params.yaml' found in '{str(type_dir)}' cache directory.")
            continue
        these_params = yaml.safe_load(str(params_file))
        if these_params == params:
            return element
    return None


def save_to_cache(  #
        *,  #
        cache_directory: pathlib.Path,  #
        type_name: str,  #
        instance_name: str,  #
        params: JsonSerializable,  #
        data: dict[str, Any],  #
) -> None:
    type_dir = cache_directory / type_name
    type_dir.mkdir(parents=True, exist_ok=True)

    exists: pathlib.Path | None = _find_cache_instance(type_dir=type_dir, params=params)

    if exists is None:
        instance_dir = type_dir / instance_name
        instance_dir.mkdir(parents=True, exist_ok=True)
    else:
        if exists.name != instance_name:
            note = f" under different instance name '{exists.name}'"
        else:
            note = ""
        logger.info(
            f"Value already exists in '{str(type_dir)}' cache for params:\n{pprint.pformat(params)}\n{note}; overwriting.")
        instance_dir = exists

    for k, v in data.items():
        data_file = instance_dir / f"{k}.pkl"
        data_file.write_bytes(pickle.dumps(v))


def load_from_cache(  #
        *,  #
        cache_directory: pathlib.Path,  #
        type_name: str,  #
        params: JsonSerializable,  #
) -> dict[str, Any] | None:
    type_dir = cache_directory / type_name
    type_dir.mkdir(parents=True, exist_ok=True)

    exists: pathlib.Path | None = _find_cache_instance(type_dir=type_dir, params=params)

    if exists is None:
        return None

    ret = {}
    for element in exists.iterdir():
        if not element.is_dir() or element.suffix != ".pkl":
            continue
        data = pickle.loads(element.read_bytes())
        ret[element.stem] = data

    return ret
