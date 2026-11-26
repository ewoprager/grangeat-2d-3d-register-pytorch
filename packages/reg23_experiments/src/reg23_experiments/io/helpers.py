import hashlib
import logging
import pathlib

import SimpleITK as sitk
import yaml
from platformdirs import user_cache_dir
from tqdm import tqdm

from reg23_experiments.data.structs import Error

__all__ = ["slice_files_from_one", "deterministic_hash_string", "contents_based_hash", "contents_based_hash_slices"]

logger = logging.getLogger(__name__)


def slice_files_from_one(a_slice_file_path: pathlib.Path) -> list[pathlib.Path]:
    suffix = a_slice_file_path.suffix
    directory = a_slice_file_path.parent
    return sorted(  #
        e  #
            for e in directory.iterdir()  #
            if e.is_file() and e.suffix == suffix  #
    )


def deterministic_hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def contents_based_hash(image: sitk.Image) -> str:
    return hashlib.sha256(memoryview(sitk.GetArrayViewFromImage(image)).cast("B")).hexdigest()


def contents_based_hash_slices(a_slice_file_path: pathlib.Path) -> str | Error:
    # Get all the slice files
    if not a_slice_file_path.is_file():
        return Error(f"File '{a_slice_file_path}' does not exist.")
    directory = a_slice_file_path.parent

    # Load the cache
    cache_dir = pathlib.Path(user_cache_dir(appname="reg23_experiments"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = cache_dir / "contents_based_hash_slices.yml"
    if not cache_file_path.is_file():
        cache = {}
        cache_file_path.write_text(yaml.safe_dump(cache))
    else:
        cache = yaml.safe_load(cache_file_path.read_text())

    # Return cached value if present
    if str(directory) in cache:
        return cache[str(directory)]

    # Calculate new value
    slice_file_paths = slice_files_from_one(a_slice_file_path)
    # Loop through each slice, accumulating the hash
    hasher = hashlib.sha256()
    for slice_file_path in tqdm(slice_file_paths, desc="Accumulating hash"):
        try:
            image = sitk.ReadImage(slice_file_path)
        except Exception as e:
            return Error(f"Failed to read slice file '{str(slice_file_path)}': {e}")
        hasher.update(memoryview(sitk.GetArrayViewFromImage(image)).cast("B"))
    ret = hasher.hexdigest()

    # Add new value to the cache
    cache[str(directory)] = ret
    cache_file_path.write_text(yaml.safe_dump(cache))

    return ret
