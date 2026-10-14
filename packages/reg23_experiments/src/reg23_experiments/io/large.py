import configparser
import logging
import pathlib

import SimpleITK as sitk
import zarr

from reg23_experiments.data.structs import Error
from reg23_experiments.io.helpers import slice_files_from_one

__all__ = ["large_volume_to_zarr"]

logger = logging.getLogger(__name__)


def large_volume_to_zarr(  #
        *,  #
        group: zarr.Group,  #
        volume_id: str,  #
        a_slice_file_path: pathlib.Path,  #
) -> zarr.Array | Error:
    # Check id not already present
    if volume_id in group.keys():
        return Error(f"Volume with id '{volume_id}' already exists in given group.")
    # Get a sorted list of all the slice file paths
    if not a_slice_file_path.is_file():
        return Error(f"Provided slice file '{str(a_slice_file_path)}' doesn't exist.")
    slice_file_paths = slice_files_from_one(a_slice_file_path)
    # Read the size and type from the first one and create an array accordingly
    first = sitk.GetArrayFromImage(sitk.ReadImage(slice_file_paths[0]))
    if len(first.shape) != 2:
        return Error(f"Volume slices are expected to be 2D images; found image of shape {first.shape}")
    volume_shape = (len(slice_file_paths), *first.shape)
    array = group.create_array(  #
        name=volume_id,  #
        shape=volume_shape,  #
        dtype=first.dtype,  #
    )
    # Read the slices into the array one by one
    for i, slice_path in enumerate(slice_file_paths):
        slice_array = sitk.GetArrayFromImage(sitk.ReadImage(slice_path))
        if slice_array.shape != first.shape:
            return Error(
                f"Not all slices shared the same size; slice {i} was of shape {slice_array.shape}, but first slice "
                f"was of "
                f"shape {first.shape}")
        if slice_array.dtype != first.dtype:
            return Error(
                f"Not all slices shared the same dtype; slice {i} was of dtype {slice_array.dtype}, but first slice "
                f"was of "
                f"dtype {first.dtype}")
        array[i, :, :] = slice_array

    # If there's an xtekVolume file, read the spacing from it
    xtek_volume_path = [e for e in a_slice_file_path.parent.iterdir() if e.is_file() and e.suffix == ".xtekVolume"]
    if len(xtek_volume_path) == 1:
        c = configparser.ConfigParser()
        c.read(xtek_volume_path[0])
        spacing = [  #
            float(c["XTekCT"][f"VoxelSize{axis}"])  #
            for axis in ("X", "Y", "Z")  #
        ]
    else:
        logger.warning(f"Found {len(xtek_volume_path)} xtekVolume files in directory of slice, so can't load spacing")
        spacing = None

    array.attrs["spacing"] = spacing

    return array
