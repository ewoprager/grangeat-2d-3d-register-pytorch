import copy
import pathlib

import SimpleITK as sitk
import numpy as np
import pydicom
import traitlets
from pydicom.uid import generate_uid

from reg23_experiments.data.structs import Error

__all__ = ["DCMSeriesInfo", "find_ct_series", "load_ct_series", "load_image_file", "save_ct_series"]


class DCMSeriesInfo(traitlets.HasTraits):
    uid: str = traitlets.Unicode(allow_none=False)
    slice_count: int = traitlets.Int(allow_none=False)


def find_ct_series(path: str | pathlib.Path) -> dict[str, DCMSeriesInfo] | Error:
    """
    Get information about all the DICOM CT series present in the given directory.
    :param path: A directory in which to look for the DICOM series.
    :return: A dict of {uid: DCMSeriesInfo}, or an Error if encountered.
    """
    path = pathlib.Path(path)
    if not path.is_dir():
        return Error(f"Path '{str(path)}' is not a directory.")
    reader = sitk.ImageSeriesReader()
    uids: tuple[str, ...] = reader.GetGDCMSeriesIDs(str(path))
    ret: dict[str, DCMSeriesInfo] = {}
    for uid in uids:
        ret[uid] = DCMSeriesInfo(uid=uid, slice_count=len(reader.GetGDCMSeriesFileNames(str(path), uid)))
    return ret


def load_ct_series(path: str | pathlib.Path, series_uid: str) -> sitk.Image | Error:
    path = pathlib.Path(path)
    if not path.is_dir():
        return Error(f"Path '{str(path)}' is not a directory.")
    reader = sitk.ImageSeriesReader()
    series_uids: tuple[str, ...] = reader.GetGDCMSeriesIDs(str(path))
    if not series_uids:
        return Error(f"No DICOM series found at path '{str(path)}'.")
    if series_uid not in series_uids:
        return Error(f"Series '{series_uid}' not found in directory '{str(path)}'.")
    files: list[str] = reader.GetGDCMSeriesFileNames(str(path), series_uid)
    reader.SetFileNames(files)
    try:
        return reader.Execute()
    except Exception as e:
        return Error(f"Failed to load CT series '{series_uid}' from path '{str(path)}': {str(e)}")


def load_image_file(path: str | pathlib.Path) -> sitk.Image | Error:
    path = pathlib.Path(path)
    if not path.is_file():
        return Error(f"Path '{str(path)}' is not a file.")
    try:
        return sitk.ReadImage(str(path))
    except Exception as e:
        return Error(f"Failed to load image from path '{str(path)}': {str(e)}")


def save_ct_series(  #
        *,  #
        image: sitk.Image,  #
        path: str | pathlib.Path,  #
        template_slice: pydicom.FileDataset  #
) -> None | Error:
    path = pathlib.Path(path)
    if len(image.GetSize()) != 3:
        return Error("Expected a 3D image to save as a CT series.")
    path.mkdir(exist_ok=True, parents=True)
    new_series_uid: pydicom.uid.UID = generate_uid()
    volume: np.ndarray = sitk.GetArrayFromImage(image)
    for i, slice_hu in enumerate(volume):
        new_ds = copy.deepcopy(template_slice)

        stored = ((slice_hu - float(template_slice.RescaleIntercept)) / float(template_slice.RescaleSlope)).astype(
            np.int16)
        new_ds.PixelData = stored.tobytes()
        new_ds.Rows, new_ds.Columns = stored.shape
        new_ds.SOPInstanceUID = generate_uid()
        new_ds.SeriesInstanceUID = new_series_uid
        new_ds.InstanceNumber = i + 1

        # Update ImagePositionPatient for each new slice
        origin = image.GetOrigin()
        spacing_z = image.GetSpacing()[2]
        new_ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + i * spacing_z]
        new_ds.SliceLocation = origin[2] + i * spacing_z

        new_ds.save_as(path / f"slice_{i:04d}.dcm")
    return None


if False:
    def load_ct_volume(path: pathlib.Path, series_uid: str | None = None) -> sitk.Image | Error:
        if path.is_dir():
            series_uids: list[str] = find_ct_series(path)
            if not series_uids:
                return Error(f"No DICOM series found in directory '{str(path)}'.")
            if series_uid is None:
                if len(series_uids) > 1:
                    return Error(
                        f"Multiple DICOM series found in directory '{str(path)}': {", ".join(series_uids)}; please "
                        f"specify one.")
                return load_ct_series(path, series_uids[0])
            else:
                if series_uid in series_uids:
                    return load_ct_series(path, series_uid)
                else:
                    return Error(f"DICOM series '{series_uid}' not found in directory '{str(path)}'.")
        elif path.is_file():
            if series_uid is not None:
                return Error(f"Path '{str(path)}' is a file, cannot specify series UID.")
            return load_image_file(path)
        else:
            return Error(f"Path '{str(path)}' is neither a file nor directory.")
