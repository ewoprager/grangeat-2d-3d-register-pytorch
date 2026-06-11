import logging
import pathlib
import pprint

from reg23_experiments.data.structs import Error
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.serialize import serialize_recursive
from ._data import OneSeries, SeriesDescription, Volume
from ._dicom import DICOMVolumeLoader
from ._nii import NiiVolumeLoader
from ._nrrd import NrrdVolumeLoader
from ._xtekct import XTekCTVolumeLoader

_registry = [NiiVolumeLoader, NrrdVolumeLoader, DICOMVolumeLoader, XTekCTVolumeLoader]

__all__ = ["find_ct_series", "load_ct_series", "load_one_ct_series", "get_input_ct_series_choice"]

logger = logging.getLogger(__name__)


def find_ct_series(path: pathlib.Path) -> dict[str, SeriesDescription | OneSeries]:
    ret: dict[str, SeriesDescription | OneSeries] = {}
    for loader in _registry:
        series = loader.series_available(path)
        if isinstance(series, OneSeries):
            ret[series.file_type] = series
        else:
            if overlap := ret.keys() & series.keys():
                logger.warning(  #
                    f"Found duplicates of CT series with the following UIDs at path '{str(path)}': "
                    f"{", ".join(list(overlap))}")
            ret |= series
    return ret


def load_ct_series(path: pathlib.Path, key: str) -> Volume | Error:
    for loader in _registry:
        available = loader.series_available(path)
        if isinstance(available, OneSeries):
            if key == available.file_type:
                return loader.load(path, None)
        elif key in available:
            return loader.load(path, key)
    return Error(f"Couldn't find loader for series '{key}' at path '{str(path)}'.")


def load_one_ct_series(path: pathlib.Path) -> Volume | Error:
    for loader in _registry:
        series = loader.series_available(path)
        if isinstance(series, OneSeries):
            return loader.load(path, None)
        elif series:
            return loader.load(path, next(iter(series)))
    return Error(f"Couldn't find loader for path '{str(path)}'.")


def get_input_ct_series_choice(series: dict[str, SeriesDescription | OneSeries]) -> str:
    return get_string_required(  #
        f"Please choose one of the following CT series:\n"
        f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
        lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
    )
