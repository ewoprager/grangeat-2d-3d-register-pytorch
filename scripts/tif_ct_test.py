import sys
import pathlib
import logging

import tifffile

from reg23_experiments.utils import logs_setup

def tiff_tag_to_python(tag):
    value = tag.value

    # Already scalar → return directly
    if isinstance(value, (int, float, str)):
        return value

    # Rational stored as (num, denom)
    if tag.dtype.name in ["RATIONAL", "SRATIONAL"]:
        num, denom = value
        return num / denom if denom != 0 else float("nan")

    # Arrays → convert element-wise
    try:
        return [tiff_tag_to_python(type("T", (), {"value": v, "dtype": tag.dtype})) for v in value]
    except Exception:
        return value


def main():
    d = pathlib.Path(sys.argv[1])
    assert d.is_dir()

    tif_paths = [f for f in d.iterdir() if f.is_file() and (f.suffix == ".tif" or f.suffix == ".tiff")]
    with tifffile.TiffFile(tif_paths[0]) as first_file_object:
        logger.info(f"Found {len(first_file_object.pages)} pages in first .tif file '{str(tif_paths[0])}':")
        for page in first_file_object.pages:
            logger.info(f"\tshape = {page.shape}, dtype = {page.dtype}, axes = {page.axes}, XResolution = "
                        f"{tiff_tag_to_python(page.tags["XResolution"])}, dtype = {page.tags["XResolution"].dtype}, YResolution = "
                        f"{page.tags["YResolution"].value}, dtype = {page.tags["YResolution"].dtype}, ImageWidth = "
                        f"{page.tags["ImageWidth"].value}, "
                        f"dtype = {page.tags["ImageWidth"].dtype}")
            logger.info(page.asarray().dtype)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    main()
