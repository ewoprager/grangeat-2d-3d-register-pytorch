import logging
import pathlib
from typing import NamedTuple

import pandas as pd

from reg23_experiments.data.structs import Error
from reg23_experiments.data.transformation_save_data import TransformationSaveManager
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveManager
from reg23_experiments.io.image import XrayDICOM, read_dicom

__all__ = ["ImageSpecificConfigurations"]

logger = logging.getLogger(__name__)


class ImageSpecificConfigurations(NamedTuple):
    saved_transformations: pd.DataFrame
    saved_xray_reg_configs: pd.DataFrame

    @staticmethod
    def load() -> 'ImageSpecificConfigurations':
        # -----
        # Load all saved transformations; these are searched through for ground truth alignments
        saved_transformations: pd.DataFrame = TransformationSaveManager(  #
            directory=pathlib.Path("data/app_transformation_save_data"),  #
        ).get_all()
        logger.info(f"Saved transformation data:\n{saved_transformations.to_string()}")

        # -----
        # Load all saved X-ray configs; these are used for manual X-ray configurations
        saved_xray_reg_configs: pd.DataFrame = XRayRegSaveManager(  #
            directory=pathlib.Path("data/xray_reg_save_data"),  #
        ).get_all()
        logger.info(f"Saved X-ray reg configs:\n{saved_xray_reg_configs.to_string()}")

        return ImageSpecificConfigurations(saved_transformations, saved_xray_reg_configs)

    def check_xray_path(self, p: str | pathlib.Path) -> Error | None:
        """
        Check that all X-rays exist, have ground truth transformations available, and have reg configs available
        :param p:
        :return:
        """
        p = pathlib.Path(p)
        if not p.is_file():
            return Error(f"X-ray file '{str(p)}' doesn't exist.")
        try:
            dicom: XrayDICOM = read_dicom(p)
        except Exception as e:
            return Error(f"Failed to read X-ray file: {e}")
        idx = (dicom["uid"], "gold_standard")
        try:
            self.saved_transformations.loc[idx]
        except KeyError:
            return Error(f"No ground truth saved for X-ray '{str(p)}' with UID '{dicom["uid"]}'.")
        idx = dicom["uid"]
        try:
            self.saved_xray_reg_configs.loc[idx]
        except KeyError:
            return Error(f"No reg config saved for X-ray '{str(p)}' with UID '{dicom["uid"]}'.")
        return None
