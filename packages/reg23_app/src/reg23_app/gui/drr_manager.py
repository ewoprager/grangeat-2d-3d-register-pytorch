import logging
from datetime import datetime

import numpy as np
import pydicom
import torch
from pydicom.uid import ExplicitVRLittleEndian, UID, XRayRadiofluoroscopicImageStorage
from qtpy.QtWidgets import QFileDialog

from reg23_app.gui.viewer_singleton import viewer
from reg23_app.state import AppState
from reg23_experiments.data.structs import Error
from reg23_experiments.experiments.multi_xray_truncation_updaters import project_drr
from reg23_experiments.experiments.parameters import XrayParameters
from reg23_experiments.ops.data_manager import ChildDADG, DirectedAcyclicDataGraph

__all__ = ["DRRManager"]

logger = logging.getLogger(__name__)


class DRRManager:
    """
    No widgets

    Reads from and write to the state and DADG only
    """

    def __init__(self, state: AppState, dadg: DirectedAcyclicDataGraph):
        self._state = state
        self._dadg = dadg

        self._state.observe(self._button_create_drr, names=["button_create_drr"])

    def _button_create_drr(self, change) -> None:
        if not change.new:
            return
        self._state.button_create_drr = False

        if self._state.drr_name_input in self._state.parameters.xray_parameters:
            logger.error(f"Can't create DRR with name '{self._state.drr_name_input}' as this name is already in use.")
            return

        temp_dadg = ChildDADG(self._dadg)

        temp_dadg.set("fixed_image_size", torch.tensor([self._state.drr_params.height, self._state.drr_params.width]))
        temp_dadg.set("source_distance", self._state.drr_params.source_distance)
        temp_dadg.set("fixed_image_spacing",
                      torch.tensor([self._state.drr_params.x_spacing, self._state.drr_params.y_spacing]))
        temp_dadg.set("downsample_level", 0)
        temp_dadg.set("translation_offset", torch.zeros(2))
        temp_dadg.set("fixed_image_offset", torch.zeros(2))
        temp_dadg.set("image_2d_scale_factor", 1.0)

        err = temp_dadg.add_updater("project_drr", project_drr)
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")

        drr: torch.Tensor | Error = temp_dadg.get("moving_image")
        if isinstance(drr, Error):
            logger.error(f"Failed to project DRR: {drr.description}")
            return

        path, _ = QFileDialog.getSaveFileName(viewer().window._qt_window, "Save DRR", ".", "DICOM file (*.dcm)", )
        if path is None:
            logger.warning("No valid path given.")
            return

        ds = pydicom.Dataset()
        ds.PatientName = f"DRR_{self._state.drr_name_input}"
        ds.PatientID = "00000"
        now = datetime.now()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S.%f")  # long format with micro seconds
        file_meta = pydicom.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = XRayRadiofluoroscopicImageStorage
        file_meta.MediaStorageSOPInstanceUID = UID("1.2.3.4.5")
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = file_meta
        ds.RescaleIntercept = drr.min()
        drr -= ds.RescaleIntercept
        ds.RescaleSlope = drr.max() / 65536.0
        drr /= ds.RescaleSlope
        ds.PixelData = drr.cpu().numpy().astype(np.uint16).tobytes()
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = drr.size()[0]
        ds.Columns = drr.size()[1]
        ds.ImagerPixelSpacing = [self._state.drr_params.y_spacing, self._state.drr_params.x_spacing]
        ds.DistanceSourceToDetector = self._state.drr_params.source_distance
        ds.SOPInstanceUID = "12345." + now.strftime("%Y%m%d.%H%M%S.%f")

        logger.info(f"Saving DRR '{self._state.drr_name_input}' to '{path}'.")
        ds.save_as(path, enforce_file_format=True)

        self._state.parameters.xray_parameters = {**self._state.parameters.xray_parameters,
                                                  self._state.drr_name_input: XrayParameters(file_path=str(path))}
