import logging
import pathlib

from magicgui.widgets import request_values
from traitlets import TraitError

from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState
from reg23_experiments.experiments.parameters import XrayParameters

__all__ = ["FileManager"]

logger = logging.getLogger(__name__)


class FileManager:
    """
    No widgets, but does pop up file dialogues

    Reads from and write to the state only
    """

    def __init__(self, state: AppState):
        self._state = state

        self._state.observe(self._button_open_ct_file, names=["button_open_ct_file"])
        self._state.observe(self._button_open_ct_dir, names=["button_open_ct_dir"])
        self._state.observe(self._button_open_xray_file, names=["button_open_xray_file"])

    def _button_open_ct_file(self, change) -> None:
        if not change.new:
            return
        self._state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        file, _ = QFileDialog.getOpenFileName(viewer().window._qt_window, "Open a CT volume file")
        if not file:
            return
        logger.info(f"Opening CT volume file '{file}'")
        self._open_ct_path(file)

    def _button_open_ct_dir(self, change) -> None:
        if not change.new:
            return
        self._state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        dire = QFileDialog.getExistingDirectory(viewer().window._qt_window,
                                                "Open a directory of DICOM files as slices of a CT volume")
        if not dire:
            return
        logger.info(f"Opening DICOM files in directory '{dire}' as slices of CT volume")
        self._open_ct_path(dire)

    def _open_ct_path(self, path: str) -> None:
        try:
            self._state.parameters.ct_path = path
        except TraitError:
            logger.warning(f"CT path not valid: '{path}'")
            return

    def _button_open_xray_file(self, change) -> None:
        if not change.new:
            return
        self._state.button_open_xray_file = False

        # Get the user to choose a file
        from qtpy.QtWidgets import QFileDialog
        file, _ = QFileDialog.getOpenFileName(viewer().window._qt_window, "Open a CT volume file")
        if not file:
            return
        logger.info(f"Opening X-ray image file '{file}'")
        for _, ps in self._state.parameters.xray_parameters.items():
            if ps.file_path == file:
                logger.warning(f"X-ray '{file}' is already open.")
                return

        # Get a valid, unique name for the X-ray from the user
        prompt_string = "Enter a unique name for the X-ray"
        name = pathlib.Path(file).stem[:6]
        if name not in self._state.parameters.xray_parameters:
            prompt_string += f"; leave empty for default value '{name}' generated from file path"
        prompt_string += ":"
        values = request_values(name={"annotation": str, "label": prompt_string})
        if values["name"]:
            name = values["name"]
        while not name or name in self._state.parameters.xray_parameters:
            values = request_values(name={"annotation": str, "label": prompt_string})
            if values["name"]:
                name = values["name"]

        # Append it to the parameter state
        self._state.parameters.xray_parameters = {**self._state.parameters.xray_parameters,
                                                  name: XrayParameters(file_path=file)}
