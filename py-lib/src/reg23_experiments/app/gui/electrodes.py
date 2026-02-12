import logging

import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.app.state import AppState
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager

__all__ = ["ElectrodesGUI"]

logger = logging.getLogger(__name__)


class ElectrodesGUI:
    def __init__(self, app_state: AppState):
        self._app_state = app_state
        self._save_manager = ElectrodeSaveManager(self._app_state.electrode_save_directory)
        tensor = self._save_manager.get(self._app_state.dadg.get("xray_path"))
        if tensor is None:
            self._points_layer = viewer().add_points(ndim=2, size=4.0, name="CI electrodes")
        else:
            self._points_layer = viewer().add_points(tensor.numpy(), size=4.0, name="CI electrodes")
            self._points_layer.text.values = [f"{i}" for i in range(tensor.size()[0])]
        self._app_state.dadg.set("electrode_points", tensor)
        self._points_layer.events.connect(self._on_layer_change)

    def _on_layer_change(self, event):
        if event.type == "data":
            tensor = torch.tensor(self._points_layer.data)
            self._points_layer.text.values = [f"{i}" for i in range(tensor.size()[0])]
            self._app_state.dadg.set("electrode_points", tensor)
            res = self._save_manager.set(self._app_state.dadg.get("xray_path"), tensor)
            if isinstance(res, Error):
                logger.error(f"Error saving electrode point data: {res.description}")
