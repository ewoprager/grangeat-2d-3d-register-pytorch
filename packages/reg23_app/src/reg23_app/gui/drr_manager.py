import logging

import torch

from reg23_app.state import AppState
from reg23_experiments.data.structs import Error
from reg23_experiments.experiments.multi_xray_truncation_updaters import project_drr
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

        drr = temp_dadg.get("moving_image")
        logger.info(f"DRR projected: {drr}")
