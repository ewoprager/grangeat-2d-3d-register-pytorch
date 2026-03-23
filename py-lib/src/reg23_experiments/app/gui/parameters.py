import logging

from magicgui import widgets

from reg23_experiments.data.structs import Transformation, Error
from reg23_experiments.app.gui.helpers import TraitletsWidget
from reg23_experiments.app.context import AppContext
from reg23_experiments.ops.data_manager import args_from_dadg
from reg23_experiments.ops.geometry import get_crop_nonzero_drr, get_crop_full_depth_drr

__all__ = ["ParametersWidget"]

logger = logging.getLogger(__name__)


class ParametersWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(widgets=[], layout='vertical', labels=False)

        self._ctx = ctx

        self.append(widgets.Label(value="Values:"))
        self.append(TraitletsWidget(self._ctx.state.parameters))

        self.append(widgets.Label(value="Modifiers:"))

        self._crop_nonzero_drr_button = widgets.PushButton(label="Crop to nonzero drr")
        self._crop_nonzero_drr_button.changed.connect(self._on_crop_nonzero_drr)
        self.append(self._crop_nonzero_drr_button)

        self._crop_full_depth_drr_button = widgets.PushButton(label="Crop to full depth drr")
        self._crop_full_depth_drr_button.changed.connect(self._on_crop_full_depth_drr)
        self.append(self._crop_full_depth_drr_button)

        self._set_to_ground_truth_button = widgets.PushButton(label="Set transformation to G.T.")
        self._set_to_ground_truth_button.changed.connect(self._on_set_to_ground_truth)
        self.append(self._set_to_ground_truth_button)

    def _on_crop_nonzero_drr(self, *args) -> None:
        self._ctx.state.parameters.cropping = "fixed"
        self._ctx.state.parameters.cropping_value = args_from_dadg(dadg=self._ctx.dadg)(get_crop_nonzero_drr)()

    def _on_crop_full_depth_drr(self, *args) -> None:
        self._ctx.state.parameters.cropping = "fixed"
        self._ctx.state.parameters.cropping_value = args_from_dadg(dadg=self._ctx.dadg)(get_crop_full_depth_drr)()

    def _on_set_to_ground_truth(self, *args) -> None:
        transformation_gt: Transformation | Error = self._ctx.dadg.get("transformation_gt")
        if isinstance(transformation_gt, Error):
            logger.warning("No ground truth transformation exists; can't set to it.")
            return
        self._ctx.dadg.set("current_transformation", transformation_gt.clone())
