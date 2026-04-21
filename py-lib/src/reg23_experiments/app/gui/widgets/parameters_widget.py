import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.data.structs import Transformation, Error
from reg23_experiments.app.gui.widgets.hastraits_widget import HasTraitsWidget
from reg23_experiments.app.context import AppContext
from reg23_experiments.ops.data_manager import args_from_dadg
from reg23_experiments.ops.geometry import get_crop_nonzero_drr, get_crop_full_depth_drr
from reg23_experiments.app.param_dadg_parity_manager import ParamDADGParityManager

__all__ = ["ParametersWidget"]

logger = logging.getLogger(__name__)


class ParametersWidget(widgets.Container):
    def __init__(self, ctx: AppContext, can_load_more_images: bool = True):
        super().__init__(widgets=[], layout='vertical', labels=False)

        self._ctx = ctx

        # -----
        # Parameters struct
        # -----
        self.append(widgets.Label(value="Values:"))
        self._traitlets_widget = HasTraitsWidget(self._ctx.state.parameters)
        self._traitlets_widget.expanded = True
        self.append(self._traitlets_widget)

        # -----
        # Functionality
        # -----
        self.append(widgets.Label(value="Modifiers:"))

        if can_load_more_images:
            # CT file opening
            self._open_ct_file_button = widgets.PushButton(label="Open CT file")
            self._open_ct_file_button.changed.connect(self._on_open_ct_file)
            self._open_ct_dir_button = widgets.PushButton(label="Open CT directory")
            self._open_ct_dir_button.changed.connect(self._on_open_ct_dir)

            self.append(widgets.Container(widgets=[  #
                self._open_ct_file_button,  #
                self._open_ct_dir_button  #
            ], layout="horizontal"))

            # X-ray file opening
            self._open_xray_file_button = widgets.PushButton(label="Open X-ray file")
            self._open_xray_file_button.changed.connect(self._on_open_xray_file)
            self.append(self._open_xray_file_button)

            # X-ray file unloading
            self._unload_xray_file_button = widgets.PushButton(label="Unload X-ray file: ")
            self._unload_xray_file_button.changed.connect(self._on_unload_xray_file)
            self._unload_xray_select = widgets.ComboBox(choices=self._get_xray_choices)
            self._ctx.state.parameters.observe(self._xray_params_changed, names=["xray_parameters"])
            self._unload_xray_select.changed.connect(self._on_unload_xray_choice_changed)
            self._on_unload_xray_choice_changed()
            self.append(widgets.Container(widgets=[  #
                self._unload_xray_file_button,  #
                self._unload_xray_select  #
            ], layout="horizontal"))

        # Cropping
        self._crop_nonzero_drr_button = widgets.PushButton(label="Crop to nonzero drr")
        self._crop_nonzero_drr_button.changed.connect(self._on_crop_nonzero_drr)
        self.append(self._crop_nonzero_drr_button)

        self._crop_full_depth_drr_button = widgets.PushButton(label="Crop to full depth drr")
        self._crop_full_depth_drr_button.changed.connect(self._on_crop_full_depth_drr)
        self.append(self._crop_full_depth_drr_button)

        # Transformations
        self._set_to_ground_truth_button = widgets.PushButton(label="Set transformation to G.T.")
        self._set_to_ground_truth_button.changed.connect(self._on_set_to_ground_truth)
        self.append(self._set_to_ground_truth_button)

    def _on_open_ct_file(self, *args) -> None:
        self._ctx.state.button_open_ct_file = True

    def _on_open_ct_dir(self, *args) -> None:
        self._ctx.state.button_open_ct_dir = True

    def _on_open_xray_file(self, *args) -> None:
        self._ctx.state.button_open_xray_file = True

    def _on_unload_xray_file(self, *args) -> None:
        self._ctx.state.button_unload_xray_file = True

    def _get_xray_choices(self, *args) -> list[str]:
        return list(self._ctx.state.parameters.xray_parameters.keys())

    def _xray_params_changed(self, change) -> None:
        self._unload_xray_select.reset_choices()

    def _on_unload_xray_choice_changed(self, *args) -> None:
        self._ctx.state.unload_xray_choice = self._unload_xray_select.value

    def _on_crop_nonzero_drr(self, *args) -> None:
        for k, v in self._ctx.state.parameters.xray_parameters.items():
            v.cropping = "Fixed"
            v.cropping_value = args_from_dadg(  #
                dadg=self._ctx.dadg,  #
                namespace_captures={e: k for e in ParamDADGParityManager.XRAY_SPECIFIC_DADG_KEYS}  #
            )(get_crop_nonzero_drr)()

    def _on_crop_full_depth_drr(self, *args) -> None:
        for k, v in self._ctx.state.parameters.xray_parameters.items():
            v.cropping = "Fixed"
            v.cropping_value = args_from_dadg(  #
                dadg=self._ctx.dadg,  #
                namespace_captures={e: k for e in ParamDADGParityManager.XRAY_SPECIFIC_DADG_KEYS}  #
            )(get_crop_full_depth_drr)()

    def _on_set_to_ground_truth(self, *args) -> None:
        for k, v in self._ctx.state.parameters.xray_parameters.items():
            transformation_gt: Transformation | Error = self._ctx.dadg.get(f"{k}__transformation_gt")
            if isinstance(transformation_gt, Error):
                logger.warning(f"No ground truth transformation exists for X-ray {k}; can't set to it.")
                return
            self._ctx.dadg.set(f"{k}__current_transformation", transformation_gt.clone())
