import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_app.context import AppContext

__all__ = ["FiducialsWidget"]

logger = logging.getLogger(__name__)


class FiducialsWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(widgets=[], layout='vertical', labels=True)

        self._ctx = ctx

        # Fiducial parameters
        self._fiducial_diameter_box = widgets.FloatSpinBox(label="Assumed fiducial diameter", min=0.0)
        self._fiducial_diameter_box.changed.connect(self._on_diameter)
        self.append(self._fiducial_diameter_box)

        # -----
        # CT
        # -----
        self._refine_ct_fiducials_button = widgets.PushButton(label="Auto-refine CT fiducial seg.")
        self._refine_ct_fiducials_button.changed.connect(self._on_refine_ct_fiducials)
        self.append(self._refine_ct_fiducials_button)

        # -----
        # X-ray specific
        # -----
        # X-ray selection
        self._xray_select = widgets.ComboBox(choices=self._get_xray_choices, label="X-ray")
        self._ctx.state.parameters.observe(self._xray_params_changed, names=["xray_parameters"])
        self._xray_select.changed.connect(self._on_xray_choice_changed)
        self._on_xray_choice_changed()
        self.append(self._xray_select)

        # Refine segmentation
        self._refine_xray_fiducials_button = widgets.PushButton(label="Auto-refine X-ray fiducial seg.")
        self._refine_xray_fiducials_button.changed.connect(self._on_refine_xray_fiducials)
        self.append(self._refine_xray_fiducials_button)

        # Register using fiducials
        self._register_button = widgets.PushButton(label="Register using fiducials")
        self._register_button.changed.connect(self._on_register)
        self.append(self._register_button)

    def _on_register(self, *args) -> None:
        self._ctx.state.button_fiducial_register = True

    def _get_xray_choices(self, *args) -> list[str]:
        return list(self._ctx.state.parameters.xray_parameters.keys())

    def _xray_params_changed(self, change) -> None:
        self._xray_select.reset_choices()

    def _on_xray_choice_changed(self, *args) -> None:
        self._ctx.state.register_fiducial_xray_choice = self._xray_select.value

    def _on_refine_ct_fiducials(self, *args) -> None:
        self._ctx.state.button_refine_ct_fiducials = True

    def _on_refine_xray_fiducials(self, *args) -> None:
        self._ctx.state.button_refine_xray_fiducials = True

    def _on_diameter(self, *args) -> None:
        self._ctx.state.assumed_fiducial_diameter = self._fiducial_diameter_box.value
