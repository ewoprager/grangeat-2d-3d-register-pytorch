from typing import NamedTuple, Any, Callable
import logging

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import magicgui, widgets

from registration.interface.lib.structs import *


def build_view_widget(get_view_params: Callable[[], ViewParams],
                      set_view_params: Callable[[ViewParams], None]) -> widgets.Widget:
    @magicgui(auto_call=True,
              translation_sensitivity={"widget_type": "FloatSlider", "min": 0.005, "max": 0.5, "step": 0.005,
                                       "label": "Translation sensitivity"},
              rotation_sensitivity={"widget_type": "FloatSlider", "min": 0.0005, "max": 0.05, "step": 0.0005,
                                    "label": "Rotation sensitivity"})
    def view_params(translation_sensitivity: float = 0.06, rotation_sensitivity: float = 0.002):
        nonlocal set_view_params
        set_view_params(
            ViewParams(translation_sensitivity=translation_sensitivity, rotation_sensitivity=rotation_sensitivity))

    return view_params
