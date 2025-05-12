from typing import NamedTuple, Any, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import magicgui, widgets

from registration.lib.structs import Transformation
from registration.interface.lib.structs import *


class SavedTransformation(NamedTuple):
    name: str
    value: Transformation


def build_transformations_widget(get_transformation: Callable[[], Transformation],
                                 set_transformation: Callable[[Transformation], None], refresh: Callable[[], None]) -> (
        widgets.Widget):
    @magicgui(auto_call=True, z_translation={"widget_type": "FloatSlider", "min": -300.0, "max": 300, "step": 0.2})
    def moving_parameters(z_translation: float = 0.0):
        nonlocal set_transformation, get_transformation, refresh
        tr = get_transformation()
        tr.translation[2] = z_translation
        set_transformation(tr)
        refresh()

    save_button = widgets.PushButton(label="Save transformation")
    saved_transformations_widget = WidgetSelectData(initial_choices={"initial": get_transformation()})
    set_to_saved_button = widgets.PushButton(label="Set transformation to selected")
    del_button = widgets.PushButton(label="Delete selected")

    @save_button.changed.connect
    def _():
        nonlocal get_transformation, saved_transformations_widget
        new_name = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        saved_transformations_widget.add_choice(new_name, get_transformation())

    @set_to_saved_button.changed.connect
    def _():
        nonlocal set_transformation, saved_transformations_widget, refresh
        current = saved_transformations_widget.get_selected()
        if len(current) == 1:
            set_transformation(saved_transformations_widget.get_data(current[0]))
            refresh()
        elif len(current) == 0:
            logger.warning("No transformation selected.")
        else:
            logger.warning("Multiple transformations selected.")

    @del_button.changed.connect
    def _():
        nonlocal saved_transformations_widget
        current = saved_transformations_widget.get_selected()  # this actually returns a list of strings
        if len(current) == 0:
            logger.warning("No transformation selected.")
            return
        saved_transformations_widget.del_choices(current)

    return widgets.Container(
        widgets=[moving_parameters, save_button, saved_transformations_widget.widget, set_to_saved_button, del_button])
