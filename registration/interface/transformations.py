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


class TransformationManager:
    def __init__(self, initial_transformation: Transformation, refresh_render: Callable[[Transformation], None]):
        self.current_transformation = initial_transformation
        self.refresh_render_function = refresh_render
        self.widget = self.build_transformations_widget()
        self.supress_callbacks = False

    def get_current_transformation(self) -> Transformation:
        return self.current_transformation

    def set_transformation(self, new_value: Transformation) -> None:
        self.current_transformation = new_value
        self.supress_callbacks = True
        self.widget[0][0].set_value(self.current_transformation.translation[0])
        self.widget[0][1].set_value(self.current_transformation.translation[1])
        self.widget[0][2].set_value(self.current_transformation.translation[2])
        self.widget[0][3].set_value(self.current_transformation.rotation[0])
        self.widget[0][4].set_value(self.current_transformation.rotation[1])
        self.widget[0][5].set_value(self.current_transformation.rotation[2])
        self.supress_callbacks = False
        self.refresh_render_function(self.get_current_transformation())

    def get_widget(self) -> widgets.Widget:
        return self.widget

    def build_transformations_widget(self) -> widgets.Widget:
        @magicgui(auto_call=True, x_translation={"widget_type": "FloatSlider", "min": -200.0, "max": 200, "step": 0.2,
                                                 "label": "X translation"},
                  y_translation={"widget_type": "FloatSlider", "min": -200.0, "max": 200, "step": 0.2,
                                 "label": "Y translation"},
                  z_translation={"widget_type": "FloatSlider", "min": -300.0, "max": 300, "step": 0.2,
                                 "label": "Z translation"},
                  x_rotation={"widget_type": "FloatSlider", "min": -10.0, "max": 10, "step": 0.01,
                              "label": "X rotation"},
                  y_rotation={"widget_type": "FloatSlider", "min": -10.0, "max": 10, "step": 0.01,
                              "label": "Y rotation"},
                  z_rotation={"widget_type": "FloatSlider", "min": -10.0, "max": 10, "step": 0.01,
                              "label": "Z rotation"})
        def moving_parameters(x_translation: float = self.current_transformation.translation[0],
                              y_translation: float = self.current_transformation.translation[1],
                              z_translation: float = self.current_transformation.translation[2],
                              x_rotation: float = self.current_transformation.rotation[0],
                              y_rotation: float = self.current_transformation.rotation[1],
                              z_rotation: float = self.current_transformation.rotation[2]):
            nonlocal self
            if self.supress_callbacks:
                return
            self.current_transformation = Transformation(
                translation=torch.tensor([x_translation, y_translation, z_translation]),
                rotation=torch.tensor([x_rotation, y_rotation, z_rotation])).to(
                device=self.get_current_transformation().device())
            self.refresh_render_function(self.get_current_transformation())

        save_button = widgets.PushButton(label="Save new")
        saved_transformations_widget = WidgetSelectData(initial_choices={"initial": self.current_transformation})
        set_to_saved_button = widgets.PushButton(label="Load selected")
        del_button = widgets.PushButton(label="Delete selected")

        @save_button.changed.connect
        def _():
            nonlocal self, saved_transformations_widget
            if self.supress_callbacks:
                return
            new_name = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            saved_transformations_widget.add_choice(new_name, self.get_current_transformation())

        @set_to_saved_button.changed.connect
        def _():
            nonlocal self, saved_transformations_widget
            if self.supress_callbacks:
                return
            current = saved_transformations_widget.get_selected()
            if len(current) == 1:
                self.set_transformation(saved_transformations_widget.get_data(current[0]))
                self.refresh_render_function(self.get_current_transformation())
            elif len(current) == 0:
                logger.warning("No transformation selected.")
            else:
                logger.warning("Multiple transformations selected.")

        @del_button.changed.connect
        def _():
            nonlocal saved_transformations_widget
            if self.supress_callbacks:
                return
            current = saved_transformations_widget.get_selected()  # this actually returns a list of strings
            if len(current) == 0:
                logger.warning("No transformation selected.")
                return
            saved_transformations_widget.del_choices(current)

        return widgets.Container(widgets=[moving_parameters, widgets.Label(value="Transformations"),
                                          widgets.Container(widgets=[save_button, set_to_saved_button, del_button],
                                                            layout="horizontal"), saved_transformations_widget.widget],
                                 labels=False)
