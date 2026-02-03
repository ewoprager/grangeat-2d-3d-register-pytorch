import logging
from typing import Callable

import torch
from magicgui import widgets

from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import data_manager
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters
from reg23_experiments.ui.viewer_singleton import viewer

__all__ = ["RegisterGUI"]

logger = logging.getLogger(__name__)


class RegisterGUI(widgets.Container):
    def __init__(self, objective_function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(labels=False)

        self._objective_function = objective_function

        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_result_label = widgets.Label(label="Result:", value="n/a")

        self.append(widgets.Container(widgets=[self._eval_once_button, self._eval_result_label], layout="horizontal",
                                      label="Obj. func."))

        viewer().window.add_dock_widget(self, name="Register", area="right", menu=viewer().window.window_menu,
                                        tabify=True)

    def _on_eval_once(self, *args) -> None:
        tr = data_manager().get("current_transformation")
        if isinstance(tr, Error):
            logger.error(f"Error getting 'current_transformation' for eval. once: {tr.description}")
            return
        self._eval_result_label.value = self._objective_function(mapping_transformation_to_parameters(tr)).item()
