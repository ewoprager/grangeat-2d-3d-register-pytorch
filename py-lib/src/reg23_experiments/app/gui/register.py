import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState

__all__ = ["RegisterGUI"]

logger = logging.getLogger(__name__)


class RegisterGUI(widgets.Container):
    def __init__(self, app_state: AppState):
        super().__init__(labels=False)
        self._app_state = app_state

        # -----
        # Evaluate once button and result
        # -----
        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_once_result_label = widgets.Label(label="Result:", value="n/a")
        self._app_state.observe(self._update_eval_once_result_label, names=["eval_once_result"])

        self.append(widgets.Container(widgets=[  #
            self._eval_once_button,  #
            self._eval_once_result_label,  #
        ], layout="horizontal", label="Obj. func."))

        # -----
        # Run optimisation
        # -----
        self._job_state_description_label = widgets.Label()
        self.append(self._job_state_description_label)
        self._app_state.observe(self._update_job_state_description_label, names=["job_state_description"])

        self._one_iteration_button = widgets.PushButton(label="One iteration")
        self._one_iteration_button.changed.connect(self._on_one_iteration)

        self.append(widgets.Container(widgets=[  #
            self._job_state_description_label,  #
            self._one_iteration_button,  #
        ], layout="horizontal", label="Optimise"))

        # add self as widget in dock to the right
        viewer().window.add_dock_widget(self, name="Register", area="right", menu=viewer().window.window_menu,
                                        tabify=True)

    def _on_eval_once(self, *args) -> None:
        self._app_state.button_evaluate_once = True

    def _update_eval_once_result_label(self, change) -> None:
        self._eval_once_result_label.value = "n/a" if change.new is None else change.new

    def _update_job_state_description_label(self, change) -> None:
        self._job_state_description_label.value = "No job has been run." if change.new is None else change.new

    def _on_one_iteration(self, *args) -> None:
        self._app_state.button_run_one_iteration = True
