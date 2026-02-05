import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.data.structs import Error
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

        # -----
        # Save transformations
        # -----
        self._saved_transformations_select = widgets.Select(choices=self._app_state.saved_transformation_names)
        self._app_state.observe(self._update_saved_transformations_select, names=["saved_transformation_names"])
        self.append(self._saved_transformations_select)
        self._transformation_name_input = widgets.LineEdit(value="")
        self._transformation_name_input.changed.connect(self._on_transformation_name_input)
        self._save_transformation = widgets.PushButton(label="Save current")
        self._save_transformation.changed.connect(self._on_save_transformation)
        self._load_transformation = widgets.PushButton(label="Load selected")
        self._load_transformation.changed.connect(self._on_load_transformation)
        self._delete_transformation = widgets.PushButton(label="Delete selected")
        self._delete_transformation.changed.connect(self._on_delete_transformation)
        self.append(widgets.Container(widgets=[  #
            self._transformation_name_input,  #
            self._save_transformation,  #
            self._load_transformation,  #
            self._delete_transformation,  #
        ], layout="horizontal"))

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

    def _update_saved_transformations_select(self, change) -> None:
        # ToDo: manage persistence of selection?
        self._saved_transformations_select.choices = change.new

    def _on_transformation_name_input(self, *args) -> None:
        self._app_state.text_input_transformation_name = self._transformation_name_input.get_value()

    def _on_save_transformation(self, *args) -> None:
        self._app_state.button_save_transformation = True

    def _get_single_transformation_selection(self) -> str | Error:
        selected: list[str] = self._saved_transformations_select.current_choice
        if not selected:
            return Error("No item selected.")
        if len(selected) > 1:
            return Error("Multiple items selected.")
        return selected[0]

    def _on_load_transformation(self, *args) -> None:
        res = self._get_single_transformation_selection()
        if isinstance(res, Error):
            logger.warning(f"Failed to load transformation: {res.description}")
            return
        self._app_state.button_load_transformation_of_name = res

    def _on_delete_transformation(self, *args) -> None:
        res = self._get_single_transformation_selection()
        if isinstance(res, Error):
            logger.warning(f"Failed to delete transformation: {res.description}")
            return
        self._app_state.button_delete_transformation_of_name = res
