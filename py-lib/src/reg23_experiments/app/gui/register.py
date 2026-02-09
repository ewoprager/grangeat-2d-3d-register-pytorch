import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets
import torch

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation

__all__ = ["RegisterGUI"]

logger = logging.getLogger(__name__)


class RegisterGUI(widgets.Container):
    def __init__(self, app_state: AppState):
        super().__init__(labels=True)
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
        self._run_button = widgets.PushButton(label="Run")
        self._run_button.changed.connect(self._on_run)

        self._one_iteration_button = widgets.PushButton(label="One iteration")
        self._one_iteration_button.changed.connect(self._on_one_iteration)

        self._job_state_description_label = widgets.Label()
        self._app_state.observe(self._update_job_state_description_label, names=["current_best_x"])

        self._load_current_best_button = widgets.PushButton(label="Load best x")
        self._load_current_best_button.changed.connect(self._on_load_current_best)

        self.append(widgets.Container(widgets=[  #
            widgets.Container(widgets=[  #
                self._run_button,  #
                self._one_iteration_button  #
            ], labels=False, layout="horizontal"),  #
            widgets.Container(widgets=[  #
                self._job_state_description_label,  #
                self._load_current_best_button  #
            ], labels=False, layout="horizontal"),  #
        ], layout="vertical", label="Optimise", labels=False))

        # ----
        # Transformations
        # ----
        current_t: Transformation = self._app_state.dag.get("current_transformation")
        current_params: torch.Tensor = mapping_transformation_to_parameters(current_t)
        # Float spin boxes for the current transformation in parameter space
        self._x_widgets = [  #
            widgets.FloatSpinBox(value=current_params[i].item(), step=0.01, min=-1.0e6, max=1.0e6)  #
            for i in range(6)  #
        ]
        self.append(widgets.Container(widgets=self._x_widgets, layout="horizontal", labels=False, label="x"))
        for i in range(6):
            self._x_widgets[i].changed.connect(self._update_current_transformation_from_x)
        self._x_loop_preventer = False
        # Float spin boxes for the current transformation in native units
        self._rotation_widgets = [  #
            widgets.FloatSpinBox(value=current_t.rotation[i].item(), step=0.001, min=-1.0e4, max=1.0e4)  #
            for i in range(3)  #
        ]
        self._translation_widgets = [  #
            widgets.FloatSpinBox(value=current_t.translation[i].item(), step=0.1, min=-1.0e8, max=1.0e8)  #
            for i in range(3)  #
        ]
        self.append(widgets.Container(widgets=self._rotation_widgets + self._translation_widgets, layout="horizontal",
                                      labels=False, label="T"))
        for i in range(3):
            self._rotation_widgets[i].changed.connect(self._update_current_transformation_from_t)
            self._translation_widgets[i].changed.connect(self._update_current_transformation_from_t)
        self._t_loop_preventer = False
        # A callback to keep both x and t spin boxes updated
        self._app_state.dag.add_callback("current_transformation", "x_t_display", self._update_x_t_display)

        # -----
        # Saving and loading transformations
        # -----
        self._saved_transformations_select = widgets.Select(choices=self._app_state.saved_transformation_names,
                                                            label="Saved\nTrs.")
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
        self._job_state_description_label.value = "No job has been run." if change.new is None else (f"Best x = "
                                                                                                     f"{change.new}.")

    def _on_one_iteration(self, *args) -> None:
        self._app_state.button_run_one_iteration = True

    def _on_run(self, *args) -> None:
        self._app_state.button_run = True

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

    def _update_current_transformation_from_x(self, *args) -> None:
        if self._x_loop_preventer:
            return
        self._x_loop_preventer = True
        params: list[float] = [widget.value for widget in self._x_widgets]
        current_t: Transformation = self._app_state.dag.get("current_transformation")
        new_params = torch.tensor(params, device=current_t.rotation.device, dtype=current_t.rotation.dtype)
        self._app_state.dag.set_data("current_transformation", mapping_parameters_to_transformation(new_params))
        self._x_loop_preventer = False

    def _update_current_transformation_from_t(self, *args) -> None:
        if self._t_loop_preventer:
            return
        self._t_loop_preventer = True
        current_t: Transformation = self._app_state.dag.get("current_transformation")
        rotation = torch.tensor([widget.value for widget in self._rotation_widgets], dtype=current_t.rotation.dtype,
                                device=current_t.rotation.device)
        translation = torch.tensor([widget.value for widget in self._translation_widgets],
                                   dtype=current_t.rotation.dtype, device=current_t.rotation.device)
        self._app_state.dag.set_data("current_transformation",
                                     Transformation(rotation=rotation, translation=translation))
        self._t_loop_preventer = False

    def _update_x_t_display(self, current_transformation: Transformation) -> None:
        if not self._x_loop_preventer:
            self._x_loop_preventer = True
            params: torch.Tensor = mapping_transformation_to_parameters(current_transformation)
            for i in range(6):
                self._x_widgets[i].value = params[i].item()
            self._x_loop_preventer = False
        if not self._t_loop_preventer:
            self._t_loop_preventer = True
            for i in range(3):
                self._rotation_widgets[i].value = current_transformation.rotation[i].item()
                self._translation_widgets[i].value = current_transformation.translation[i].item()
            self._t_loop_preventer = False

    def _on_load_current_best(self, *args) -> None:
        self._app_state.button_load_current_best = True
