import traitlets
import logging
from typing import Any

from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Qt
import napari
from magicgui.widgets import Widget, FloatSpinBox, SpinBox, CheckBox, ComboBox, Container, Label

from reg23_experiments.data.structs import Error
from reg23_experiments.experiments.parameters import NoParameters

__all__ = ["TraitletsWidget", "FloatingWidget"]

logger = logging.getLogger(__name__)


def value_from_widget(widget: Widget) -> Any:
    if isinstance(widget, TraitletsWidget):
        return widget.value
    if isinstance(widget, Container):
        # ToDo: Currently only works with dicts
        return {  #
            element.name: value_from_widget(element)  #
            for element in widget  #
        }
    return widget.value


class TraitletsWidget(Container):
    def __init__(self, hastraits: traitlets.HasTraits, **widget_kwargs):
        super().__init__(widgets=[], layout='vertical', labels=True, **widget_kwargs)

        self._callback_loop_prevention: bool = False
        self._hastraits = hastraits

        if isinstance(self._hastraits, NoParameters):
            self.labels = False
            child = Label(value="n/a")
            self.append(child)
            return

        for name, trait in self._hastraits.traits().items():
            if not trait.metadata.get("ui", False):
                continue

            value = getattr(self._hastraits, name)
            child = TraitletsWidget._construct_child_widget(name=name, trait=trait, value=value)
            if isinstance(child, Error):
                logger.warning(f"Error constructing child widget: {child.description}")
                continue

            # Any traits that aren't instances of HasTraits have their values read directly from the widget to update
            # trait value
            if not isinstance(trait, traitlets.Instance):
                def update_hastraits_from_widget(new_value: Any, _name=name) -> None:
                    if self._callback_loop_prevention:
                        return
                    self._callback_loop_prevention = True
                    setattr(self._hastraits, _name, new_value)
                    self._callback_loop_prevention = False

                child.changed.connect(update_hastraits_from_widget)

            # All traits have the widget either updated or re-built when the hastraits value changes
            def update_widget_from_hastraits(change, _name=name, _trait=trait, widget=child) -> None:
                if self._callback_loop_prevention:
                    return
                self._callback_loop_prevention = True
                if isinstance(trait, traitlets.Instance) or isinstance(trait, traitlets.Dict):
                    self.remove(_name)
                    self.append(TraitletsWidget._construct_child_widget(name=_name, trait=_trait, value=change["new"]))
                else:
                    widget.value = change["new"]
                self._callback_loop_prevention = False

            self._hastraits.observe(update_widget_from_hastraits, names=[name])

            # Dict traits have additional callbacks for the elements of the dict
            if isinstance(trait, traitlets.Dict):
                def update_dict_trait_from_widget(new_value: Any, _name=name, _widget=child) -> None:
                    if self._callback_loop_prevention:
                        return
                    self._callback_loop_prevention = True
                    setattr(self._hastraits, _name, value_from_widget(_widget))
                    self._callback_loop_prevention = False

                for element_widget in child:
                    element_widget.changed.connect(update_dict_trait_from_widget)

            self.append(child)

    @property
    def value(self) -> traitlets.HasTraits:
        return self._hastraits

    @staticmethod
    def _construct_child_widget(*, name: str, trait: traitlets.TraitType, value: Any) -> Widget | Error:
        # Float
        if isinstance(trait, traitlets.Float):
            ret = FloatSpinBox(name=name, value=value)
            if trait.min is not None:
                ret.min = trait.min
            if trait.max is not None:
                ret.max = trait.max
            return ret
        # Int
        elif isinstance(trait, traitlets.Int):
            ret = SpinBox(name=name, value=value)
            if trait.min is not None:
                ret.min = trait.min
            if trait.max is not None:
                ret.max = trait.max
            return ret
        # Bool
        elif isinstance(trait, traitlets.Bool):
            ret = CheckBox(name=name, value=value)
            return ret
        # Enum
        elif isinstance(trait, traitlets.Enum):
            ret = ComboBox(name=name, choices=trait.values, value=value)
            return ret
        # Unicode; ToDo: Currently read-only
        elif isinstance(trait, traitlets.Unicode):
            ret = Label(name=name, value=value)
            return ret
        # Sub-config
        elif isinstance(trait, traitlets.Instance) and isinstance(value, traitlets.HasTraits):
            ret = TraitletsWidget(value, name=name)
            return ret
        # Dict
        elif isinstance(trait, traitlets.Dict):
            ret = Container(name=name)
            for key, dict_value in value.items():
                dict_value_widget = TraitletsWidget._construct_child_widget(name=key, trait=trait._value_trait,
                                                                            value=dict_value)
                ret.append(dict_value_widget)
            return ret
        # Unsupported
        return Error(f"Unsupported trait class type '{trait.__class__.__name__}' encountered while building parameters "
                     f"widget.")


class FloatingWidget(QWidget):
    def __init__(self, title: str, widget: QWidget):
        super().__init__(napari.Viewer().window._qt_window)

        self.setWindowTitle(title)

        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

        # Optional: behave like a tool panel
        self.setWindowFlags(Qt.Window |  # make it a real window
                            Qt.Tool  # stays on top of parent, no taskbar entry
                            )
