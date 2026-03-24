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


class TraitletsWidget(Container):
    def __init__(self, params: traitlets.HasTraits, **widget_kwargs):
        super().__init__(widgets=[], layout='vertical', labels=True, **widget_kwargs)

        self._params = params

        if isinstance(params, NoParameters):
            self.labels = False
            child = Label(value="n/a")
            self.append(child)
            return

        for name, trait in params.traits().items():
            if not trait.metadata.get("ui", False):
                continue

            value = getattr(params, name)
            child = TraitletsWidget._construct_child_widget(parent=self._params, name=name, trait_object=trait,
                                                            value=value)
            if isinstance(child, Error):
                logger.warning(f"Error constructing child widget: {child.description}")
                continue
            # set the value change callback appropriately
            if isinstance(child, Container):
                def replace_child(change, _name=name, _trait_object=trait):
                    self.remove(_name)
                    new_child = TraitletsWidget._construct_child_widget(parent=self._params, name=_name,
                                                                        trait_object=_trait_object, value=change["new"])
                    self.append(new_child)

                self._params.observe(replace_child, names=name)
            else:
                child.changed.connect(lambda v, n=name: setattr(self._params, n, v))

            self.append(child)

    @property
    def params(self) -> traitlets.HasTraits:
        return self._params

    @staticmethod
    def _construct_child_widget(*, parent: traitlets.HasTraits, name: str, trait_object: traitlets.TraitType,
                                value: Any) -> Widget | Error:
        # Float
        if isinstance(trait_object, traitlets.Float):
            child = FloatSpinBox(name=name, value=value)
            if trait_object.min is not None:
                child.min = trait_object.min
            if trait_object.max is not None:
                child.max = trait_object.max
            return child
        # Int
        elif isinstance(trait_object, traitlets.Int):
            child = SpinBox(name=name, value=value)
            if trait_object.min is not None:
                child.min = trait_object.min
            if trait_object.max is not None:
                child.max = trait_object.max
            return child
        # Bool
        elif isinstance(trait_object, traitlets.Bool):
            child = CheckBox(name=name, value=value)
            return child
        # Enum
        elif isinstance(trait_object, traitlets.Enum):
            child = ComboBox(name=name, choices=trait_object.values, value=value)
            return child
        # Unicode; ToDo: Currently read-only
        elif isinstance(trait_object, traitlets.Unicode):
            child = Label(name=name, value=value)
            return child
        # Sub-config
        elif isinstance(trait_object, traitlets.Instance) and isinstance(value, traitlets.HasTraits):
            child = TraitletsWidget(value, name=name)
            return child
        # Dict
        elif isinstance(trait_object, traitlets.Dict):
            child = Container()

            def update_dict_from_widget(_parent=parent, _name=name, container_widget=child):
                new_value = {  #
                    w.name: w.value  #
                    for w in container_widget  #
                }
                setattr(_parent, _name, new_value)

            for key, _value in value.items():
                widget = TraitletsWidget._construct_child_widget(parent=parent, name=key,
                                                                 trait_object=trait_object._value_trait, value=_value)
                if isinstance(child, Container):
                    def update_widget_from_dict(change, _parent=parent, _name=name, _trait_object=trait_object):
                        _parent.remove(_name)
                        new_child = TraitletsWidget._construct_child_widget(parent=_parent, name=_name,
                                                                            trait_object=_trait_object,
                                                                            value=change["new"])
                        _parent.append(new_child)

                    parent.observe(update_widget_from_dict, names=name)
                else:
                    widget.changed.connect(lambda v: update_dict_from_widget())
                child.append(widget)
            return child
        # Unsupported
        return Error(
            f"Unsupported trait class type '{trait_object.__class__.__name__}' encountered while building parameters "
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
