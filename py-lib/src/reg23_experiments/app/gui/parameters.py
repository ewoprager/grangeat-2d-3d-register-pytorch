import traitlets
import logging
from typing import Any

from magicgui.widgets import Widget, FloatSpinBox, SpinBox, CheckBox, ComboBox, Container, Label

from reg23_experiments.experiments.parameters import NoParameters

logger = logging.getLogger(__name__)


class ParameterWidget(Container):
    def __init__(self, params: traitlets.HasTraits):
        super().__init__(widgets=[], layout='vertical', labels=True)

        self._params = params
        self._subwidgets: dict[str, Any] = {}

        if isinstance(params, NoParameters):
            self.labels = False
            child = Label(value="n/a")
            self.append(child)
            return

        for name, trait in params.traits().items():
            if not trait.metadata.get("ui", False):
                continue

            value = getattr(params, name)

            # Float
            if isinstance(trait, traitlets.Float):
                child = FloatSpinBox(name=name, value=value)
                if trait.min is not None:
                    child.min = trait.min
                if trait.max is not None:
                    child.max = trait.max
                child.changed.connect(lambda v, n=name: setattr(params, n, v))
                self.append(child)
            # Int
            elif isinstance(trait, traitlets.Int):
                child = SpinBox(name=name, value=value)
                if trait.min is not None:
                    child.min = trait.min
                if trait.max is not None:
                    child.max = trait.max
                child.changed.connect(lambda v, n=name: setattr(params, n, v))
                self.append(child)
            # Bool
            elif isinstance(trait, traitlets.Bool):
                child = CheckBox(name=name, value=value)
                child.changed.connect(lambda s, n=name: setattr(params, n, bool(s)))
                self.append(child)
            # Enum
            elif isinstance(trait, traitlets.Enum):
                child = ComboBox(name=name, choices=trait.values, value=value)
                child.changed.connect(lambda v, n=name: setattr(params, n, v))
                self.append(child)
            # Unicode; ToDo: Currently read-only
            elif isinstance(trait, traitlets.Unicode):
                child = Label(name=name, value=value)
                self.append(child)
            # Sub-config
            elif isinstance(trait, traitlets.Instance) and isinstance(value, traitlets.HasTraits):
                child = ParameterWidget(value)
                child.name = name
                self._subwidgets[name] = child

                # callback for the instance changing
                def replace_child(change, _name=name):
                    old = self._subwidgets[_name]
                    self.remove(_name)
                    del old

                    new_child = ParameterWidget(change["new"])
                    new_child.name = _name
                    self.append(new_child)
                    self._subwidgets[_name] = new_child

                params.observe(replace_child, names=name)
                self.append(child)
            # Unsupported
            else:
                logger.warning(
                    f"Unsupported trait class type '{trait.__class__.__name__}' encountered while building parameters "
                    f"widget.")

    @property
    def params(self) -> traitlets.HasTraits:
        return self._params
