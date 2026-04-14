import logging
from typing import Any

import traitlets
from magicgui.widgets import CheckBox, ComboBox, Container, FloatSpinBox, Label, SpinBox, Widget

from reg23_experiments.data.structs import Error

__all__ = ["HasTraitsWidget"]

logger = logging.getLogger(__name__)


def value_from_widget(widget: Widget) -> Any:
    if isinstance(widget, HasTraitsWidget):
        return widget.value
    if isinstance(widget, Container):
        # ToDo: Currently only works with dicts
        return {  #
            element.name: value_from_widget(element)  #
            for element in widget  #
        }
    return widget.value


class HasTraitsWidget(Container):
    def __init__(self, hastraits: traitlets.HasTraits, **widget_kwargs):
        super().__init__(widgets=[], layout="vertical", labels=False, **widget_kwargs)

        self._expand_toggle = CheckBox(value=False, text="expand")
        self._expand_toggle.changed.connect(self._expand_toggled)
        self.append(self._expand_toggle)
        self._inner_container = Container(widgets=[], layout="vertical", labels=True)
        self.append(self._inner_container)

        self._expand_toggled()

        self._callback_loop_prevention: set = set({})  # just a flag keyed by widget name
        self._hastraits = hastraits

        for name, trait in self._hastraits.traits().items():
            if not trait.metadata.get("ui", False):
                continue

            value = getattr(self._hastraits, name)
            child = HasTraitsWidget._construct_child_widget(name=name, trait=trait, value=value)
            if isinstance(child, Error):
                logger.warning(f"Error constructing child widget: {child.description}")
                continue

            # Any traits that aren't instances of HasTraits have their values read directly from the widget to update
            # trait value
            if not (isinstance(trait, traitlets.Instance) or isinstance(trait, traitlets.Union)):
                def update_hastraits_from_widget(new_value: Any, _name=name) -> None:
                    if _name in self._callback_loop_prevention:
                        return
                    self._callback_loop_prevention.add(_name)
                    setattr(self._hastraits, _name, new_value)
                    self._callback_loop_prevention.remove(_name)

                child.changed.connect(update_hastraits_from_widget)

            # All traits have the widget either updated or re-built when the hastraits value changes
            def update_widget_from_hastraits(change, _name=name, _trait=trait, widget=child) -> None:
                if _name in self._callback_loop_prevention:
                    return
                self._callback_loop_prevention.add(_name)
                if isinstance(_trait, traitlets.Instance) or isinstance(_trait, traitlets.Dict) or isinstance(widget,
                                                                                                              Container):
                    self._inner_container.remove(_name)
                    self._inner_container.append(
                        HasTraitsWidget._construct_child_widget(name=_name, trait=_trait, value=change["new"]))
                else:
                    widget.value = change["new"]
                self._callback_loop_prevention.remove(_name)

            self._hastraits.observe(update_widget_from_hastraits, names=[name])

            # Dict traits have additional callbacks for the elements of the dict
            if isinstance(trait, traitlets.Dict):
                def update_dict_trait_from_widget(new_value: Any, _name=name, _widget=child) -> None:
                    if _name in self._callback_loop_prevention:
                        return
                    self._callback_loop_prevention.add(_name)
                    setattr(self._hastraits, _name, value_from_widget(_widget))
                    self._callback_loop_prevention.remove(_name)

                for element_widget in child:
                    element_widget.changed.connect(update_dict_trait_from_widget)

            self._inner_container.append(child)

    # Forward all members of the inner container
    def __getattr__(self, item):
        return getattr(self._inner_container, item)

    @property
    def expanded(self) -> bool:
        return self._expand_toggle.value

    @expanded.setter
    def expanded(self, value) -> None:
        self._expand_toggle.value = value

    @property
    def value(self) -> traitlets.HasTraits:
        return self._hastraits

    @staticmethod
    def _construct_child_widget(*, name: str, trait: traitlets.TraitType, value: Any) -> Widget | Error:
        if trait.allow_none and value is None:
            ret = Label(name=name, value="n/a")
            return ret
        # Union
        elif isinstance(trait, traitlets.Union):
            for t in trait.trait_types:
                try:
                    t.validate(None, value)
                except traitlets.TraitError:
                    continue
                return HasTraitsWidget._construct_child_widget(name=name, trait=t, value=value)
            return Error(f"`value` conformed to none of the `Union` trait's `trait_types`.")
        # Float
        elif isinstance(trait, traitlets.Float):
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
            ret.native.setWordWrap(True)
            return ret
        # Sub-config
        elif isinstance(trait, traitlets.Instance) and isinstance(value, traitlets.HasTraits):
            ret = HasTraitsWidget(value, name=name)
            return ret
        # Dict
        elif isinstance(trait, traitlets.Dict):
            ret = Container(name=name)
            for key, dict_value in value.items():
                dict_value_widget = HasTraitsWidget._construct_child_widget(name=key, trait=trait._value_trait,
                                                                            value=dict_value)
                ret.append(dict_value_widget)
            return ret
        # Unsupported
        return Error(f"Unsupported trait class type '{trait.__class__.__name__}' encountered while building parameters "
                     f"widget.")

    def _expand_toggled(self, *args) -> None:
        self._inner_container.visible = self._expand_toggle.value
