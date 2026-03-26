import logging
import traitlets

logger = logging.getLogger(__name__)

__all__ = ["JsonSerializable", "HasTraitsSerializable", "serialize_recursive", "deserialize_recursive"]

type JsonSerializable = None | bool | int | float | str | list[JsonSerializable] | dict[str, JsonSerializable]

type HasTraitsSerializable = None | bool | int | float | str | traitlets.HasTraits | list[HasTraitsSerializable] | dict[
    str, HasTraitsSerializable]  # where all traits of a HasTraits must be HasTraitsSerializable


def serialize_recursive(value: HasTraitsSerializable) -> JsonSerializable:
    if isinstance(value, traitlets.HasTraits):
        return {k: serialize_recursive(v) for k, v in value.trait_values().items()}
    if isinstance(value, dict):
        return {k: serialize_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_recursive(e) for e in value]
    return value


def deserialize_recursive(*, value: JsonSerializable, old_value: HasTraitsSerializable | None = None,
                          trait: traitlets.TraitType | None = None) -> HasTraitsSerializable:
    if isinstance(value, list):
        element_trait = None
        if trait is not None and isinstance(trait, traitlets.List):
            element_trait = trait._trait
        if isinstance(old_value, list):
            if len(old_value) < len(value):
                old_value += [None] * (len(value) - len(old_value))
            return [deserialize_recursive(value=e, old_value=o, trait=element_trait) for e, o in zip(value, old_value)]
        return [deserialize_recursive(value=e, trait=element_trait) for e in value]
    if isinstance(value, dict):
        hastraits_spec = None
        if isinstance(old_value, traitlets.HasTraits):
            hastraits_spec = old_value.traits()
        elif isinstance(trait, traitlets.Instance) and issubclass(trait.klass, traitlets.HasTraits):
            hastraits_spec = trait.klass._traits
        if hastraits_spec is not None:
            new_dict = dict({})
            for k, t in hastraits_spec.items():
                if k not in value:
                    continue
                try:
                    o = getattr(old_value, k)
                except Exception:
                    o = None
                new_dict[k] = deserialize_recursive(value=value[k], old_value=o, trait=t)

            if isinstance(old_value, traitlets.HasTraits):
                for k, v in new_dict.items():
                    try:
                        setattr(old_value, k, v)
                    except Exception as e:
                        logger.warning(f"Invalid value found for key '{k}' while deserializing into HasTraits: '{e}'")
                        continue
                return old_value
            else:
                return trait.klass(**new_dict)

        value_trait = None
        if trait is not None and isinstance(trait, traitlets.Dict):
            value_trait = trait._value_trait
        if isinstance(old_value, dict):
            return {
                k: deserialize_recursive(value=v, old_value=old_value[k] if k in old_value else None, trait=value_trait)
                for k, v in value.items()}
        return {k: deserialize_recursive(value=v, trait=value_trait) for k, v in value.items()}
    return value
