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


def deserialize_recursive(value: JsonSerializable,
                          old_value: HasTraitsSerializable | None = None) -> HasTraitsSerializable:
    if isinstance(value, list):
        if isinstance(old_value, list):
            if len(old_value) < len(value):
                old_value += [None] * (len(value) - len(old_value))
            return [deserialize_recursive(e, o) for e, o in zip(value, old_value)]
        return [deserialize_recursive(e) for e in value]
    if isinstance(value, dict):
        if isinstance(old_value, dict):
            return {k: deserialize_recursive(v, old_value[k] if k in old_value else None) for k, v in value.items()}
        if isinstance(old_value, traitlets.HasTraits):
            for k, v in value.items():
                try:
                    o = getattr(old_value, k)
                except Exception:
                    o = None
                n = deserialize_recursive(v, o)
                try:
                    setattr(old_value, k, n)
                except traitlets.TraitError as e:
                    logger.warning(f"Invalid value found for key '{k}' while deserializing into HasTraits: '{e}'")
                    continue
            return old_value
        return {k: deserialize_recursive(v) for k, v in value.items()}
    return value
