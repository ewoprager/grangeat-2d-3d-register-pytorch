import logging

import traitlets

logger = logging.getLogger(__name__)

__all__ = ["JsonSerializable", "HasTraitsSerializable", "serialize_recursive", "deserialize_recursive"]

type JsonSerializable = None | bool | int | float | str | list[JsonSerializable] | dict[str, JsonSerializable]
"""Any type that can be trivially serialized into JSON / YAML"""

type HasTraitsSerializable = None | bool | int | float | str | traitlets.HasTraits | list[HasTraitsSerializable] | dict[
    str, HasTraitsSerializable]  # where all traits of a HasTraits must be HasTraitsSerializable
"""Any type that can be converted to and from `JsonSerializable` with `serialize_recursive` and `deserialize_recursive`.
This is any type that is/recursively contains trivially serializable types, or objects that derive from
`traitlets.HasTraits`.
"""


def serialize_recursive(value: HasTraitsSerializable, *, trait: traitlets.TraitType | None = None) -> JsonSerializable:
    """
    Convert a `HasTraitsSerializable` into a `JsonSerializable`.
    :param value: The value to serialize
    :param trait: [Optional] A trait which the value should conform to. This allows serializing unions and HasTraits
    objects.
    :return: A copy of the given value that is trivially serializable into JSON / YAML.
    """
    # Given a trait that allows None, and the value is None
    if trait is not None and trait.allow_none and value is None:
        return None
    # Given a trait, and it's a `List`
    if isinstance(trait, traitlets.List):
        if not isinstance(value, list):
            logger.error(f"Non-`list` value '{str(value)}' found while serializing `List` trait.")
            return None
        return [serialize_recursive(e, trait=trait._trait) for e in value]
    # Given a trait, and it's a `Dict`
    if isinstance(trait, traitlets.Dict):
        if not isinstance(value, dict):
            logger.error(f"Non-`dict` value '{str(value)}' found while serializing `Dict` trait.")
            return None
        return {k: serialize_recursive(v, trait=trait._value_trait) for k, v in value.items()}
    # Given a trait, and it's an `Instance` of `HasTraits`
    if isinstance(trait, traitlets.Instance):
        if not issubclass(trait.klass, traitlets.HasTraits):
            logger.error(
                "Can't serialize `Instance` traits for classes that aren't derived from `traitlets.HasTraits`.")
            return None
        if not isinstance(value, trait.klass):
            logger.error(f"Failed to serialize value '{str(value)}' as it's type does not match that of trait.")
            return None
        return {  #
            k: serialize_recursive(getattr(value, k), trait=t)  #
            for k, t in trait.klass._traits.items()  #
            if hasattr(value, k)  #
        }
    # Given a trait, and it's a `Union`
    if isinstance(trait, traitlets.Union):
        if not all([isinstance(t, traitlets.Instance) for t in trait.trait_types]):
            logger.error(f"Only `Instance` trait types are currently supported for (de)serializing `Union` traits.")
            return None
        if any([t.allow_none for t in trait.trait_types]):
            logger.error(
                f"Only trait types that allow None are not currently supported for (de)serializing `Union` traits.")
            return None
        for t in trait.trait_types:
            if isinstance(value, t.klass):
                return {  #
                    "___union_choice": t.klass.__name__,  #
                    "___union_value": serialize_recursive(value, trait=t),  #
                }
        logger.error(f"`Union` trait not satified by value of type '{type(value).__name__}'.")
        return None

    # Not given a trait, or it's unrecognised
    if isinstance(value, traitlets.HasTraits):
        return {  #
            k: serialize_recursive(getattr(value, k), trait=t)  #
            for k, t in value.traits().items()  #
            if hasattr(value, k)  #
        }
    if isinstance(value, dict):
        return {k: serialize_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_recursive(e) for e in value]
    return value


def deserialize_recursive(value: JsonSerializable, *, old_value: HasTraitsSerializable | None = None,
                          trait: traitlets.TraitType | None = None) -> HasTraitsSerializable:
    """
    Convert a `JsonSerializable` into a `HasTraitsSerializable`.
    :param value: The value to convert
    :param old_value: [Optional] An old version of this value that can be used for two things:
        - Where values are missing from containers in `value`, values in `old_value` will be used,
        - Where `old_value` contains `HasTraits` objects, these will be used to construct new `HasTraits` objects from
        `value`, where otherwise a simple `dict` would be produced.
    :param trait: [Optional] A trait that the output should conform to. This will override the use of `old_value`'s type
    if `old_value` is also provided.
    :return:
    """
    # Given a trait that allows None, and the value is None
    if trait is not None and trait.allow_none and value is None:
        return None
    # Given a trait, and it's a `List`
    if isinstance(trait, traitlets.List):
        if not isinstance(value, list):
            logger.error(f"Non-`list` value '{str(value)}' found while deserializing `List` trait.")
            return None
        element_trait = trait._trait
        if isinstance(old_value, list):
            if len(old_value) < len(value):
                old_value += [None] * (len(value) - len(old_value))
            return [deserialize_recursive(e, old_value=o, trait=element_trait) for e, o in zip(value, old_value)]
        return [deserialize_recursive(e, trait=element_trait) for e in value]
    # Given a trait, and it's a `Dict`
    if isinstance(trait, traitlets.Dict):
        if not isinstance(value, dict):
            logger.error(f"Non-`dict` value '{str(value)}' found while deserializing `Dict` trait.")
            return None
        value_trait = trait._value_trait
        if isinstance(old_value, dict):
            return {k: deserialize_recursive(v, old_value=old_value[k] if k in old_value else None, trait=value_trait)
                    for k, v in value.items()}
        return {k: deserialize_recursive(v, trait=value_trait) for k, v in value.items()}
    # Given a trait, and it's an `Instance` of `HasTraits`
    if isinstance(trait, traitlets.Instance):
        if not issubclass(trait.klass, traitlets.HasTraits):
            logger.error(
                "Can't deserialize `Instance` traits for classes that aren't derived from `traitlets.HasTraits`.")
            return None
        new_dict = dict({})
        for k, t in trait.klass._traits.items():
            if k not in value:
                continue
            try:
                o = getattr(old_value, k)
            except Exception:
                o = None
            new_dict[k] = deserialize_recursive(value[k], old_value=o, trait=t)
        if isinstance(old_value, trait.klass):
            for k, v in new_dict.items():
                try:
                    setattr(old_value, k, v)
                except Exception as e:
                    logger.warning(f"Invalid value found for key '{k}' while deserializing into HasTraits: '{e}'")
                    continue
            return old_value
        else:
            return trait.klass(**new_dict)
    # Given a trait, and it's a `Union`
    if isinstance(trait, traitlets.Union):
        if not isinstance(value, dict):
            logger.error(f"Non-`dict` value '{str(value)}' found while deserializing `Union` trait.")
            return None
        if "___union_choice" not in value:
            logger.error(f"Key '___union_choice' not found in dict for `Union` trait.")
            return None
        if "___union_value" not in value:
            logger.error(f"Key '___union_value' not found in dict for `Union` trait.")
            return None
        if not all([isinstance(t, traitlets.Instance) for t in trait.trait_types]):
            logger.error(f"Only `Instance` trait types are currently supported for (de)serializing `Union` traits.")
            return None
        for t in trait.trait_types:
            if t.klass.__name__ == value["___union_choice"]:
                return deserialize_recursive(value["___union_value"], old_value=old_value, trait=t)
        # the given choice did not match any of the options in the `Union` trait
        logger.error(f"Invalid union choice '{value["___union_choice"]}' found while deserializing `Union` trait.")
        return None

    # Not given a trait, or it's unrecognised
    if isinstance(value, list):
        if isinstance(old_value, list):
            if len(old_value) < len(value):
                old_value += [None] * (len(value) - len(old_value))
            return [deserialize_recursive(e, old_value=o) for e, o in zip(value, old_value)]
        return [deserialize_recursive(e) for e in value]
    if isinstance(value, dict):
        if isinstance(old_value, traitlets.HasTraits):
            for k, t in old_value.traits().items():
                if k not in value:
                    continue
                try:
                    o = getattr(old_value, k)
                except Exception:
                    o = None
                v = deserialize_recursive(value[k], old_value=o, trait=t)
                try:
                    setattr(old_value, k, v)
                except Exception as e:
                    logger.warning(f"Error assigning field '{k}' while deserializing into HasTraits: '{e}'")
                    continue
            return old_value
        if isinstance(old_value, dict):
            return {k: deserialize_recursive(v, old_value=old_value[k] if k in old_value else None) for k, v in
                    value.items()}
        return {k: deserialize_recursive(v) for k, v in value.items()}
    return value
