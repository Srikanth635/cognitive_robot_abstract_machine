"""
The descriptors managing the fields that stand for ontology object properties.

What the ontology entails about a property's values in the closed world holds as soon as
a field is set or read: a sub-property's values are values of its super-properties, and
a symmetric property holds both ways. The descriptors keep this within the objects
themselves, so knowledge needs no graph outside the knowledge component.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from operator import attrgetter

from typing_extensions import Any, ClassVar


@dataclass
class ObjectPropertyDescriptor:
    """
    Manages the field standing for one ontology object property on a vocabulary class.

    A subclass stands for one property, and a subclass of that stands for one of its
    sub-properties.
    """

    domain: type
    """
    The vocabulary class declaring the managed field.
    """

    field_name: str
    """
    The name of the managed field.
    """

    symmetric: ClassVar[bool] = False
    """
    Whether the property is symmetric, so a value's field holds the instance in turn.
    """

    def __get__(self, instance: Any, owner: type) -> Any:
        """
        :return: The descriptor itself when read from the class; otherwise the field's
            values, joined by the values of the fields standing for its sub-properties.
        """
        if instance is None:
            return self
        stated = vars(instance)[self.field_name]
        if not isinstance(stated, list) or not self.is_specialised():
            return stated
        joined = list(stated)
        for value in self.specialised_values(instance):
            if value not in joined:
                joined.append(value)
        return joined

    def __set__(self, instance: Any, value: Any) -> None:
        vars(instance)[self.field_name] = value
        if not self.symmetric:
            return
        for target in value if isinstance(value, list) else [value]:
            if target is not None:
                self.hold(target, instance)

    @classmethod
    def is_specialised(cls) -> bool:
        """
        :return: Whether other properties specialise this one, so its values include
            theirs.
        """
        return bool(cls.__subclasses__())

    def specialised_values(self, instance: Any) -> list[Any]:
        """
        :param instance: An instance of a class with this property's field.
        :return: The values of the instance's fields standing for this property's
            sub-properties.
        """
        values = []
        for declared in fields(instance):
            descriptor = inspect.getattr_static(type(instance), declared.name, None)
            if not isinstance(descriptor, type(self)) or type(descriptor) is type(self):
                continue
            held = attrgetter(declared.name)(instance)
            values.extend(held if isinstance(held, list) else [held])
        return [value for value in values if value is not None]

    def hold(self, target: Any, instance: Any) -> None:
        """
        Make the target's field for this symmetric property hold the instance.

        :param target: A value of the instance's field.
        :param instance: The instance whose field was set.
        """
        held = vars(target).get(self.field_name)
        if isinstance(held, list):
            if instance not in held:
                held.append(instance)
            return
        vars(target)[self.field_name] = instance
