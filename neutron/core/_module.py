import dataclasses
from dataclasses import dataclass, fields
from typing import Any
from neutron.core._tracer import Tracer

def field(
        *,
        static: bool = False,
        **kwargs: Any
) -> Any:
    """
    Adding this because we kinda need the static functionality, that is, to avoid having things traced when they shouldn't be.   
    """
    try: # Ripped this try except off of Equinox c:
        metadata = dict(kwargs.pop("metadata"))
    except KeyError:
        metadata = {}

    if static:
        metadata["static"] = True

    return dataclasses.field(metadata = metadata, **kwargs)

class ModuleMeta(type):
    def __new__(cls, name, bases, clsdict):
        """
        Just creates the class, but doesn't initialize it.
        We also already turn it into a dataclass, so we can use our static metadata.

        :param cls: Metaclass
        :param name: Name of the class
        :param bases: class
        :param clsdict: Attributes and variables of the class.
        """
        cls = super().__new__(cls, name, bases, clsdict)
        dataclass(cls)

        return cls
    
    def __call__(self, *args, **kwds):
        """
        Gets called each time a new instance of this MetaClass is called.
        This is a good excuse to trace the variables inside.
        Update: Realized too late that i also had __init_subclass__ as a valid method without creating a metaclass.
        """
        instance = super().__call__(*args, **kwds)

        instance.__trace__()

        return instance

@dataclass
class Module(metaclass = ModuleMeta):
    """
    This is gonna be a simple module thing, which will handle hashing and eq's.
    It will also have a static method for variables we do not want to change to Tracers.
    And finally, will also have a method to automatically change it's own non-static variables to Tracers.
    """

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, value) -> bool:
        """
        For equivalence i'd say for now it's always false, since i may want to use the exact same layers as duplicate.
        """

        if not isinstance(value, self.__class__):
            return NotImplemented
        
        return self is value

    def __trace__(self) -> None:
        """
        Converts every non-static variable to a Tracer(value).
        """
        variables: dict = vars(self)
        for variable in variables:
            if _check_for_static(self, variable) == False and not isinstance(getattr(self, variable), Tracer):
                setattr(self, variable, Tracer(getattr(self, variable)))

    def __update__(self, updates) -> None:
        """
        Replaces old values with the new updates given by the optimizer.

        :param updates: New parameter updates.
        :type updates: dict
        """
    
        variables: dict = vars(self)
        for variable in variables:
            if _check_for_static(self, variable) == False and not isinstance(getattr(self, variable), Tracer):
                setattr(self, variable, updates[variable]["value"])


def _check_for_static(class_target, variable):
    """
    Simply checks for the field metadata "static".
    """
    for field in fields(class_target):
        if field.name == variable:
            return field.metadata.get("static", False)
    return False

