from neutron.core._tracer import Tracer

import dataclasses
from dataclasses import dataclass, fields
from typing import Any
from pprint import pprint
import numpy as np

#############################
### Adds the static field ###
#############################

def field(
        *,
        static: bool = False,
        **kwargs: Any
) -> Any:
    """
    Adding this because we kinda need the static functionality, that is, to avoid having things traced when they shouldn't be.   
    """
    try: # Ripped this try except from Equinox c:
        metadata = dict(kwargs.pop("metadata"))
    except KeyError:
        metadata = {}

    if static:
        metadata["static"] = True

    return dataclasses.field(metadata = metadata, **kwargs)

############################
### Metaclass for Module ###
############################

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
        dataclass(cls, eq=False)

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

###########################
### Actual Module class ###
###########################

@dataclass
class Module(metaclass = ModuleMeta):
    """
    This is gonna be a simple module thing, which will handle hashing and eq's.
    It will also have a static method for variables we do not want to change to Tracers.
    And finally, will also have a method to automatically change it's own non-static variables to Tracers.
    """
    def _temp(self):
        """
        You shouldn't be looking here?
        """
        print("aaa")

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
            var_value = getattr(self, variable)
            if (
                _check_for_static(self, variable) == False and
                not isinstance(var_value, Tracer) and
                not isinstance(var_value, Module) and
                isinstance(var_value, (float, np.ndarray))
            ):
                setattr(self, variable, Tracer(getattr(self, variable)))

#########################
### Checks for static ###
#########################

def _check_for_static(class_target, variable: str):
    """
    Simply checks for the field metadata "static".

    :param class_target: Instance or type class to check.
    :type class_target: Preferably Module, but any should work.
    :param variable: Name of the variable you want to check.
    :type variable: str
    """
    for field in fields(class_target):
        if field.name == variable:
            return field.metadata.get("static", False)
    return False
