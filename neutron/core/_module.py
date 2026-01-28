import dataclasses
from dataclasses import dataclass, fields
from typing import Any
from neutron.core._tracer import Tracer
from pprint import pprint

import numpy as np

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

    def _update(self, updates) -> None:
        """
        Replaces old values with the new updates given by the optimizer.

        :param updates: New parameter updates.
        :type updates: dict
        """
        variables: dict = vars(self)        

        for layers in updates:
            if layers in variables and isinstance(variables[layers], Module):
                variables[layers]._update(updates[layers])
            elif layers in variables:
                # print(f"Before {layers} : {getattr(self, layers).value}\n")
                setattr(self, layers, updates[layers]["value"])
                # print(f"After {layers} : {getattr(self, layers).value}")
        return

def _check_for_static(class_target, variable):
    """
    Simply checks for the field metadata "static".
    """
    for field in fields(class_target):
        if field.name == variable:
            return field.metadata.get("static", False)
    return False

def make_tree(model: Module):
    """
    Creates a dictionary object containing name and attributes of each trainable and non-trainable object.\n
    Any traceable variable will be automatically converted to a Tracer object, by how the Module class is built.
    
    :param model: The object to flatten.
    :type model: Module
    """
    complete_tree: dict = {}
    inside_tree: dict   = {}

    for field in fields(model):
        instance = getattr(model, field.name)
        if isinstance(instance, Module):
            inner_tree = make_tree(instance)
            inside_tree[field.name] = inner_tree

        elif not field.metadata.get("static", False):
            inside_tree[field.name] = getattr(model, field.name)

    complete_tree[model.__class__.__name__] = inside_tree

    return complete_tree

def print_tree(model: Module):
    """
    Prints the object given as a dictionary with it's inner values.\n
    Any traceable variable will be automatically converted to a Tracer object, by how the Module class is built.

    :param model: The object to print.
    :type model: Module
    """
    pprint(make_tree(model), indent = 1)