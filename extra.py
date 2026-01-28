from neutron.core._module import Module, _check_for_static
from neutron.core._tracer import Tracer
from neutron.core._layers import ignore_list

from dataclasses import fields
from pprint import pprint

###################################
### Retrieves parameters amount ###
###################################

def get_params(model: Module) -> None:
    """
    Retrieves the total parameters for each layer and the total of both trainable and non-trainable parameters.\n
    """
    layers = vars(model)

    params: dict = {}
    total_params: int           = 0
    total_trainable: int        = 0
    total_non_trainable: int    = 0

    for layer in layers:
        current_layer   = getattr(model, layer)
        if isinstance(current_layer, Module):
            trainable, non_trainable = current_layer._get_layer_params()

            total_layer = trainable + non_trainable

            params[layer]       = total_layer
            total_params        += total_layer
            total_trainable     += trainable
            total_non_trainable += non_trainable
        else:
            if _check_for_static(model, layer) == True:
                total_non_trainable += 1
            else:
                total_trainable += 1
            total_params += 1
            params[layer] = 1

    params["total"] = total_params
    params["trainable_total"] = total_trainable
    params["non_trainable_total"] = total_non_trainable


    return params

########################
### Tree based stuff ### NOTE: Possibly a worse copy of get_tree()
########################

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

