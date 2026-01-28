from neutron.core._tracer import Tracer
from neutron.core._module import Module
from neutron.core._module import _check_for_static

from typing import Any
from pprint import pprint
import numpy as np

def autograd(model: Module, inputs, optimizer) -> None:
    """
    Autograd is the function that gets called when the used starts the training process.
    Handles both the Forwards and Backwards process.

    :param model: The custom model you want to train.
    :type model: Module
    :param inputs: The inputs for the model, can also be a tuple to divide in multiple inputs.
    :type inputs: np.ndarray|tuple(np.ndarray)
    """
    def forwards() -> None:
        """
        Simply calls the model class to proceed with the forwards process.
        Retrieves the final output as a Tracer object, which initializes the backwards process.
        """
        final_tracer: Tracer = model(*inputs)
        backwards(final_tracer, model)
        return final_tracer

    def backwards(final_tracer: Tracer, model):
        final_tracer.backwards()

        updates: dict = get_current_update(model)
        # Keep in mind that as long as the final_tracer keeps track of its parents, the whole graph is in memory, even
        # temporary tracers, meaning, that we kinda need to either rewrite it to None, or use weakref or smt.

        # Optimizer here
        new_updates: dict = optimizer(updates)
        
        model._update(new_updates)
        return final_tracer
        
    final_tracer = forwards()
    return final_tracer


def get_current_update(model: Module) -> dict:
    """
    Just makes a big dict for all variables and their gradient and value.

    :param model: The model class to get the updates from.
    :type model: Module
    """
    variables: dict = vars(model)
    updates: dict   = {}
    for variable in variables:
        variable_instance = getattr(model, variable)
        if _check_for_static(model, variable) == False and isinstance(variable_instance, Tracer):
            var_value       = getattr(model, variable)
            var_gradient    = getattr(var_value, "gradient")

            updates[variable] = {"value" : var_value, "gradient" : var_gradient}
        elif isinstance(variable_instance, Module):
            updates[variable] = get_current_update(variable_instance)

    return updates

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

def get_graph_preview(model: Module) -> dict:
    """
    WIP
    Retrieves the overall graph of all generated Tracers in a simple forward with dummy input.\n
    Does not update weights nor run backwards.
    """
    ...