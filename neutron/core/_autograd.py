from neutron.core._tracer import Tracer
from neutron.core._module import Module, _check_for_static
from neutron.core._layers import ignore_list

from typing import Any
from pprint import pprint
import numpy as np

######################################
### Forwards and backwards handler ###
######################################

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

        updates: dict = get_tree(model)

        # Optimizer here
        new_updates: dict = optimizer(updates)
        update(new_updates)

        return final_tracer
        
    final_tracer = forwards()
    return final_tracer

##########################
### Update the tracers ###
##########################

def update(new_updates: dict) -> None:
    """
    Updates all instances with specified values and gradients.

    :param new_updates: A dict containing instances as key and an inner dict with key/value for value and gradient.
    :type new_updates: dict
    """
    for instance in new_updates:
        instance.value      = new_updates[instance]["value"]
        instance.gradient   = new_updates[instance]["gradient"]
    
    return

#############################
### Prints the whole tree ###
#############################

def get_tree(model: Module, instances_only: bool = True) -> dict:
    """
    Just makes a big dict for all variables and their gradient and value.\n
    Returns a dict.

    :param model: The model class to get the updates from.
    :type model: Module
    """
    def extract_inside(model: Module):
        variables   : dict = vars(model)
        updates     : dict = {}

        for variable in variables:
            variable_instance = getattr(model, variable)
            
            # Extracts instance from Tracer
            if (
                _check_for_static(model, variable) == False and
                isinstance(variable_instance, Tracer)
            ):
                updates[variable] = {
                    "instance"  : variable_instance
                }
        
            # If it's a Module but not a layer, just extract the stuff inside.
            if (
                isinstance(variable_instance, Module) and
                type(variable_instance) not in ignore_list
            ):
                module_updates: dict    = extract_inside(variable_instance)
                updates[variable]       = module_updates
            
            # If it's a Module and a layer, give just the instance.
            elif (
                isinstance(variable_instance, Module) and
                type(variable_instance) in ignore_list
            ):
                updates[variable] = {
                    "instance"  : variable_instance
                }
        
        return updates
    
    def extract_instances(model: Module):
        updates: list = []
        variables   : dict = vars(model)
        for variable in variables:
            variable_instance = getattr(model, variable)

            # Extracts instance from Tracer
            if (
                _check_for_static(model, variable) == False and
                isinstance(variable_instance, Tracer)
            ):
                updates.append(variable_instance)

            # If it's a Module but not a layer, just extract the stuff inside.
            if (
                isinstance(variable_instance, Module) and
                type(variable_instance) not in ignore_list
            ):
                module_updates: list    = extract_instances(variable_instance)

                updates.extend(module_updates)

            # If it's a Module and a layer, give just the instance.
            elif (
                isinstance(variable_instance, Module) and
                type(variable_instance) in ignore_list
            ):
                updates.append(variable_instance)
        return updates

    updates_list = extract_inside(model) if instances_only == False else extract_instances(model)
    return updates_list

