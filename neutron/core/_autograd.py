from neutron.core._tracer import Tracer
from neutron.core._module import _check_for_static

def autograd(model, inputs) -> None:
    """
    Autograd is the function that gets called when the used starts the training process.
    Handles both the Forwards and Backwards process.
    """
    def forwards() -> None:
        """
        Simply calls the model class to proceed with the forwards process.
        Retrieves the final output as a Tracer object, which initializes the backwards process.
        """
        final_tracer: Tracer = model(*inputs)
        backwards(final_tracer)

    def backwards(final_tracer) -> None:
        final_tracer.backwards()

        updates: dict = get_current_update(model)
        # Optimizer here TODO: Check if it works with already made layers (like linear).
        model.__update__(updates)
        

    forwards()


def get_current_update(model) -> dict:
    """
    Just makes a big dict for all variables and their gradient and value
    """
    variables: dict = vars(model)
    updates: dict   = {}
    for variable in variables:
        if _check_for_static(model, variable) == False:
            var_value       = getattr(model, variable)
            var_gradient    = getattr(var_value, "gradient")

            updates[variable] = {"value" : var_value, "gradient" : var_gradient}

    return updates