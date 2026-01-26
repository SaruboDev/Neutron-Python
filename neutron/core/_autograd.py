from neutron.core._tracer import Tracer

def autograd(model) -> None:
    """
    Autograd is the function that gets called when the used starts the training process.
    Handles both the Forwards and Backwards process.
    """
    def forwards() -> None:
        """
        Simply calls the model class to proceed with the forwards process.
        Retrieves the final output as a Tracer object, which initializes the backwards process.
        """
        # Before this i gotta inject the tracers.
        final_tracer = model()
        backwards(final_tracer)

    def backwards(final_tracer) -> None:
        a = final_tracer.backwards()
        # To send the result to the optimizer later.

    forwards()

