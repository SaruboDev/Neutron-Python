from neutron.core._tracer import Tracer
from neutron.core._module import Module
from neutron.extra import _check_for_static

import numpy as np

class SGD:
    def __init__(self, lr: float = 0.001):
        self.lr = lr
    
    def __call__(self, params, *args, **kwds) -> dict:
        """
        Implementation of the Stochastic Gradient Descent.

        :param params: Parameters to optimize.
        :type params: any
        :param lr: Learning Rate.
        :type lr: float
        """

        def calculate(instance) -> dict:
            updated: dict = {}

            value       = instance.value
            gradient    = instance.gradient

            new_value = value - (self.lr * gradient)

            updated[instance] = {
                "value"     : new_value,
                "gradient"  : np.zeros(np.shape(value),np.dtype(value.dtype))\
                                if isinstance(value, np.ndarray) else 0
            }

            return updated

        def extract_value_grad(params):
            new_updates: dict = {}

            for variable in params:
                if isinstance(variable, Tracer):
                    new_updates.update(calculate(variable))
                    continue
                
                for inner_tracer in vars(variable):
                    tracer_instance = getattr(variable, inner_tracer)
                    if isinstance(tracer_instance, Tracer):
                        new_updates.update(calculate(tracer_instance))
                
            return new_updates

        new_updates: dict = extract_value_grad(params = params)
        # Will return a {instance : {value, gradient}}
        return new_updates
