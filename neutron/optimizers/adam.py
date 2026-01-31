from neutron.core._tracer import Tracer

import numpy as np

class Adam:
    lr: float
    b1: float
    b2: float
    t: int

    old_step: dict

    def __init__(self, lr: float = 0.001, b1: float = 0.9, b2: float = 0.999):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.old_step = {
            "m" : {},
            "v" : {}
        }
        self.t = 0

    def __call__(self, params, *args, **kwargs) -> dict:
        self.t += 1

        def update_old(instance, m, v) -> None:
            self.old_step["m"][instance] = m
            self.old_step["v"][instance] = v
            return

        def calculate(instance):
            updated: dict = {}

            old_m = self.old_step["m"].get(instance, 0.0)
            old_v = self.old_step["v"].get(instance, 0.0)

            value       = instance.value
            gradient    = instance.gradient

            m = self.b1 * old_m + (1 - self.b1) * gradient
            v = self.b2 * old_v + (1 - self.b2) * (gradient**2)

            m_adjusted = m / (1 - self.b1 ** self.t)
            v_adjusted = v / (1 - self.b2 ** self.t)

            new_value = value - self.lr * (m_adjusted / (np.sqrt(v_adjusted) + 1e-8))

            updated[instance] = {
                "value"     : new_value,
                "gradient"  : np.zeros(np.shape(value),np.dtype(value.dtype))\
                                if isinstance(value, np.ndarray) else 0
            }

            update_old(instance, m, v)
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
        
        new_updates: dict = extract_value_grad(params)
        
        return new_updates