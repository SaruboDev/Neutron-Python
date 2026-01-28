from neutron.core._autograd import autograd, get_params
from neutron.core._tracer import Tracer
from neutron.core._module import Module, field, print_tree
from neutron.modules.linear import Linear
from neutron.optimizers.SGD import SGD

import numpy as np

"""
TO REMEMBER:
- The updates values are probably tracers, not too sure if it's good.
- As long as the "final_tracer" is loaded into memory, the whole graph is, too. UPDATE: Solved in tracer.backwards
"""

class model(Module):
    linear: Linear
    
    def __init__(self):
        self.linear     = Linear(input_dim = 1, output_dim = 1)

    def __call__(self, x, *args, **kwds):
        output = self.linear(x)

        return output

m = model()

sgd = SGD()

x = np.random.uniform(size = (1, 1))
result = autograd(m, x, sgd)
print(result)
# print_tree(m)

# print(get_params(m))