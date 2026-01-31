from neutron.core._autograd import autograd, Model
from neutron.core._tracer import Tracer
from neutron.core._module import Module, field
from neutron.modules import Linear
from neutron.optimizers import SGD, Adam
from neutron.losses import log_loss

import numpy as np

"""
TO REMEMBER:
- The updates values are probably tracers, not too sure if it's good.
- As long as the "final_tracer" is loaded into memory, the whole graph is, too. UPDATE: Solved in tracer.backwards
"""

"""
TODO:
- Make a loss fn DONE: Made the classic log_loss + softmax too.
- Make Adam optimizer DONE: Was pretty easy, now i have SGD and Adam
- Try to make a classification model
- Test to check if everything works by remaking the same one in jax or tensorflow
- Subdivide the autograd function so each module can be used separately by the users

Correct order of operations:
- Forward
- Loss
- extract graph instances : values from loss tracer
- optimizer
- update
- receive last tracer from the forward

TODO NEW:
- Add np.max functionality in the Tracer.
- Fix backwards for np.sum when i can
- Probably also fix the backwards for np.max, since it only calculates the gradient on the max value
    of the array and the other values get gradient 0.
- Remember that log_loss as of now always does with from_logits == False, so, i gotta make it
    a class, and make the training loop call it's __call__, so the user can customize it.
"""

class inner(Module):
    linear3: Linear

    def __init__(self):
        self.a = 3.0
        self.linear3 = Linear(1, 1)

    def __call__(self, inputs):
        return inputs

class model(Module):
    linear: Linear

    def __init__(self):
        self.linear     = Linear(input_dim = 32, output_dim = 32)
        self.x = 3.0

    def __call__(self, inputs, *args, **kwds):
        output = self.linear(inputs)

        return output


m = Model(model)
m.compile(
    optimizer = Adam(),
    loss=[log_loss],
    metrics=[],
    callbacks=[]
)

x = np.random.uniform(size = (32, 32))
m.fit(
    data = None,
    x_train=x,
    y_train=x,
    epochs=2
)