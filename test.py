from neutron.core._autograd import autograd, Model
from neutron.core._tracer import Tracer
from neutron.core._module import Module, field
from neutron.modules import Linear
from neutron.optimizers import SGD, Adam
from neutron.losses import log_loss

import numpy as np

class model(Module):
    linear: Linear

    def __init__(self):
        self.linear     = Linear(input_dim = 32, output_dim = 32)

    def __call__(self, inputs, *args, **kwds):
        output = self.linear(inputs)

        return output

m = Model(model)
loss_fn = log_loss(from_logits = True)
m.compile(
    optimizer = Adam(),
    loss=[loss_fn],
    metrics=[],
    callbacks=[]
)

x = np.random.randn(32, 32)
y = np.random.randint(0, 2, 32)
y_onehot = np.eye(32)[y]

res = m.fit(
    data = None,
    x_train=x,
    y_train=y_onehot,
    epochs=2
)

print(res)
print(res[0].value)