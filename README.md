# Neutron-Python

A simple autodiff compatible with numpy ndarrays.

Made by **SaruboDev**: <https://github.com/SaruboDev>.

Copyright: **Apache License 2.0 ©**, see github page for more.


### Description
I made this simple autodiff so it can be used mostly as a mix between pytorch, jax and tensorflow.

NOTE: It was made for study, not for use! It does work, but definitely not usable in big-scale projects!
You can still study how it was made, the worse thing is definitely finding out how the derivatives work for each function.

I'm keeping most of tensorflow sintaxes to start training.

Model creation require a jax-style way to build them.

Example:
```py
class model(Module):
    linear: Linear

    def __init__(self):
        self.linear     = Linear(input_dim = 32, output_dim = 32)

    def __call__(self, inputs, *args, **kwds):
        output = self.linear(inputs)

        return output
```
I'm also slowly making each module way more transparent so that you could technically grab each by itself and check how it works or make it do whatever you want.

NOTE: It's built with simple python, so no optimizations nor gpu compat (for now at least, idk, maybe in the future i'll learn rust and port it over with some tweaks, when i feel like this one is mostly completed).

### How it's made
As of now we have 4 major classes and functions that make this work:
- Module:
    - It's the class that makes creating models and layers easy, handles it's __eq__ and __hash__ (even if you paste the same layer it will count them as different).

    - On initialization, converts all non-static variables to a Tracer class object.
    - Handles parameters update for the optimizer.
- Tracer:
    - Known also as Variable or Tensor, it's the class that keeps track of any change or operations done with it's values. Keeps track of it's parents operations and automatically calculates it's parent's gradients during the backwards process.
    - Uses a really simple Topological Ordering function to know which order of calculations to do.
- Model:
    - A simple class that takes your model, your inputs and optimizer as input, automatically connects them together (for now, but if you dig enough you can also extract each function independently and do what you want), then gives show , where you can use `Tracer.value` to grab it's value and run loss and metrics on.
- field fun:
    - Technically the same exact code that Equinox uses to create it's own `eqx.static_field()` thingy, so that means you can make stuff static too here with `field(static = True)`.

As of now you'll see that i did place lots of placeholders around (like how in the Model.fit() method i added x_batch/x_eval, etc..) it's just for future proof if i want to update i know what i'm missing. The code **IS NOT** clean at all.
 
### What does and doesn't work?
As of now the only test i did was:
```py
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
```
So it's not a really good metric...

It doesn't support metrics and callbacks as of now (i have half the logic written here, but i do know how to finish it, but i'll have to update it sometime soon, maybe).

The output from the result is:
```py
Starting up...
Starting Epoch 1 / 2
Loss: 7.9106: 100%|██████████████████████████████████████████████████████████████████| 1.00/1.00 [00:00<00:00, 1.29kit/s]
Starting Epoch 2 / 2
Loss: 7.8713: 100%|██████████████████████████████████████████████████████████████████| 1.00/1.00 [00:00<00:00, 1.37kit/s]
(float32[(32, 32)],)
[[ 1.4457685e-01  1.9366897e+00  8.6078386e+00 ... -2.6224830e+00
  -9.2828715e-01  1.8296021e-01]
 [-2.1505828e-01  2.2428653e-01  3.5890570e+00 ...  1.8380655e+00
  -1.6603488e+00  1.0459497e+00]
 [-1.0003539e-01  9.6238470e-01  5.0091658e+00 ... -1.5841622e+00
   1.2702749e-03 -1.9826638e+00]
 ...
 [-2.0475726e+00 -1.9153960e-01  3.1436050e+00 ... -2.6002374e+00
  -7.2483891e-01 -1.4466027e+00]
 [-3.3510702e+00 -1.4462230e+00  4.3146396e+00 ... -9.4300604e-01
  -2.1255853e+00 -4.4674689e-01]
 [ 1.3007357e+00  1.6024392e+00  5.1451349e+00 ... -2.0669670e+00
  -8.3570933e-01 -2.9513717e+00]]
```
So you can see that the loss does lower with time! The result is a Tracer object, wrapped in a tuple just in case for multi-task models (should support them as well!), but simple enough, you only need to get the `.value` of it to get the result of the training, since i didn't write a test and predict function for it yet.

### Why did i do all this?
Funsies, it's my first actual project that is not a simple EDA or sliding window over an image to face recognition, also i hate SQL and R, so i'd never do a project on those stuff.
And i kinda wanted to know how backprop works, since most of the explanations on the internet just says "oh, backprop just calculates derivatives, dunno how though" and i was like "i suck at math! let's do it", can't lie i had lot's of researches and questions to do and answer, the code is 90% mine, some stuff i found on stackoverflow, other stuff like "what was the difference between __getattribute__ and __getattr__?" or "is it mathematically correct if i do..." i just asked ai's, lol, i ain't a mathematician. REMINDER: only for the theory, all the code you see here is mine, written once i understood the logic behind the task i needed to do.
