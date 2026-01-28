# Neutron

A simple autograd compatible with numpy ndarrays.

Made by **SaruboDev**: <https://github.com/SaruboDev>.

Copyright: **Apache License 2.0 ©**, see github page for more.


### Description
I made this simple autograd so it can be used mostly as a mix between pytorch, jax and tensorflow.

I'm keeping most of tensorflow sintaxes to start training.

Model creation require a jax-style way to build them.

Example:
```py
class model(Module):
    linear: Linear
    
    def __init__(self):
        self.linear = Linear(input_dim = 1, output_dim = 1)

    def __call__(self, x, *args, **kwds):
        output = self.linear(x)

        return output
```
I'm also slowly making each module way more transparent so that you could technically grab each by itself and check how it works or make it do whatever you want.

NOTE: It's built with simple python, so no optimizations nor gpu compat (for now at least, idk, maybe in the future i'll learn c++ and port it over with some tweaks, when i feel like this one is mostly completed).

### How it's made
As of now we have 4 major classes and functions that make this work:
- Module:
    - It's the class that makes creating models and layers easy, handles it's __eq__ and __hash__ (even if you paste the same layer it will count them as different).

    - On initialization, converts all non-static variables to a Tracer class object.
    - Handles parameters update for the optimizer.
- Tracer:
    - Known also as Variable or Tensor, it's the class that keeps track of any change or operations done with it's values. Keeps track of it's parents operations and automatically calculates it's parent's gradients during the backwards process.
    - Uses a really simple Topological Ordering function to know which order of calculations to do.
- autograd fun:
    - A simple function that takes your model, your inputs and optimizer as input, automatically connects them together (for now, later on i will probably divide it into multiple sub-modules users can use freely), then gives back the output Tracer, where you can use `Tracer.value` to grab it's value and run loss and metrics on.
- field fun:
    - Technically the same exact code that Equinox uses to create it's own `eqx.static_field()` thingy, so that means you can make stuff static too here with `field(static = True)`.
 
### What does and doesn't work?
As of now the only test i did was:
```py
m = model()

sgd = SGD()

x = np.random.uniform(size = (1, 1))
result = autograd(m, x, sgd)

print(result)
```
So it's not a really good metric...

I'll have to test:
- If tracers successfully update their own gradient (should work by that little test).
- If the optimizer and `update(dict)` works to update weight and bias values.
- Ram usage.
- Once i add a good loss, optimizer and metric i'll test a simple classification model.

### Why did i do all this?
Funsies, it's my first actual project that is not a simple EDA or sliding window over an image to face recognition, also i hate SQL and R, so i'd never do a project on those stuff.
And i kinda wanted to know how backprop works, since most of the explanations on the internet just says "oh, backprop just calculates derivatives, dunno how though" and i was like "i suck at math! let's do it", can't lie i had lot's of researches and questions to do, my code is 90% mine, some stuff i found on stackoverflow, other stuff like "what was the difference between __getattribute__ and __getattr__?" i just asked gpt, lol.
