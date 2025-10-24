# Neutron

A simple wrapper for jax, equinox and optax.

Made by **SaruboDev**: <https://github.com/SaruboDev>.

Copyright: **Apache License 2.0 ©**, see github page for more.


Should support other modules that interact with jax, depending on which one. *+[WARNING]** This was made as a small project for me to learn the jax environment more, you should expect some problems, if you find any, feel free to tell me.

### Quickstart
You can use Neutron wrapper *mostly* like you would use Tensorflow.
Make your model class in jax style, call our Model in neutron.models, compile and fit!

**Just a warning**: i'm fairly new to the jax environment myself, so lots of stuff may probably be unoptimized or not entirely correct. Feel free to let me know!

Adds a few quality of life stuff to make you able to speedrun tests, i'm not yet sure about big-project development with this module.
The goal with this wrapper is to make jax more accessible and train your models faster with Jit compilation.

For example, having your custom model (this model sucks, it's just for example)::

    class custom_model(equinox.Module):
        vocab_size: int         = equinox.field(static = True)
        embedding_dim: int      = equinox.field(static = True)
        embed   : equinox.nn.Embedding
        output  : equinox.nn.Linear
        
        def __init__(self, vocab_size: int, embedding_dim: int, key: jax.random.PRNGKey):
            keyEmbed, keyOut    = jax.random.split(key, 2)

            self.vocab_size     = vocab_size
            self.embedding_dim  = embedding_dim

            self.embed  = equinox.nn.Embedding(vocab_size, embedding_dim, key = keyEmbed)
            self.output = equinox.nn.Linear(embedding_dim, 1, key = keyOut)
        
        def __call__(self, x, key: jax.random.PRNGKey):
            # Depending on what you need you might want to vmap one, or multiple layers.
            # In this case i'll just vmap everything [DO NOT UNLESS YOU NEED IT].
            x   = jax.vmap(self.embed)(x)
            out = jax.vmap(self.output)(x)
            return out

You can now wrap your model like this (note: 512 is vocab_size and 256 is embedding dim in this example)::

    from neutron.models import Model

    model = Model(custom_model, 512, 256)
    model.compile(
        optimizer = optax.adam(learning_rate = 0.0001),
        loss = [(optax.losses.sigmoid_binary_cross_entropy, 1.0)],
        metrics = [accuracy],
        gradAccSteps = 3
    )

    history = model.fit(
        data = None,
        x_train = jnp.zeros((250,), dtype = jnp.uint16), # Just for the example.
        y_train = None,
        batch_size = 32,
        epochs = 1,
        steps_for_epoch = None,
        callbacks = [],
        verbose = True,
        starting_epoch = 1,
        starting_step_train = 0
    )

    model.predict(x)

You might notice a few things:
1) The model in the example sucks and it's probably wrong. **True**.
2) You don't really need to specify the PRNGKey if not inside your custom model, True, when you call Model() you can set "seed" if you want
    your custom PRNGKey, but otherwise it will handle it by itself.
3) The loss is a list of tuples. This is for compatibility with multi-task models, each tuple is a loss_fun - weight pair.
    If you only need one loss, you can just write "loss = loss_fun" by itself, and it will automatically get converted.
4) You have "data" and also "x/y_train", why? Because data is mostly if you're planning on using the custom class datasetGenerator to
    load bigger datasets. x_train can be used alone if you have self-supervised models, the y_train will automatically be replaced with x_train.
5) Callbacks can be used just like TensorFlow's ones, my implementation probably lacks of calls, though.
6) The predict function can take either 1 element of an array of elements and it will give the prediction based on your model.
7) The fit function returns a "history". The history element is just an array of each metric per-epoch and evaluation. Not for each steps.

#### What it currently adds:
- A Model class that accepts custom models and make you able to fit instantly without defining your own train cycle.
    - **Supports**:
    - Single Task Models;
    - Multi Task Models (mostly);
    - Self-supervised Models;
    - Gradient Accumulation;
    - Callbacks;
    - Metrics for train and evaluation.
    - A summary of your model, giving total params, trainable params and non-trainable params.
    - Evaluation is also done at the end of each epoch if the user specified at least x_eval in the "fit" function.
    - Predictions are also used 
- A custom datasetGenerator class that helps with big dataset that can't stay on memory.
- Model saving and loading for checkpoints.
- Model export to TensorFlow.

If you have some features you'd like this wrapper to support, be sure to add Pull requests or Discussions here on github.
